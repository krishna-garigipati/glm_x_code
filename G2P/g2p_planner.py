import numpy as np
import logging
from typing import List, Optional, Dict, Tuple
from .types import Subgraph, Plan
from .config import G2PConfig, DecoderConfig
from .graph_to_text import GraphToTextEncoder
from .intent_ffn import IntentFFN
from .beam_search import BeamSearchDecoder
from .heuristic_planner import HeuristicPlanner

logger = logging.getLogger(__name__)


class G2PPlanner:
    INTENT_NAMES = {
        0: "define", 1: "assert_fact", 2: "explain_cause", 3: "explain_effect",
        4: "contrast", 5: "compare", 6: "list", 7: "example",
        8: "conclude", 9: "question", 10: "uncertain", 11: "clarify",
        12: "summarize", 13: "elaborate", 14: "transition", 15: "emphasize",
    }

    def __init__(self, config: G2PConfig, label_map: Optional[Dict[int, str]] = None):
        self.config = config
        self.config.validate()
        self._sentence_model = None
        self.label_map = label_map if label_map is not None else {}

        self.graph_to_text_encoder = GraphToTextEncoder(config.graph_to_text, label_map)
        self.intent_ffn = IntentFFN(config.ffn)
        self.beam_search = BeamSearchDecoder(config.decoder)
        self.heuristic_planner = HeuristicPlanner(config.mapping)
        self._initialized = False
        self._intent_embedding_cache: Dict[int, np.ndarray] = {}

    def _get_sentence_model(self):
        if self._sentence_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._sentence_model = SentenceTransformer(
                    self.config.sentence_bert.model_name,
                    device=self.config.sentence_bert.device,
                )
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. Install with: pip install sentence-transformers"
                )
        return self._sentence_model

    def initialize(self):
        if self._initialized:
            return
        _ = self._get_sentence_model()
        self._precompute_intent_embeddings()
        self._initialized = True
        logger.info("G2PPlanner initialized successfully")

    def set_label_map(self, label_map: Dict[int, str]):
        self.label_map = label_map
        self.graph_to_text_encoder.set_label_map(label_map)

    def encode_subgraph(self, subgraph: Subgraph) -> np.ndarray:
        self._validate_input(subgraph)
        text = self.graph_to_text_encoder.encode(subgraph)
        model = self._get_sentence_model()
        embedding = model.encode(
            text,
            normalize_embeddings=self.config.sentence_bert.normalize_embeddings,
            show_progress_bar=False,
        )
        return embedding.astype(np.float32)

    def plan(self, subgraph: Subgraph) -> Plan:
        self._validate_input(subgraph)

        heuristic_result = self.heuristic_planner.evaluate(subgraph)
        if heuristic_result is not None:
            intent_seq, confidence = heuristic_result
            plan = Plan(
                intent_sequence=intent_seq,
                plan_confidence=confidence,
                heuristic_fallback_used=True,
                intent_names=[self.INTENT_NAMES[i] for i in intent_seq],
            )
            logger.debug(f"Heuristic plan: {intent_seq}")
            return plan

        embedding = self.encode_subgraph(subgraph)

        logits = self.intent_ffn.predict_logits(embedding)

        logits = self._apply_allowed_intents_mask(logits)

        intent_seq, confidence = self.beam_search.decode(logits)

        if len(intent_seq) < self.config.validation.output_plan_min_length:
            intent_seq = intent_seq + [0] * (
                self.config.validation.output_plan_min_length - len(intent_seq)
            )
        if len(intent_seq) > self.config.validation.output_plan_max_length:
            intent_seq = intent_seq[:self.config.validation.output_plan_max_length]

        plan = Plan(
            intent_sequence=intent_seq,
            plan_confidence=float(confidence),
            heuristic_fallback_used=False,
            intent_names=[self.INTENT_NAMES[i] for i in intent_seq],
        )
        logger.debug(f"FFN plan: {intent_seq} (conf={confidence:.4f})")
        return plan

    def plan_batch(self, subgraphs: List[Subgraph]) -> List[Plan]:
        return [self.plan(sg) for sg in subgraphs]

    def get_plan_confidence(self, subgraph: Subgraph) -> float:
        heuristic_result = self.heuristic_planner.evaluate(subgraph)
        if heuristic_result is not None:
            return heuristic_result[1]

        embedding = self.encode_subgraph(subgraph)
        probs = self.intent_ffn.predict_probs(embedding)
        return float(np.max(probs))

    def get_intent_name(self, intent_id: int) -> str:
        return self.INTENT_NAMES.get(intent_id, f"unknown_{intent_id}")

    def intent_to_embedding(self, intent_id: int) -> np.ndarray:
        if intent_id in self._intent_embedding_cache:
            return self._intent_embedding_cache[intent_id].copy()

        description = self.config.intent_embeddings.get(intent_id, "")
        if not description:
            emb = np.zeros(self.config.sentence_bert.model_dim, dtype=np.float32)
        else:
            model = self._get_sentence_model()
            emb = model.encode(
                description,
                normalize_embeddings=self.config.sentence_bert.normalize_embeddings,
                show_progress_bar=False,
            ).astype(np.float32)

        self._intent_embedding_cache[intent_id] = emb
        return emb.copy()

    def load_heuristic_rules(self, filepath: str):
        self.heuristic_planner.load_rules(filepath)

    def train(self, training_data: List[Tuple[Subgraph, List[int]]],
              validation_data: Optional[List[Tuple[Subgraph, List[int]]]] = None) -> Dict:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        config = self.config.training

        X_list = []
        y_list = []

        for subgraph, intent_seq in training_data:
            emb = self.encode_subgraph(subgraph)
            X_list.append(emb)
            target = intent_seq[0] if intent_seq else 0
            y_list.append(target)

        X = np.stack(X_list)
        y = np.array(y_list, dtype=np.int64)

        split_idx = int(len(X) * config.train_test_split)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        if validation_data is not None:
            X_val_list = []
            y_val_list = []
            for subgraph, intent_seq in validation_data:
                emb = self.encode_subgraph(subgraph)
                X_val_list.append(emb)
                target = intent_seq[0] if intent_seq else 0
                y_val_list.append(target)
            if X_val_list:
                X_val = np.stack(X_val_list)
                y_val = np.array(y_val_list, dtype=np.int64)

        train_dataset = TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).long(),
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(y_val).long(),
        )

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

        optimizer = optim.Adam(self.intent_ffn.parameters(), lr=config.learning_rate)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        patience_counter = 0
        metrics = {"train_loss": [], "val_loss": [], "epochs_trained": 0}

        for epoch in range(config.epochs):
            self.intent_ffn.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.intent_ffn(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            self.intent_ffn.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.intent_ffn(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)

            metrics["train_loss"].append(avg_train_loss)
            metrics["val_loss"].append(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        metrics["epochs_trained"] = epoch + 1
        metrics["best_val_loss"] = best_val_loss
        logger.info(f"Training completed: {metrics['epochs_trained']} epochs, best_val_loss={best_val_loss:.6f}")
        return metrics

    def _validate_input(self, subgraph: Subgraph):
        if len(subgraph.nodes) > self.config.validation.input_subgraph_max_nodes:
            raise ValueError(
                f"Subgraph has {len(subgraph.nodes)} nodes, "
                f"exceeds max {self.config.validation.input_subgraph_max_nodes}"
            )

    def _apply_allowed_intents_mask(self, logits: np.ndarray) -> np.ndarray:
        masked = logits.copy()
        allowed = set(self.config.validation.allowed_intents)
        for i in range(logits.shape[-1]):
            if i not in allowed:
                masked[..., i] = -1e9
        return masked

    def _precompute_intent_embeddings(self):
        for intent_id in range(16):
            description = self.config.intent_embeddings.get(intent_id, "")
            if description:
                model = self._get_sentence_model()
                emb = model.encode(
                    description,
                    normalize_embeddings=self.config.sentence_bert.normalize_embeddings,
                    show_progress_bar=False,
                ).astype(np.float32)
                self._intent_embedding_cache[intent_id] = emb
