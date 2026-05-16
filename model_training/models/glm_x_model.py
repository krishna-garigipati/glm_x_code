import logging
import json
import yaml
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from g2p.g2p_planner import G2PPlanner
from g2p.config import G2PConfig, FFNConfig, TrainingConfig, ValidationConfig
from g2p.types import Subgraph, Plan

from model_training.models.intent_ffn import IntentFFN

logger = logging.getLogger(__name__)


class GLMXModel:
    def __init__(
        self,
        g2p_config_path: Optional[Path] = None,
        intent_ffn: Optional[IntentFFN] = None,
        label_map: Optional[Dict[int, str]] = None,
    ):
        self._intent_ffn = intent_ffn
        self._g2p_planner: Optional[G2PPlanner] = None
        self._g2p_config: Optional[G2PConfig] = None
        self._g2p_config_path = g2p_config_path
        self._label_map = label_map
        self._initialized = False
        self._trained_datasets: List[str] = []

    @property
    def intent_ffn(self) -> Optional[IntentFFN]:
        return self._intent_ffn

    @property
    def g2p_planner(self) -> Optional[G2PPlanner]:
        return self._g2p_planner

    def initialize_pipeline(self, g2p_yaml_path: Optional[Path] = None):
        config_path = g2p_yaml_path or self._g2p_config_path
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        self._g2p_config = G2PConfig.from_yaml(str(config_path))
        if self._intent_ffn is not None:
            ffn_cfg = self._g2p_config.ffn
            self._intent_ffn.config["input_dim"] = ffn_cfg.input_dim
            self._intent_ffn.config["hidden_dim"] = ffn_cfg.hidden_dim
            self._intent_ffn.config["num_layers"] = ffn_cfg.num_layers
            self._intent_ffn.config["dropout"] = ffn_cfg.dropout
            self._intent_ffn.config["classifier_input_dim"] = ffn_cfg.classifier_input_dim
            self._intent_ffn.config["classifier_hidden_dim"] = ffn_cfg.classifier_hidden_dim
            self._intent_ffn.config["classifier_num_layers"] = ffn_cfg.classifier_num_layers
            self._intent_ffn.config["output_dim"] = ffn_cfg.output_dim
            self._intent_ffn._build_body()
            self._intent_ffn._build_classifier()
            self._g2p_config.ffn = ffn_cfg
        self._g2p_planner = G2PPlanner(self._g2p_config, label_map=self._label_map)
        if self._intent_ffn is not None:
            self._g2p_planner.intent_ffn = self._intent_ffn
        self._g2p_planner.initialize()

        g2p_ffn = self._g2p_planner.intent_ffn
        if self._intent_ffn is None and g2p_ffn is not None:
            cfg = g2p_ffn.config
            new_ffn = IntentFFN(
                input_dim=cfg.input_dim,
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout,
                classifier_input_dim=cfg.classifier_input_dim,
                classifier_hidden_dim=cfg.classifier_hidden_dim,
                classifier_num_layers=cfg.classifier_num_layers,
                output_dim=cfg.output_dim,
            )
            new_ffn.load_state_dict(g2p_ffn.state_dict())
            self._intent_ffn = new_ffn
            self._g2p_planner.intent_ffn = new_ffn
        self._initialized = True
        logger.info("GLMXModel full pipeline initialized")

    def plan(self, subgraph: Subgraph) -> Plan:
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize_pipeline() first.")
        return self._g2p_planner.plan(subgraph)

    def plan_batch(self, subgraphs: List[Subgraph]) -> List[Plan]:
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize_pipeline() first.")
        return self._g2p_planner.plan_batch(subgraphs)

    def encode_subgraph(self, subgraph: Subgraph) -> np.ndarray:
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized.")
        return self._g2p_planner.encode_subgraph(subgraph)

    def inject_trained_ffn(self, intent_ffn: IntentFFN):
        self._intent_ffn = intent_ffn
        if self._g2p_planner is not None:
            self._g2p_planner.intent_ffn = intent_ffn
        if intent_ffn._trained_on_datasets:
            self._trained_datasets = intent_ffn._trained_on_datasets.copy()
        logger.info(f"Injected trained IntentFFN (trained on: {self._trained_datasets})")

    def add_trained_dataset(self, dataset_name: str):
        if dataset_name not in self._trained_datasets:
            self._trained_datasets.append(dataset_name)
        if self._intent_ffn is not None:
            self._intent_ffn.add_trained_dataset(dataset_name)

    def save(self, save_dir: Path, model_name: str = "glm_x_model.pt", metadata: Optional[Dict] = None):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_dir / model_name
        if self._intent_ffn is not None:
            self._intent_ffn.save(model_path, metadata=metadata)
        manifest = {
            "model_path": str(model_path),
            "intent_ffn_config": self._intent_ffn.config if self._intent_ffn else None,
            "trained_datasets": self._trained_datasets,
            "initialized": self._initialized,
            "metadata": metadata or {},
        }
        with open(save_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"GLMXModel saved to {save_dir}")

    @classmethod
    def load(cls, load_dir: Path, g2p_config_path: Optional[Path] = None) -> "GLMXModel":
        load_dir = Path(load_dir)
        manifest_path = load_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No manifest found in {load_dir}")
        with open(manifest_path) as f:
            manifest = json.load(f)
        model_path = load_dir / Path(manifest["model_path"]).name
        intent_ffn = None
        if model_path.exists():
            intent_ffn = IntentFFN.load(model_path)
        model = cls(g2p_config_path=g2p_config_path, intent_ffn=intent_ffn)
        model._trained_datasets = manifest.get("trained_datasets", [])
        if intent_ffn is not None:
            intent_ffn._trained_on_datasets = model._trained_datasets
        logger.info(f"GLMXModel loaded from {load_dir} (trained on: {model._trained_datasets})")
        return model

    def expand_for_new_dataset(self, new_output_dim: int = 16) -> "GLMXModel":
        if self._intent_ffn is None:
            logger.warning("No IntentFFN to expand. Creating new one.")
            self._intent_ffn = IntentFFN(output_dim=new_output_dim)
            return self
        self._intent_ffn = self._intent_ffn.expand_for_new_dataset(
            new_output_classes=new_output_dim, preserve_weights=True
        )
        if self._g2p_planner is not None:
            self._g2p_planner.intent_ffn = self._intent_ffn
        logger.info("GLMXModel expanded for new dataset training")
        return self

    def get_pipeline_components(self) -> Dict[str, Any]:
        return {
            "intent_ffn_params": self._intent_ffn.get_num_params() if self._intent_ffn else 0,
            "trained_datasets": self._trained_datasets,
            "g2p_initialized": self._initialized,
            "intent_ffn_config": self._intent_ffn.config if self._intent_ffn else None,
        }
