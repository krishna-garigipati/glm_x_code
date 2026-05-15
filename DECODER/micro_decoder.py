import hashlib
import logging
import os
import threading
import time
from typing import Any, Dict, List, Tuple
import random

from .config_loader import load_config
from .copy_attention import CopyAttention
from .errors import ConfigurationError, ValidationError
from .models import Answer
from .template_decoder import TemplateDecoder
from .t5_decoder import T5Decoder
from .hybrid_decoder import HybridDecoder
from .validation import validate_answer, validate_output


class MicroDecoder:
    def __init__(self, config_path: str = None) -> None:
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "config_decoder.yaml")
        self._config = load_config(config_path)
        self._lock = threading.RLock()
        self._logger = logging.getLogger(self.__class__.__name__)
        self._mode = self._config["mode"]
        self._fallback_mode = self._config.get("fallback_mode", "template_fallback")
        self._validation_cfg = self._config["validation"]
        self._template_decoder = TemplateDecoder(
            templates=self._config["templates"]["definitions"],
            relation_phrases=self._config["templates"]["relation_phrases"],
            sentence_starters=self._config["templates"]["sentence_starters"],
            fallback_cfg=self._config["fallback"],
            validation_cfg=self._validation_cfg,
        )
        self._t5_decoder = T5Decoder(self._config["t5"], self._validation_cfg)
        self._hybrid_decoder = HybridDecoder(self._template_decoder, self._t5_decoder)
        copy_cfg = self._config.get("copy_attention", {})
        copy_source = copy_cfg.get("source", "walk_path_labels")
        if copy_source != "walk_path_labels":
            raise ConfigurationError(f"Unsupported copy_attention source: {copy_source}")
        self._copy_attention = CopyAttention(
            enabled=copy_cfg.get("enabled", False),
            copy_probability=copy_cfg.get("copy_probability", 0.0),
            exact_match=copy_cfg.get("exact_match", True),
        )

    def decode(self, walk: Any, plan: Any) -> Answer:
        with self._lock:
            node_labels, relation_labels = _extract_walk_labels(walk)
            intents = _extract_intents(plan)
            mode = self._normalize_mode(self._mode)
            generation_method = mode
            if mode == "template":
                text, ok = self._template_decoder.decode(node_labels, relation_labels, intents)
                if not ok:
                    if self._fallback_mode != "template_fallback":
                        raise ConfigurationError(f"Unknown fallback mode: {self._fallback_mode}")
                    text = self._template_decoder.fallback(node_labels, relation_labels, intents)
                    generation_method = "fallback"
                text = self._apply_copy_attention(text, node_labels)
                validate_output(
                    text=text,
                    node_labels=node_labels,
                    min_len=self._validation_cfg["min_output_length"],
                    max_len=self._validation_cfg["max_output_length"],
                    require_node_mention=self._validation_cfg["require_node_mention"],
                    max_repetitive_ngrams=self._validation_cfg["max_repetitive_ngrams"],
                )
                confidence = self.get_confidence(walk, plan, text)
                nodes_mentioned = _extract_nodes_mentioned(text, walk)
                answer = Answer(
                    text=text,
                    confidence=confidence,
                    intent_used=intents[0] if intents else 0,
                    nodes_mentioned=nodes_mentioned,
                    generation_method=generation_method,
                    walk_used=walk,
                    timestamp=time.time(),
                )
                validate_answer(answer)
                return answer
            if mode == "t5":
                prompt = _build_prompt(node_labels, relation_labels, intents)
                text, _ = self._t5_decoder.decode(prompt, node_labels)
                text = self._apply_copy_attention(text, node_labels)
                validate_output(
                    text=text,
                    node_labels=node_labels,
                    min_len=self._validation_cfg["min_output_length"],
                    max_len=self._validation_cfg["max_output_length"],
                    require_node_mention=self._validation_cfg["require_node_mention"],
                    max_repetitive_ngrams=self._validation_cfg["max_repetitive_ngrams"],
                )
                confidence = self.get_confidence(walk, plan, text)
                nodes_mentioned = _extract_nodes_mentioned(text, walk)
                answer = Answer(
                    text=text,
                    confidence=confidence,
                    intent_used=intents[0] if intents else 0,
                    nodes_mentioned=nodes_mentioned,
                    generation_method=generation_method,
                    walk_used=walk,
                    timestamp=time.time(),
                )
                validate_answer(answer)
                return answer
            if mode == "hybrid":
                prompt = _build_prompt(node_labels, relation_labels, intents)
                text, used_template, confidence = self._hybrid_decoder.decode(
                    node_labels, relation_labels, intents, prompt
                )
                text = self._apply_copy_attention(text, node_labels)
                validate_output(
                    text=text,
                    node_labels=node_labels,
                    min_len=self._validation_cfg["min_output_length"],
                    max_len=self._validation_cfg["max_output_length"],
                    require_node_mention=self._validation_cfg["require_node_mention"],
                    max_repetitive_ngrams=self._validation_cfg["max_repetitive_ngrams"],
                )
                confidence = self.get_confidence(walk, plan, text)
                nodes_mentioned = _extract_nodes_mentioned(text, walk)
                generation_hint = "template" if used_template else "t5"
                answer = Answer(
                    text=text,
                    confidence=confidence,
                    intent_used=intents[0] if intents else 0,
                    nodes_mentioned=nodes_mentioned,
                    generation_method=generation_hint,
                    walk_used=walk,
                    timestamp=time.time(),
                )
                validate_answer(answer)
                return answer
            raise ConfigurationError(f"Unknown mode: {self._mode}")

    def decode_batch(self, walks: List[Any], plans: List[Any]) -> List[Answer]:
        if len(walks) != len(plans):
            raise ValueError("walks and plans length mismatch")
        results: List[Answer] = []
        for walk, plan in zip(walks, plans):
            results.append(self.decode(walk, plan))
        return results

    def set_mode(self, mode: str) -> None:
        with self._lock:
            self._mode = mode

    def get_confidence(self, walk: Any, plan: Any, generated_text: str) -> float:
        node_labels, relation_labels = _extract_walk_labels(walk)
        intents = _extract_intents(plan)
        mode = self._normalize_mode(self._mode)
        template_text = self._template_decoder.render_for_intents(
            node_labels, relation_labels, intents
        )
        if mode == "template":
            return 1.0 if template_text == generated_text else 0.0
        prompt = _build_prompt(node_labels, relation_labels, intents)
        if mode == "t5":
            return self._t5_decoder.score(prompt, generated_text, node_labels)
        if mode == "hybrid":
            if template_text == generated_text:
                return 1.0
            return self._t5_decoder.score(prompt, generated_text, node_labels)
        return 0.0

    def add_template(self, intent_sequence: Tuple[int, ...], template_string: str) -> None:
        with self._lock:
            self._template_decoder.add_template(intent_sequence, template_string)

    def load_templates(self, filepath: str) -> None:
        with self._lock:
            data = load_config(filepath)
            definitions = data.get("definitions")
            if not isinstance(definitions, list):
                raise ConfigurationError("Invalid template definitions")
            self._template_decoder.load_templates(definitions)

    def _apply_copy_attention(self, text: str, node_labels: List[str]) -> str:
        rng = random.Random(_stable_seed(node_labels))
        return self._copy_attention.apply(text, node_labels, rng)

    def _normalize_mode(self, mode: str) -> str:
        if mode == "t5_small":
            return "t5"
        return mode


def _extract_nodes_mentioned(text: str, walk: Any) -> List[int]:
    node_labels = _get_attr(walk, "path_labels")
    path = _get_attr(walk, "path")
    if node_labels is None or path is None:
        return []
    lowered = text.lower()
    mentioned: List[int] = []
    for label, node_id in zip(node_labels, path):
        if isinstance(label, str) and label.lower() in lowered:
            mentioned.append(node_id)
    return mentioned


def _extract_walk_labels(walk: Any) -> Tuple[List[str], List[str]]:
    node_labels = _get_attr(walk, "path_labels")
    if node_labels is None:
        node_labels = _get_attr(walk, "walk_path_labels")
    relation_labels = _get_attr(walk, "path_edges")
    if relation_labels is None:
        relation_labels = _get_attr(walk, "walk_path_relations")
    if node_labels is None:
        raise ValidationError("path_labels missing on walk")
    if relation_labels is None:
        relation_labels = []
    return list(node_labels), list(relation_labels)


def _extract_intents(plan: Any) -> List[int]:
    intents = _get_attr(plan, "intent_sequence")
    if intents is None:
        intents = _get_attr(plan, "intents")
    if intents is None:
        raise ValidationError("intent_sequence missing on plan")
    return list(intents)


def _get_attr(obj: Any, name: str):
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name)
    return None


def _build_prompt(node_labels: List[str], relation_labels: List[str], intents: List[int]) -> str:
    return f"intents: {intents} | nodes: {node_labels} | relations: {relation_labels}"


def _stable_seed(node_labels: List[str]) -> int:
    digest = hashlib.blake2b("|".join(node_labels).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)
