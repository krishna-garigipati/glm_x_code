import re
from typing import Any, Dict, List, Optional, Tuple

from .errors import ConfigurationError, ValidationError
from .validation import validate_output


class TemplateDecoder:
    def __init__(
        self,
        templates: List[Dict[str, Any]],
        relation_phrases: Dict[str, str],
        sentence_starters: List[str],
        fallback_cfg: Dict[str, Any],
        validation_cfg: Dict[str, Any],
        chain_render_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        fallback_type = fallback_cfg.get("type", "concatenate")
        if fallback_type != "concatenate":
            raise ConfigurationError(f"Unsupported fallback type '{fallback_type}'")
        self._templates = templates
        self._relation_phrases = relation_phrases
        self._sentence_starters = sentence_starters
        self._fallback_cfg = fallback_cfg
        self._validation_cfg = validation_cfg
        self._chain_render_cfg = chain_render_cfg or {}

    def add_template(self, intent_sequence: Tuple[int, ...], template_string: str) -> None:
        self._templates.append({"intents": list(intent_sequence), "template": template_string})

    def add_chain_template(self, chain: List[str], template_string: str) -> None:
        self._templates.append({"chain": list(chain), "template": template_string})

    def load_templates(self, definitions: List[Dict[str, Any]]) -> None:
        self._templates = definitions

    def decode(
        self,
        node_labels: List[str],
        relation_labels: List[str],
        intents: Optional[List[int]] = None,
        chain: Optional[List[str]] = None,
    ) -> Tuple[str, bool]:
        if chain is not None:
            return self._decode_chain(node_labels, relation_labels, chain)
        template = self._select_template(intents or [], node_labels, relation_labels)
        if template is None:
            return "", False
        text = self._render_template(template, node_labels, relation_labels)
        if not text:
            return "", False
        starter = self._select_sentence_starter(intents or [])
        if starter and not text.startswith(starter):
            text = f"{starter} {text}"
        try:
            validate_output(
                text=text,
                node_labels=node_labels,
                min_len=self._validation_cfg["min_output_length"],
                max_len=self._validation_cfg["max_output_length"],
                require_node_mention=self._validation_cfg["require_node_mention"],
                max_repetitive_ngrams=self._validation_cfg["max_repetitive_ngrams"],
            )
        except ValidationError:
            return "", False
        return text, True

    def _decode_chain(
        self,
        node_labels: List[str],
        relation_labels: List[str],
        chain: List[str],
    ) -> Tuple[str, bool]:
        template = self._select_chain_template(chain, node_labels, relation_labels)
        if template is not None:
            text = self._render_template(template, node_labels, relation_labels)
        else:
            text = self._render_chain_path(node_labels, relation_labels, chain)
        if not text:
            return "", False
        starter = self._select_sentence_starter(None)
        if starter and not text.startswith(starter):
            text = f"{starter} {text}"
        try:
            validate_output(
                text=text,
                node_labels=node_labels,
                min_len=self._validation_cfg["min_output_length"],
                max_len=self._validation_cfg["max_output_length"],
                require_node_mention=self._validation_cfg["require_node_mention"],
                max_repetitive_ngrams=self._validation_cfg["max_repetitive_ngrams"],
            )
        except ValidationError:
            return "", False
        return text, True

    def render_for_intents(
        self,
        node_labels: List[str],
        relation_labels: List[str],
        intents: List[int],
    ) -> Optional[str]:
        template = self._select_template(intents, node_labels, relation_labels)
        if template is None:
            return None
        text = self._render_template(template, node_labels, relation_labels)
        if not text:
            return None
        starter = self._select_sentence_starter(intents)
        if starter and not text.startswith(starter):
            text = f"{starter} {text}"
        return text

    def render_chain(
        self,
        node_labels: List[str],
        relation_labels: List[str],
        chain: List[str],
    ) -> Optional[str]:
        """Chain-driven render only (no validation gate). Returns text or None."""
        template = self._select_chain_template(chain, node_labels, relation_labels)
        if template is not None:
            text = self._render_template(template, node_labels, relation_labels)
        else:
            text = self._render_chain_path(node_labels, relation_labels, chain)
        if not text:
            return None
        starter = self._select_sentence_starter(None)
        if starter and not text.startswith(starter):
            text = f"{starter} {text}"
        return text

    def render_no_relation(self, node_labels: List[str], chain: Optional[List[str]] = None) -> str:
        """Honest answer when the extractor fell back and the walk found no relation path."""
        template = self._chain_render_cfg.get(
            "no_relation_answer",
            "I don't have a relation in my knowledge graph that fully answers this question.",
        )
        if node_labels:
            names = ", ".join(node_labels[:3])
            template = f"{template} Closest concepts I have: {names}."
        if chain:
            template = f"{template} (No {', '.join(chain)} relation found.)"
        return template

    def fallback(
        self,
        node_labels: List[str],
        relation_labels: List[str],
        intents: Optional[List[int]] = None,
        chain: Optional[List[str]] = None,
    ) -> str:
        parts: List[str] = []
        if self._fallback_cfg.get("use_intent_prefix") and intents:
            prefix = self._fallback_cfg.get("intent_prefixes", {}).get(intents[0])
            if prefix:
                parts.append(prefix)
        parts.extend(self._build_fallback_tokens(node_labels, relation_labels))
        text = self._fallback_cfg.get("separator", " ").join(parts)
        max_words = self._fallback_cfg.get("max_words", 200)
        text = " ".join(text.split()[:max_words])
        starter = self._select_sentence_starter(intents or None)
        if starter and not text.startswith(starter):
            text = f"{starter} {text}"
        validate_output(
            text=text,
            node_labels=node_labels,
            min_len=self._validation_cfg["min_output_length"],
            max_len=self._validation_cfg["max_output_length"],
            require_node_mention=self._validation_cfg["require_node_mention"],
            max_repetitive_ngrams=self._validation_cfg["max_repetitive_ngrams"],
        )
        return text

    def _select_chain_template(
        self,
        chain: List[str],
        node_labels: List[str],
        relation_labels: List[str],
    ) -> Optional[str]:
        candidates = [item for item in self._templates if item.get("chain") == list(chain)]
        for item in candidates:
            template = item.get("template", "")
            if self._render_template(template, node_labels, relation_labels):
                return template
        return None

    def _render_chain_path(
        self,
        node_labels: List[str],
        relation_labels: List[str],
        chain: List[str],
    ) -> str:
        if len(node_labels) < 2:
            return ""
        steps = []
        pair_count = min(len(node_labels) - 1, len(relation_labels))
        for i in range(pair_count):
            phrase = self._relation_phrases.get(relation_labels[i], relation_labels[i])
            steps.append(f"{node_labels[i]} {phrase} {node_labels[i + 1]}")
        if not steps:
            return ""
        connector = self._chain_render_cfg.get("step_connector", ", and ")
        text = connector.join(steps) + "."
        return text

    def _select_template(self, intents: List[int], node_labels: List[str], relation_labels: List[str]) -> Optional[str]:
        candidates = [item for item in self._templates if item.get("intents") == intents]
        for item in candidates:
            template = item.get("template", "")
            text = self._render_template(template, node_labels, relation_labels)
            if text:
                return template
        return None

    def _render_template(self, template: str, node_labels: List[str], relation_labels: List[str]) -> str:
        needed_nodes = _extract_slot_indices(template, "node")
        needed_relations = _extract_slot_indices(template, "relation")
        if needed_nodes and max(needed_nodes) >= len(node_labels):
            return ""
        if needed_relations and max(needed_relations) >= len(relation_labels):
            return ""
        rendered = template.replace("{count}", str(len(node_labels)))
        for idx in needed_nodes:
            rendered = rendered.replace(f"{{node{idx}}}", node_labels[idx])
        for idx in needed_relations:
            label = relation_labels[idx]
            phrase = self._relation_phrases.get(label, label)
            rendered = rendered.replace(f"{{relation{idx}}}", phrase)
        return rendered

    def _build_fallback_tokens(self, node_labels: List[str], relation_labels: List[str]) -> List[str]:
        tokens: List[str] = []
        pair_count = min(len(node_labels), len(relation_labels))
        for idx in range(pair_count):
            tokens.append(node_labels[idx])
            tokens.append(self._relation_phrases.get(relation_labels[idx], relation_labels[idx]))
        if len(node_labels) > pair_count:
            tokens.extend(node_labels[pair_count:])
        return tokens

    def _select_sentence_starter(self, intents: Optional[List[int]]) -> Optional[str]:
        if not self._sentence_starters:
            return None
        if not intents:
            return self._sentence_starters[0]
        index = sum(intents) % len(self._sentence_starters)
        return self._sentence_starters[index]


def _extract_slot_indices(template: str, slot: str) -> List[int]:
    pattern = rf"\{{{slot}(\d+)\}}"
    return [int(item) for item in re.findall(pattern, template)]