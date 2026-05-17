from typing import Any, Iterable, List, Optional

from .errors import ValidationError


def validate_output(
    text: str,
    node_labels: List[str],
    min_len: int,
    max_len: int,
    require_node_mention: bool,
    max_repetitive_ngrams: int,
    reject_patterns: Optional[List[str]] = None,
) -> None:
    char_count = len(text)
    if char_count < min_len or char_count > max_len:
        raise ValidationError("Output length out of bounds")
    if require_node_mention and node_labels:
        lowered = text.lower()
        if not any(label.lower() in lowered for label in node_labels):
            raise ValidationError("No node mention in output")
    if max_repetitive_ngrams > 0:
        _check_repetitive_ngrams(text, max_repetitive_ngrams)
    if reject_patterns:
        lowered = text.lower()
        for pat in reject_patterns:
            if pat.lower() in lowered:
                raise ValidationError(f"Output contains rejected pattern: {pat}")


def _check_repetitive_ngrams(text: str, max_n: int) -> None:
    tokens = text.split()
    if not tokens:
        return
    for n in range(1, max_n + 1):
        if len(tokens) < n + 1:
            continue
        seen = set()
        for idx in range(len(tokens) - n + 1):
            gram = " ".join(tokens[idx:idx + n])
            if gram in seen:
                raise ValidationError("Repetitive ngrams detected")
            seen.add(gram)


VALID_GENERATION_METHODS = {"template", "t5", "hybrid", "fallback"}


def validate_answer(answer: Any) -> None:
    if not hasattr(answer, "confidence") or not (0.0 <= answer.confidence <= 1.0):
        raise ValidationError("confidence must be in [0.0, 1.0]")
    if not hasattr(answer, "intent_used") or not (0 <= answer.intent_used <= 15):
        raise ValidationError("intent_used must be in [0, 15]")
    if (
        hasattr(answer, "nodes_mentioned")
        and answer.nodes_mentioned is not None
        and hasattr(answer, "walk_used")
        and answer.walk_used is not None
    ):
        path = answer.walk_used.path if hasattr(answer.walk_used, "path") else None
        if path is not None:
            for nid in answer.nodes_mentioned:
                if nid not in path:
                    raise ValidationError(f"node {nid} not in walk_used.path")
    if not hasattr(answer, "generation_method") or answer.generation_method not in VALID_GENERATION_METHODS:
        raise ValidationError("generation_method must be in [template, t5, hybrid, fallback]")
