from typing import Any, Dict, List, Tuple

from .t5_decoder import T5Decoder
from .template_decoder import TemplateDecoder


class HybridDecoder:
    def __init__(
        self,
        template_decoder: TemplateDecoder,
        t5_decoder: T5Decoder,
    ) -> None:
        self._template_decoder = template_decoder
        self._t5_decoder = t5_decoder

    def decode(
        self,
        node_labels: List[str],
        relation_labels: List[str],
        intents: List[int],
        prompt: str,
    ) -> Tuple[str, bool, float]:
        text, ok = self._template_decoder.decode(node_labels, relation_labels, intents)
        if ok:
            return text, True, 1.0
        text, confidence = self._t5_decoder.decode(prompt, node_labels)
        return text, False, confidence
