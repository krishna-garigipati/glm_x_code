import logging
from typing import Any, Dict, List, Optional, Tuple

from .errors import ConfigurationError
from .validation import validate_output

logger = logging.getLogger(__name__)


class T5Decoder:
    def __init__(
        self,
        model_name: str = "t5-small",
        max_input_length: int = 512,
        max_output_length: int = 128,
        num_beams: int = 4,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        do_sample: bool = True,
        fallback_cfg: Optional[Dict[str, Any]] = None,
        validation_cfg: Optional[Dict[str, Any]] = None,
    ):
        self._model_name = model_name
        self._max_input_length = max_input_length
        self._max_output_length = max_output_length
        self._num_beams = num_beams
        self._temperature = temperature
        self._top_p = top_p
        self._repetition_penalty = repetition_penalty
        self._do_sample = do_sample
        self._fallback_cfg = fallback_cfg or {}
        self._validation_cfg = validation_cfg or {}
        self._model = None
        self._tokenizer = None
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return
        try:
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            self._tokenizer = T5Tokenizer.from_pretrained(self._model_name)
            self._model = T5ForConditionalGeneration.from_pretrained(self._model_name)
            self._model.eval()
            self._initialized = True
            logger.info("T5Decoder initialized with %s", self._model_name)
        except ImportError:
            raise ConfigurationError("transformers is required. Install with: pip install transformers")
        except Exception as e:
            raise ConfigurationError(f"Failed to load T5 model {self._model_name}: {e}")

    def _build_input_text(
        self,
        node_labels: List[str],
        relation_labels: List[str],
        intents: List[int],
    ) -> str:
        intent_names = {
            0: "define", 1: "assert", 2: "explain cause", 3: "explain effect",
            4: "contrast", 5: "compare", 6: "list", 7: "example",
            8: "conclude", 9: "question", 10: "uncertain", 11: "clarify",
            12: "summarize", 13: "elaborate", 14: "transition", 15: "emphasize",
        }
        intent_str = " -> ".join(intent_names.get(i, f"intent_{i}") for i in intents)
        path_str = " | ".join(
            f"{node_labels[i]}"
            + (f" [{relation_labels[i]}]" if i < len(relation_labels) else "")
            for i in range(len(node_labels))
        )
        return f"generate: intent={intent_str} path={path_str}"

    def decode(
        self,
        node_labels: List[str],
        relation_labels: List[str],
        intents: List[int],
    ) -> Tuple[str, bool]:
        if not self._initialized:
            try:
                self.initialize()
            except ConfigurationError:
                return "", False
        if not node_labels:
            return "", False
        input_text = self._build_input_text(node_labels, relation_labels, intents)
        try:
            inputs = self._tokenizer(
                input_text,
                return_tensors="pt",
                max_length=self._max_input_length,
                truncation=True,
                padding=True,
            )
            outputs = self._model.generate(
                **inputs,
                max_length=self._max_output_length,
                num_beams=self._num_beams,
                temperature=self._temperature,
                top_p=self._top_p,
                repetition_penalty=self._repetition_penalty,
                do_sample=self._do_sample,
            )
            text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            if not text or len(text.strip()) < 3:
                return "", False
            try:
                validate_output(
                    text=text,
                    node_labels=node_labels,
                    min_len=self._validation_cfg.get("min_output_length", 3),
                    max_len=self._validation_cfg.get("max_output_length", 500),
                    require_node_mention=self._validation_cfg.get("require_node_mention", False),
                    max_repetitive_ngrams=self._validation_cfg.get("max_repetitive_ngrams", 0),
                )
            except Exception:
                return text, False
            return text, True
        except Exception as e:
            logger.warning("T5 generation failed: %s", e)
            return "", False

    def fallback(
        self,
        node_labels: List[str],
        relation_labels: List[str],
        intents: List[int],
    ) -> str:
        separator = self._fallback_cfg.get("separator", " ")
        text = separator.join(node_labels[:5])
        return text
