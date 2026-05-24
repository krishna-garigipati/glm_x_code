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
                    reject_patterns=self._validation_cfg.get("reject_patterns", None),
                )
            except Exception:
                return text, False
            return text, True
        except Exception as e:
            logger.warning("T5 generation failed: %s", e)
            return "", False

    def fine_tune(
        self,
        train_inputs: List[str],
        train_targets: List[str],
        val_inputs: Optional[List[str]] = None,
        val_targets: Optional[List[str]] = None,
        learning_rate: float = 3e-4,
        epochs: int = 25,
        batch_size: int = 8,
        output_dir: str = "./finetuned_t5",
        use_early_stopping: bool = True,
        patience: int = 5,
    ) -> Dict[str, Any]:
        if not self._initialized:
            self.initialize()
        from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
        import torch
        from torch.utils.data import Dataset

        class T5Dataset(Dataset):
            def __init__(self, inputs, targets, tokenizer, max_in, max_out):
                self.inputs = inputs
                self.targets = targets
                self.tokenizer = tokenizer
                self.max_in = max_in
                self.max_out = max_out

            def __len__(self):
                return len(self.inputs)

            def __getitem__(self, idx):
                inp = self.inputs[idx]
                tgt = self.targets[idx]
                model_inputs = self.tokenizer(
                    inp, truncation=True, padding="max_length",
                    max_length=self.max_in, return_tensors="pt",
                )
                labels = self.tokenizer(
                    tgt, truncation=True, padding="max_length",
                    max_length=self.max_out, return_tensors="pt",
                )
                item = {k: v.squeeze(0) for k, v in model_inputs.items()}
                item["labels"] = labels["input_ids"].squeeze(0)
                item["labels"][item["labels"] == self.tokenizer.pad_token_id] = -100
                return item

        train_dataset = T5Dataset(train_inputs, train_targets, self._tokenizer,
                                   self._max_input_length, self._max_output_length)
        val_dataset = None
        if val_inputs and val_targets:
            val_dataset = T5Dataset(val_inputs, val_targets, self._tokenizer,
                                     self._max_input_length, self._max_output_length)

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            evaluation_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            predict_with_generate=True,
            generation_max_length=self._max_output_length,
            generation_num_beams=self._num_beams,
            logging_dir=f"{output_dir}/logs",
            logging_strategy="epoch",
            report_to="none",
        )

        data_collator = DataCollatorForSeq2Seq(self._tokenizer, model=self._model)
        trainer = Seq2SeqTrainer(
            model=self._model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self._tokenizer,
            data_collator=data_collator,
        )

        train_result = trainer.train()
        eval_metrics = trainer.evaluate() if val_dataset else {}
        trainer.save_model(output_dir)
        self._tokenizer.save_pretrained(output_dir)
        logger.info(f"Fine-tuned T5 model saved to {output_dir}")

        metrics = {
            "train_loss": round(float(train_result.metrics.get("train_loss", 0)), 6),
            "eval_loss": round(float(eval_metrics.get("eval_loss", 0)), 6) if eval_metrics else None,
            "epochs": int(train_result.metrics.get("epoch", epochs)),
            "training_time": round(train_result.metrics.get("train_runtime", 0), 2),
        }
        return metrics

    def save_finetuned(self, output_dir: str):
        if self._model is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        from pathlib import Path
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(output_dir)
        self._tokenizer.save_pretrained(output_dir)
        logger.info(f"Fine-tuned model saved to {output_dir}")

    def load_finetuned(self, model_dir: str):
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        self._model = T5ForConditionalGeneration.from_pretrained(model_dir)
        self._tokenizer = T5Tokenizer.from_pretrained(model_dir)
        self._model.eval()
        self._initialized = True
        logger.info(f"Fine-tuned model loaded from {model_dir}")

    def fallback(
        self,
        node_labels: List[str],
        relation_labels: List[str],
        intents: List[int],
    ) -> str:
        separator = self._fallback_cfg.get("separator", " ")
        text = separator.join(node_labels[:5])
        return text
