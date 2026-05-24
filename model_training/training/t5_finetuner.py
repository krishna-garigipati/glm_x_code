import logging
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


class T5FineTuner:
    def __init__(
        self,
        model_name: str = "t5-small",
        output_dir: Optional[str] = None,
        max_input_length: int = 512,
        max_output_length: int = 128,
        num_beams: int = 4,
        learning_rate: float = 3e-4,
        batch_size: int = 8,
        gradient_accumulation_steps: int = 2,
        epochs: int = 25,
        early_stopping_patience: int = 5,
        warmup_steps: int = 100,
        weight_decay: float = 0.01,
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir) if output_dir else None
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.num_beams = num_beams
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay

        self._model = None
        self._tokenizer = None
        self._metrics: Dict[str, Any] = {}

    def _lazy_init(self):
        if self._model is not None:
            return
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        self._tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self._model = T5ForConditionalGeneration.from_pretrained(self.model_name)

    def _build_input_target(
        self, edge_triples: List[Tuple[str, str, str]], intent_label: str
    ) -> Tuple[str, str]:
        path_str = " | ".join(f"{s} [{r}] {o}" for s, r, o in edge_triples)
        input_text = f"{intent_label}: {path_str}"
        if edge_triples:
            target = f"{edge_triples[0][0]} {edge_triples[0][1]} {edge_triples[0][2]}."
        else:
            target = ""
        return input_text, target

    def prepare_dataset(
        self,
        edge_triples_list: List[List[Tuple[str, str, str]]],
        intent_labels: List[str],
        val_split: float = 0.1,
        test_split: float = 0.1,
    ) -> Tuple:
        self._lazy_init()
        inputs, targets = [], []
        for triples, label in zip(edge_triples_list, intent_labels):
            inp, tgt = self._build_input_target(triples, label)
            if inp and tgt:
                inputs.append(inp)
                targets.append(tgt)

        n = len(inputs)
        n_val = int(n * val_split)
        n_test = int(n * test_split)
        n_train = n - n_val - n_test

        split = {
            "train": (inputs[:n_train], targets[:n_train]),
            "val": (inputs[n_train:n_train+n_val], targets[n_train:n_train+n_val]),
            "test": (inputs[n_train+n_val:], targets[n_train+n_val:]),
        }
        return split

    def _tokenize(self, texts: List[str], max_len: int):
        return self._tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=max_len, return_tensors="pt",
        )

    def train(
        self,
        train_inputs: List[str],
        train_targets: List[str],
        val_inputs: List[str],
        val_targets: List[str],
        test_inputs: Optional[List[str]] = None,
        test_targets: Optional[List[str]] = None,
        is_continual: bool = False,
    ) -> Dict[str, Any]:
        self._lazy_init()
        from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
        from transformers import DataCollatorForSeq2Seq
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
                                   self.max_input_length, self.max_output_length)
        val_dataset = T5Dataset(val_inputs, val_targets, self._tokenizer,
                                 self.max_input_length, self.max_output_length)

        lr = self.learning_rate * 0.5 if is_continual else self.learning_rate
        run_name = f"t5_finetune_{int(time.time())}"
        output_dir = str(self.output_dir / run_name) if self.output_dir else f"./t5_{run_name}"

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            learning_rate=lr,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            num_train_epochs=self.epochs,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            predict_with_generate=True,
            generation_max_length=self.max_output_length,
            generation_num_beams=self.num_beams,
            warmup_steps=self.warmup_steps,
            weight_decay=self.weight_decay,
            logging_dir=f"{output_dir}/logs",
            logging_strategy="epoch",
            report_to="none",
            fp16=False,
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
        eval_metrics = trainer.evaluate()

        self._metrics = {
            "t5_model_name": self.model_name,
            "is_continual_training": is_continual,
            "train_samples": len(train_inputs),
            "val_samples": len(val_inputs),
            "epochs_trained": int(train_result.metrics.get("epoch", self.epochs)),
            "best_val_loss": round(float(eval_metrics.get("eval_loss", 0)), 6),
            "training_time_seconds": round(train_result.metrics.get("train_runtime", 0), 2),
            "learning_rate": lr,
            "batch_size": self.batch_size,
        }

        if self.output_dir:
            self._model.save_pretrained(str(self.output_dir / "t5_decoder"))
            self._tokenizer.save_pretrained(str(self.output_dir / "t5_decoder"))
            logger.info(f"T5 model saved to {self.output_dir / 't5_decoder'}")

        logger.info(
            f"T5 fine-tuning complete: {self._metrics['epochs_trained']} epochs, "
            f"best_val_loss={self._metrics['best_val_loss']:.6f}"
        )
        return self._metrics

    def get_model_path(self) -> Optional[str]:
        if self.output_dir:
            path = self.output_dir / "t5_decoder"
            if path.exists():
                return str(path)
        return None

    def get_metrics(self) -> Dict[str, Any]:
        return dict(self._metrics)
