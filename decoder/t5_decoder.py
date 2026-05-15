from typing import Any, Dict, List, Optional, Tuple

from .errors import ConfigurationError
from .validation import validate_output


class T5Decoder:
    def __init__(self, t5_cfg: Dict[str, Any], validation_cfg: Dict[str, Any]) -> None:
        self._cfg = t5_cfg
        self._validation_cfg = validation_cfg
        self._model = None
        self._tokenizer = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import T5ForConditionalGeneration, T5Tokenizer
        except Exception as exc:
            raise ConfigurationError("transformers is required for t5 decoder") from exc
        model_name = self._cfg["model_name"]
        self._tokenizer = T5Tokenizer.from_pretrained(model_name)
        self._model = T5ForConditionalGeneration.from_pretrained(model_name)

    def finetune(self, examples: List[Tuple[str, str]]) -> None:
        ft_cfg = self._cfg.get("finetune", {})
        learning_rate = ft_cfg.get("learning_rate", 3e-5)
        batch_size = ft_cfg.get("batch_size", 8)
        epochs = ft_cfg.get("epochs", 3)
        warmup_steps = ft_cfg.get("warmup_steps", 500)
        weight_decay = ft_cfg.get("weight_decay", 0.01)
        self._load_model()
        try:
            import torch
            from torch.utils.data import DataLoader, Dataset
        except Exception as exc:
            raise ConfigurationError("torch is required for fine-tuning") from exc

        class _FTDataset(Dataset):
            def __init__(self, pairs, tokenizer, max_input, max_output):
                self._pairs = pairs
                self._tokenizer = tokenizer
                self._max_input = max_input
                self._max_output = max_output

            def __len__(self):
                return len(self._pairs)

            def __getitem__(self, idx):
                prompt_text, target = self._pairs[idx]
                inputs = self._tokenizer(
                    prompt_text,
                    max_length=self._max_input,
                    truncation=True,
                    return_tensors="pt",
                )
                labels = self._tokenizer(
                    target,
                    max_length=self._max_output,
                    truncation=True,
                    return_tensors="pt",
                )
                return {
                    "input_ids": inputs.input_ids[0],
                    "attention_mask": inputs.attention_mask[0],
                    "labels": labels.input_ids[0],
                }

        dataset = _FTDataset(examples, self._tokenizer, self._cfg["max_input_length"], self._cfg["max_output_length"])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, (step + 1) / max(warmup_steps, 1)),
        )
        self._model.train()
        for _ in range(epochs):
            for batch in dataloader:
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]
                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

    def decode(self, prompt: str, node_labels: List[str]) -> Tuple[str, float]:
        self._load_model()
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self._cfg["max_input_length"],
            truncation=True,
        )
        outputs = self._model.generate(
            **inputs,
            max_length=self._cfg["max_output_length"],
            num_beams=self._cfg["num_beams"],
            temperature=self._cfg["temperature"],
            top_p=self._cfg["top_p"],
            repetition_penalty=self._cfg["repetition_penalty"],
            do_sample=self._cfg["do_sample"],
            output_scores=True,
            return_dict_in_generate=True,
        )
        text = self._tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        confidence = _compute_confidence(outputs.scores)
        validate_output(
            text=text,
            node_labels=node_labels,
            min_len=self._validation_cfg["min_output_length"],
            max_len=self._validation_cfg["max_output_length"],
            require_node_mention=self._validation_cfg["require_node_mention"],
            max_repetitive_ngrams=self._validation_cfg["max_repetitive_ngrams"],
        )
        return text, confidence

    def score(self, prompt: str, text: str, node_labels: List[str]) -> float:
        self._load_model()
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self._cfg["max_input_length"],
            truncation=True,
        )
        labels = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=self._cfg["max_output_length"],
            truncation=True,
        ).input_ids
        outputs = self._model(**inputs, labels=labels)
        logits = outputs.logits
        pad_token_id = self._tokenizer.pad_token_id
        confidence = _confidence_from_logits(logits, labels, 0 if pad_token_id is None else pad_token_id)
        validate_output(
            text=text,
            node_labels=node_labels,
            min_len=self._validation_cfg["min_output_length"],
            max_len=self._validation_cfg["max_output_length"],
            require_node_mention=self._validation_cfg["require_node_mention"],
            max_repetitive_ngrams=self._validation_cfg["max_repetitive_ngrams"],
        )
        return confidence


def _compute_confidence(scores: Optional[List[Any]]) -> float:
    if not scores:
        return 0.0
    try:
        import torch
    except Exception:
        return 0.0
    total = 0.0
    count = 0
    for step in scores:
        probs = torch.softmax(step, dim=-1)
        values = probs.max(dim=-1).values
        total += float(values.mean().item())
        count += 1
    if count == 0:
        return 0.0
    return total / count


def _confidence_from_logits(logits: Any, labels: Any, pad_token_id: int) -> float:
    try:
        import torch
    except Exception:
        return 0.0
    log_probs = torch.log_softmax(logits, dim=-1)
    labels = labels.clone()
    labels[labels == -100] = pad_token_id
    gathered = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    mask = labels != pad_token_id
    if mask.sum().item() == 0:
        return 0.0
    avg_log_prob = gathered[mask].mean().item()
    return float(torch.exp(torch.tensor(avg_log_prob)).item())
