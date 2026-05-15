import numpy as np
from typing import List, Tuple, Optional
from .config import DecoderConfig


class BeamSearchDecoder:
    def __init__(self, config: DecoderConfig):
        self.config = config
        self.intent_names: Optional[dict] = None

    def set_intent_names(self, names: dict):
        self.intent_names = names

    def decode(self, logits: np.ndarray) -> Tuple[List[int], float]:
        beam_width = self.config.beam_width
        max_length = self.config.max_length
        temperature = self.config.temperature
        rep_penalty = self.config.repetition_penalty

        log_probs = logits - np.max(logits)
        log_probs = log_probs / temperature
        log_probs = log_probs - np.log(np.sum(np.exp(log_probs)))

        beams = [([], 0.0)]

        for _ in range(max_length):
            candidates = []
            for seq, score in beams:
                current_logits = log_probs.copy()

                if len(seq) > 0:
                    for t in set(seq):
                        current_logits[t] -= rep_penalty

                current_probs = np.exp(current_logits - np.max(current_logits))
                current_probs = current_probs / np.sum(current_probs)

                top_indices = np.argsort(current_probs)[-beam_width:][::-1]

                for idx in top_indices:
                    new_seq = seq + [int(idx)]
                    new_score = score + np.log(max(current_probs[idx], 1e-10))
                    candidates.append((new_seq, new_score))

            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_width]

        beams.sort(key=lambda x: x[1], reverse=True)
        best_seq, best_score = beams[0]

        if len(best_seq) > max_length:
            best_seq = best_seq[:max_length]

        confidence = float(np.exp(best_score / max(1, len(best_seq))))
        confidence = max(0.0, min(1.0, confidence))

        return best_seq, confidence
