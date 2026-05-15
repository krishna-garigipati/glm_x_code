from typing import List


class CopyAttention:
    def __init__(self, enabled: bool, copy_probability: float, exact_match: bool) -> None:
        self.enabled = enabled
        self.copy_probability = copy_probability
        self.exact_match = exact_match

    def apply(self, text: str, node_labels: List[str], rng) -> str:
        if not self.enabled or not node_labels:
            return text
        if rng.random() > self.copy_probability:
            return text
        lowered = text.lower()
        if any(label.lower() in lowered for label in node_labels):
            return text
        selected = node_labels[int(rng.random() * len(node_labels))]
        if not self.exact_match:
            selected = selected.lower()
        return f"{selected} {text}"
