from typing import Iterable
import numpy as np


def quantize_embedding(
    embedding: np.ndarray,
    embedding_dim: int,
    logical_min: float,
    logical_max: float,
) -> np.ndarray:
    if embedding.shape != (embedding_dim,):
        raise ValueError(f"Embedding must be length {embedding_dim}")
    if embedding.dtype == np.int8:
        return embedding
    if not np.issubdtype(embedding.dtype, np.floating):
        raise ValueError("Embedding must be float or int8")
    int8_max = float(np.iinfo(np.int8).max)
    scale = int8_max / max(abs(logical_min), logical_max)
    clipped = np.clip(embedding.astype(np.float32), logical_min, logical_max)
    return np.rint(clipped * scale).astype(np.int8)


def dequantize_embedding(
    embedding: np.ndarray,
    embedding_dim: int,
    logical_min: float,
    logical_max: float,
) -> np.ndarray:
    if embedding.shape != (embedding_dim,):
        raise ValueError(f"Embedding must be length {embedding_dim}")
    if embedding.dtype != np.int8:
        raise ValueError("Embedding must be int8")
    int8_max = float(np.iinfo(np.int8).max)
    scale = int8_max / max(abs(logical_min), logical_max)
    return embedding.astype(np.float32) / scale


def validate_label(label: str, max_length: int) -> None:
    if not isinstance(label, str):
        raise ValueError("Label must be a string")
    if len(label) > max_length:
        raise ValueError("Label exceeds max length")


def validate_node_type(node_type: str, allowed: Iterable[str]) -> None:
    if node_type not in allowed:
        raise ValueError("Invalid node type")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("Mismatched embedding shapes")
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)
