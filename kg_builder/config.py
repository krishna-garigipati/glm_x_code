from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class KGBuilderConfig:
    spaCy_model: str = "en_core_web_sm"
    sbert_model: str = "BAAI/bge-small-en-v1.5"
    sbert_dim: int = 384

    ner_labels: List[str] = field(default_factory=lambda: [
        "PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "NORP",
    ])
    min_np_length: int = 1
    min_entity_freq: int = 1

    cascade_levels: List[str] = field(default_factory=lambda: ["rules", "spacy_llm"])
    enable_spacy_llm: bool = False
    llm_model: str = "phi-3-mini"

    embed_merge_threshold: float = 0.92
    triple_coherence_threshold: float = 0.65

    string_similarity_threshold: float = 0.85
    embed_similarity_threshold: float = 0.72
    graph_neighbor_overlap_threshold: float = 0.4

    pattern_weight: float = 0.5
    resolution_weight: float = 0.3
    frequency_weight: float = 0.2
    min_confidence: float = 0.2

    batch_size: int = 64
    max_workers: int = 2
    verbose: bool = True
    streaming: bool = True
