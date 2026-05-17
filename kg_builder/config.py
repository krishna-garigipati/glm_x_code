from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class KGBuilderConfig:
    spaCy_model: str = "en_core_web_sm"
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    sbert_dim: int = 384

    ner_labels: List[str] = field(default_factory=lambda: [
        "PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "NORP",
    ])
    min_np_length: int = 1
    min_entity_freq: int = 1

    cascade_levels: List[str] = field(default_factory=lambda: ["rules", "spacy_llm"])
    enable_spacy_llm: bool = False
    llm_model: str = "phi-3-mini"
    llm_relation_types: List[str] = field(default_factory=lambda: [
        "is_a", "has_property", "causes", "part_of", "antonym", "associated_with",
    ])

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

    relation_map: Dict[str, str] = field(default_factory=lambda: {
        "is": "is_a", "are": "is_a", "was": "is_a", "were": "is_a", "became": "is_a",
        "has": "has_property", "have": "has_property", "had": "has_property",
        "contains": "has_property", "consists_of": "part_of",
        "causes": "causes", "caused": "causes", "leads_to": "causes",
        "located_in": "associated_with", "lives_in": "associated_with",
        "works_at": "associated_with", "founded": "associated_with",
        "created": "associated_with", "discovered": "associated_with",
        "default": "associated_with",
    })

    antonym_triggers: List[str] = field(default_factory=lambda: [
        "opposite", "unlike", "contrary", "reverse", "inverse",
        "versus", "vs", "antonym",
    ])
    synonym_triggers: List[str] = field(default_factory=lambda: [
        "also_known_as", "aka", "alias", "same_as", "synonym",
        "also_called", "otherwise_known",
    ])
