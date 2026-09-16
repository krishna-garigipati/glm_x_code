import dataclasses
import yaml
from typing import Dict, List
from dataclasses import dataclass, field


@dataclass
class SentenceBERTConfig:
    model_name: str = "BAAI/bge-small-en-v1.5"
    model_dim: int = 384
    pooling: str = "mean"
    normalize_embeddings: bool = True
    device: str = "cpu"
    batch_size: int = 32


@dataclass
class RelationExtractionConfig:
    similarity_threshold: float = 0.35
    max_chain_length: int = 3
    collapse_max: int = 3
    default_chain: List[str] = field(default_factory=lambda: ["has_property"])
    clause_split: List[str] = field(
        default_factory=lambda: ["which", "that", "what", "how", "why", "when", "where", "because", "since", ",", " and "]
    )
    relation_variants: Dict[str, List[str]] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, raw: Dict) -> "RelationExtractionConfig":
        field_spec = cls.__dataclass_fields__["clause_split"]
        default_split = field_spec.default
        if default_split is dataclasses.MISSING:
            default_split = field_spec.default_factory()
        return cls(
            similarity_threshold=float(raw.get("similarity_threshold", 0.35)),
            max_chain_length=int(raw.get("max_chain_length", 3)),
            collapse_max=int(raw.get("collapse_max", 3)),
            default_chain=[str(r) for r in raw.get("default_chain", ["has_property"])],
            clause_split=[str(p) for p in raw.get("clause_split", default_split)],
            relation_variants={
                str(k): [str(v) for v in vals]
                for k, vals in raw.get("relation_variants", {}).items()
            },
        )

    def validate(self) -> None:
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be in [0.0, 1.0]")
        if not (1 <= self.max_chain_length <= 8):
            raise ValueError("max_chain_length must be in [1, 8]")
        if not self.relation_variants:
            raise ValueError("relation_variants must be non-empty")
        for relation in self.default_chain:
            if relation not in self.relation_variants:
                raise ValueError(f"default_chain relation '{relation}' missing from relation_variants")


@dataclass
class ValidationConfig:
    input_subgraph_max_nodes: int = 1000
    output_plan_max_length: int = 8
    output_plan_min_length: int = 1


@dataclass
class G2PConfig:
    sentence_bert: SentenceBERTConfig = field(default_factory=SentenceBERTConfig)
    extraction: RelationExtractionConfig = field(default_factory=RelationExtractionConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "G2PConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}

        config = cls()

        sb = raw.get("sentence_bert", {})
        config.sentence_bert = SentenceBERTConfig(
            model_name=sb.get("model_name", config.sentence_bert.model_name),
            model_dim=sb.get("model_dim", config.sentence_bert.model_dim),
            pooling=sb.get("pooling", config.sentence_bert.pooling),
            normalize_embeddings=sb.get("normalize_embeddings", config.sentence_bert.normalize_embeddings),
            device=sb.get("device", config.sentence_bert.device),
            batch_size=sb.get("batch_size", config.sentence_bert.batch_size),
        )

        ex = raw.get("extraction", {})
        config.extraction = RelationExtractionConfig.from_yaml(ex)

        vl = raw.get("validation", {})
        config.validation = ValidationConfig(
            input_subgraph_max_nodes=vl.get("input_subgraph_max_nodes", config.validation.input_subgraph_max_nodes),
            output_plan_max_length=vl.get("output_plan_max_length", config.validation.output_plan_max_length),
            output_plan_min_length=vl.get("output_plan_min_length", config.validation.output_plan_min_length),
        )

        return config

    def validate(self):
        assert self.sentence_bert.model_dim > 0, "sentence_bert.model_dim must be positive"
        self.extraction.validate()


# ============================================================
# DORMANT LEGACY CONFIG (DEVIATION 9)
# Superseded by RelationExtractionConfig. Kept on disk only so
# legacy training imports (model_training/, g2p/train.py) do not
# crash at import time. NOT referenced by G2PConfig or the runtime
# pipeline. Do not use in new code.
# ============================================================


@dataclass
class GraphToTextConfig:
    max_nodes_in_text: int = 100
    node_format: str = "{label} ({relation_to_previous})"
    separator: str = " "
    include_activations: bool = True
    activation_threshold: float = 0.1
    sort_by: str = "activation"
    sort_order: str = "descending"
    node_template_active: str = "{label}({activation:.2f})"
    node_template_simple: str = "{label}"
    include_edge_types: bool = True
    edge_format: str = " [via {relation}] "


@dataclass
class FFNConfig:
    input_dim: int = 384
    hidden_dim: int = 128
    num_layers: int = 2
    activation: str = "relu"
    dropout: float = 0.1
    classifier_input_dim: int = 128
    classifier_hidden_dim: int = 64
    classifier_num_layers: int = 1
    output_dim: int = 16


@dataclass
class DecoderConfig:
    type: str = "beam_search"
    beam_width: int = 2
    max_length: int = 8
    repetition_penalty: float = 1.2
    temperature: float = 1.0


@dataclass
class SyntheticDataConfig:
    num_samples: int = 2000
    min_nodes_per_graph: int = 3
    max_nodes_per_graph: int = 50
    graph_generation: str = "random_walk"
    seed: int = 42


@dataclass
class TrainingConfig:
    train_test_split: float = 0.8
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    optimizer: str = "adam"
    loss_function: str = "cross_entropy"
    early_stopping_patience: int = 5
    validation_split: float = 0.1
    synthetic_data: SyntheticDataConfig = field(default_factory=SyntheticDataConfig)


@dataclass
class RuleDefinition:
    condition: str
    plan: List[int]


@dataclass
class MappingConfig:
    heuristic_rules_enabled: bool = True
    rule_definitions: List[RuleDefinition] = field(default_factory=list)