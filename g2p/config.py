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


@dataclass
class ValidationConfig:
    input_subgraph_max_nodes: int = 1000
    output_plan_max_length: int = 8
    output_plan_min_length: int = 1
    allowed_intents: List[int] = field(default_factory=lambda: list(range(16)))


@dataclass
class G2PConfig:
    sentence_bert: SentenceBERTConfig = field(default_factory=SentenceBERTConfig)
    graph_to_text: GraphToTextConfig = field(default_factory=GraphToTextConfig)
    ffn: FFNConfig = field(default_factory=FFNConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    intent_embeddings: Dict[int, str] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> "G2PConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

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

        gt = raw.get("graph_to_text", {})
        config.graph_to_text = GraphToTextConfig(
            max_nodes_in_text=gt.get("max_nodes_in_text", config.graph_to_text.max_nodes_in_text),
            node_format=gt.get("node_format", config.graph_to_text.node_format),
            separator=gt.get("separator", config.graph_to_text.separator),
            include_activations=gt.get("include_activations", config.graph_to_text.include_activations),
            activation_threshold=gt.get("activation_threshold", config.graph_to_text.activation_threshold),
            sort_by=gt.get("sort_by", config.graph_to_text.sort_by),
            sort_order=gt.get("sort_order", config.graph_to_text.sort_order),
            node_template_active=gt.get("node_template_active", config.graph_to_text.node_template_active),
            node_template_simple=gt.get("node_template_simple", config.graph_to_text.node_template_simple),
            include_edge_types=gt.get("include_edge_types", config.graph_to_text.include_edge_types),
            edge_format=gt.get("edge_format", config.graph_to_text.edge_format),
        )

        ff = raw.get("ffn", {})
        config.ffn = FFNConfig(
            input_dim=ff.get("input_dim", config.ffn.input_dim),
            hidden_dim=ff.get("hidden_dim", config.ffn.hidden_dim),
            num_layers=ff.get("num_layers", config.ffn.num_layers),
            activation=ff.get("activation", config.ffn.activation),
            dropout=ff.get("dropout", config.ffn.dropout),
            classifier_input_dim=ff.get("classifier_input_dim", config.ffn.classifier_input_dim),
            classifier_hidden_dim=ff.get("classifier_hidden_dim", config.ffn.classifier_hidden_dim),
            classifier_num_layers=ff.get("classifier_num_layers", config.ffn.classifier_num_layers),
            output_dim=ff.get("output_dim", config.ffn.output_dim),
        )

        dc = raw.get("decoder", {})
        config.decoder = DecoderConfig(
            type=dc.get("type", config.decoder.type),
            beam_width=dc.get("beam_width", config.decoder.beam_width),
            max_length=dc.get("max_length", config.decoder.max_length),
            repetition_penalty=dc.get("repetition_penalty", config.decoder.repetition_penalty),
            temperature=dc.get("temperature", config.decoder.temperature),
        )

        tr = raw.get("training", {})
        sd = tr.get("synthetic_data", {})
        config.training = TrainingConfig(
            train_test_split=tr.get("train_test_split", config.training.train_test_split),
            batch_size=tr.get("batch_size", config.training.batch_size),
            epochs=tr.get("epochs", config.training.epochs),
            learning_rate=tr.get("learning_rate", config.training.learning_rate),
            optimizer=tr.get("optimizer", config.training.optimizer),
            loss_function=tr.get("loss_function", config.training.loss_function),
            early_stopping_patience=tr.get("early_stopping_patience", config.training.early_stopping_patience),
            validation_split=tr.get("validation_split", config.training.validation_split),
            synthetic_data=SyntheticDataConfig(
                num_samples=sd.get("num_samples", 2000),
                min_nodes_per_graph=sd.get("min_nodes_per_graph", 3),
                max_nodes_per_graph=sd.get("max_nodes_per_graph", 50),
                graph_generation=sd.get("graph_generation", "random_walk"),
                seed=sd.get("seed", 42),
            ),
        )

        mp = raw.get("mapping", {})
        rules_raw = mp.get("rule_definitions", [])
        rules = []
        for r in rules_raw:
            rules.append(RuleDefinition(condition=r.get("condition", ""), plan=r.get("plan", [])))
        config.mapping = MappingConfig(
            heuristic_rules_enabled=mp.get("heuristic_rules_enabled", config.mapping.heuristic_rules_enabled),
            rule_definitions=rules,
        )

        vl = raw.get("validation", {})
        config.validation = ValidationConfig(
            input_subgraph_max_nodes=vl.get("input_subgraph_max_nodes", config.validation.input_subgraph_max_nodes),
            output_plan_max_length=vl.get("output_plan_max_length", config.validation.output_plan_max_length),
            output_plan_min_length=vl.get("output_plan_min_length", config.validation.output_plan_min_length),
            allowed_intents=vl.get("allowed_intents", config.validation.allowed_intents),
        )

        config.intent_embeddings = raw.get("intent_embeddings", {})

        return config

    def validate(self):
        assert self.ffn.input_dim == self.sentence_bert.model_dim, "FFN input_dim must match Sentence-BERT model_dim"
        assert self.ffn.output_dim == 16, "FFN output_dim must be 16 (INTENT_VOCAB_SIZE)"
        assert self.decoder.max_length <= self.validation.output_plan_max_length, "decoder max_length exceeds validation limit"
