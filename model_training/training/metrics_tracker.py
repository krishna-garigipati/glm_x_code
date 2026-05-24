import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .metrics_schemas import COMPONENT_SCHEMAS

logger = logging.getLogger(__name__)


class MetricsTracker:
    def __init__(self, run_id: str, checkpoint_dir: str):
        self.run_id = run_id
        self.checkpoint_dir = Path(checkpoint_dir)
        self.dataset_name: str = ""
        self.source_file: str = ""
        self.is_continual: bool = False
        self.previous_checkpoint_path: str = ""

        self._components: Dict[str, dict] = {}
        for name, schema in COMPONENT_SCHEMAS.items():
            self._components[name] = dict(schema)

    def set_component(self, name: str, data: dict) -> None:
        if name in self._components:
            base = dict(self._components[name])
            base.update(data)
            self._components[name] = base
        else:
            self._components[name] = dict(data)

    def get_component(self, name: str) -> dict:
        return dict(self._components.get(name, {}))

    def add_epoch_metric(self, component: str, epoch_data: dict) -> None:
        key = f"{component}_epochs"
        if key not in self._components:
            self._components[key] = []
        self._components[key].append(epoch_data)

    def generate_summary(self) -> Dict[str, Any]:
        summary = {
            "run_id": self.run_id,
            "dataset": {
                "name": self.dataset_name,
                "source": self.source_file,
            },
        }
        mapping = {
            "graph_store": "graph",
            "intent_ffn": "intent_ffn",
            "t5_decoder": "t5",
            "walker": "walker",
            "resonance": "resonance",
            "decoder": "decoder",
            "qa_eval": "qa",
            "continual": "continual",
            "data_loader": "data_loader",
            "kg_builder": "kg_builder",
        }
        for src, dst in mapping.items():
            if src in self._components:
                summary[dst] = dict(self._components[src])
        summary["system"] = {
            "duration_seconds": 0,
            "model_version": "2.0",
        }
        return summary

    def save_all(self) -> str:
        run_dir = self.checkpoint_dir / "metrics" / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "run_id": self.run_id,
            "dataset_name": self.dataset_name,
            "source_file": self.source_file,
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "duration_seconds": 0,
            "is_continual_training": self.is_continual,
            "model_version": "2.0",
        }
        with open(run_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        name_map = {
            "data_loader": "data_loader_metrics.json",
            "kg_builder": "kg_builder_metrics.json",
            "graph_store": "graph_store_metrics.json",
            "intent_ffn": "intent_ffn_metrics.json",
            "t5_decoder": "t5_decoder_metrics.json",
            "walker": "walker_metrics.json",
            "resonance": "resonance_metrics.json",
            "decoder": "decoder_metrics.json",
            "qa_eval": "qa_evaluation_metrics.json",
            "continual": "continual_learning_metrics.json",
        }
        for comp, fname in name_map.items():
            if comp in self._components:
                with open(run_dir / fname, "w") as f:
                    json.dump(self._components[comp], f, indent=2)

        summary = self.generate_summary()
        with open(run_dir / "unified_metrics_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Metrics saved to {run_dir}")
        return str(run_dir)

    def append_to_history(self) -> None:
        history_path = self.checkpoint_dir / "metrics_history.jsonl"
        entry = {
            "run_id": self.run_id,
            "dataset": self.dataset_name,
            "source": self.source_file,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if "graph_store" in self._components:
            gs = self._components["graph_store"]
            entry["nodes"] = gs.get("node_count", 0)
            entry["edges"] = gs.get("edge_count", 0)
        if "intent_ffn" in self._components:
            ffn = self._components["intent_ffn"]
            entry["intent_ffn_test_top1"] = ffn.get("test_top1_accuracy", 0)
            entry["intent_ffn_test_loss"] = ffn.get("test_loss", 0)
            entry["epochs"] = ffn.get("epochs_trained", 0)
        if "qa_eval" in self._components:
            qa = self._components["qa_eval"]
            entry["qa_accuracy"] = qa.get("qa_accuracy", 0)
            entry["qa_correct"] = qa.get("qa_correct", 0)
            entry["qa_total"] = qa.get("qa_total", 0)
        if "continual" in self._components:
            c = self._components["continual"]
            entry["catastrophic_forgetting"] = c.get("forgetting_evaluation", {}).get("catastrophic_forgetting_score")
            entry["transfer_score"] = c.get("transfer_evaluation", {}).get("transfer_score")

        with open(history_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
