import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List

from .intent_ffn import IntentFFN

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._model: Optional[IntentFFN] = None

    @property
    def model(self) -> Optional[IntentFFN]:
        return self._model

    def create_model(self, **ffn_kwargs) -> IntentFFN:
        self._model = IntentFFN(**ffn_kwargs)
        logger.info(
            f"Created new IntentFFN: {self._model.get_num_params()} params, "
            f"config={self._model.config}"
        )
        return self._model

    def load_model(self, dataset_name: str, filename: str = "intent_ffn_best.pt") -> IntentFFN:
        path = self.base_dir / dataset_name / filename
        self._model = IntentFFN.load(path)
        return self._model

    def save_model(
        self,
        dataset_name: str,
        filename: str = "intent_ffn_best.pt",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if self._model is None:
            raise ValueError("No model to save. Create or load one first.")
        save_dir = self.base_dir / dataset_name
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / filename
        self._model.save(save_path, metadata=metadata)

    def expand_model(self, new_output_dim: int = 16) -> IntentFFN:
        if self._model is None:
            raise ValueError("No model to expand. Load or create one first.")
        self._model = self._model.expand_for_new_dataset(
            new_output_classes=new_output_dim, preserve_weights=True
        )
        return self._model

    def list_saved_models(self) -> List[dict]:
        results = []
        for dataset_dir in self.base_dir.iterdir():
            if dataset_dir.is_dir():
                pt_files = list(dataset_dir.glob("*.pt"))
                json_files = list(dataset_dir.glob("*.json"))
                for ptf in pt_files:
                    info = {
                        "dataset": dataset_dir.name,
                        "model_path": str(ptf),
                        "size_bytes": ptf.stat().st_size,
                    }
                    for jf in json_files:
                        if "metadata" in jf.name:
                            info["metadata"] = str(jf)
                    results.append(info)
        return results
