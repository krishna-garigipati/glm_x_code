import numpy as np
import logging
from typing import List, Tuple, Optional, Dict
from .types import Subgraph
from .g2p_planner import G2PPlanner
from .config import G2PConfig

logger = logging.getLogger(__name__)


def generate_synthetic_data(
    config: G2PConfig,
    planner: Optional[G2PPlanner] = None,
) -> List:
    logger.info("Synthetic data disabled — planner uses pure embedding similarity")
    return []


def train_g2p(fname: str, training_data: Optional[List] = None,
              validation_data: Optional[List] = None,
              label_map: Optional[Dict] = None) -> Dict:
    """DORMANT (DEVIATION 9): IntentFFN training was removed. Import-compat shim."""
    logger.info("train_g2p disabled — QueryRelationExtractor requires no training (Deviation 9)")
    return {"epochs": 0, "train_loss": None, "val_loss": None}
