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
