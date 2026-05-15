from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Generator, List, Tuple

import numpy as np
import pytest

from .fixtures.config_provider import (
    build_minimal_core_config,
    build_minimal_loaded_configs,
    build_minimal_resonance_config,
    get_test_config_dir,
    load_test_configs,
)
from .fixtures.toy_data import (
    animal_dataset_seeds,
    animal_query_embedding,
    build_animal_kingdom_graph,
)
from .fixtures.toy_graph_builder import (
    build_chain_graph,
    build_cluster_graph,
    build_contradiction_graph,
    build_dense_graph,
    build_disconnected_graph,
    build_empty_graph,
    build_multi_edge_graph,
    build_self_loop_graph,
    build_single_node_graph,
    build_star_graph,
    build_two_node_graph,
    make_query_embedding,
)
from .fixtures.toy_graph_store import ToyGraphStore

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s [%(name)s] %(message)s")


@pytest.fixture(scope="session")
def test_config_dir() -> Path:
    return get_test_config_dir()


@pytest.fixture(scope="session")
def loaded_configs():
    return load_test_configs()


@pytest.fixture(scope="function")
def core_config():
    return build_minimal_core_config()


@pytest.fixture(scope="function")
def resonance_config():
    return build_minimal_resonance_config()


@pytest.fixture(scope="function")
def minimal_configs():
    return build_minimal_loaded_configs()


@pytest.fixture(scope="function")
def query_embedding() -> np.ndarray:
    return make_query_embedding()


@pytest.fixture(scope="function")
def empty_graph() -> ToyGraphStore:
    return build_empty_graph()


@pytest.fixture(scope="function")
def single_node_graph() -> ToyGraphStore:
    return build_single_node_graph()


@pytest.fixture(scope="function")
def two_node_graph() -> ToyGraphStore:
    return build_two_node_graph()


@pytest.fixture(scope="function")
def chain_graph() -> ToyGraphStore:
    return build_chain_graph(length=8, relation="follows")


@pytest.fixture(scope="function")
def star_graph() -> ToyGraphStore:
    return build_star_graph(center_id=0, leaf_count=8)


@pytest.fixture(scope="function")
def cluster_graph() -> ToyGraphStore:
    return build_cluster_graph(num_clusters=3, nodes_per_cluster=5, bridge_edges=2)


@pytest.fixture(scope="function")
def disconnected_graph() -> ToyGraphStore:
    return build_disconnected_graph()


@pytest.fixture(scope="function")
def dense_graph() -> ToyGraphStore:
    return build_dense_graph(node_count=20, edge_density=0.3)


@pytest.fixture(scope="function")
def self_loop_graph() -> ToyGraphStore:
    return build_self_loop_graph()


@pytest.fixture(scope="function")
def multi_edge_graph() -> ToyGraphStore:
    return build_multi_edge_graph()


@pytest.fixture(scope="function")
def contradiction_graph() -> ToyGraphStore:
    return build_contradiction_graph()


@pytest.fixture(scope="session")
def animal_graph() -> ToyGraphStore:
    return build_animal_kingdom_graph()


@pytest.fixture(scope="session")
def animal_seeds() -> List[int]:
    return animal_dataset_seeds()


@pytest.fixture(scope="session")
def animal_query() -> np.ndarray:
    return animal_query_embedding()


def nonzero_embedding() -> np.ndarray:
    return np.ones(384, dtype=np.float32)


def zero_embedding() -> np.ndarray:
    return np.zeros(384, dtype=np.float32)


def nan_embedding() -> np.ndarray:
    return np.full(384, np.nan, dtype=np.float32)


def inf_embedding() -> np.ndarray:
    return np.full(384, np.inf, dtype=np.float32)
