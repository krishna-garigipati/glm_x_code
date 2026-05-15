import sys, os, tempfile, shutil, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pytest
from knowledge_graph.graph_component_implementation.graph_store import GraphStore

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_SRC = os.path.join(PROJECT_ROOT, "graph_component_implementation", "config_graph.yaml")


def make_graph():
    tmp = tempfile.mkdtemp()
    with open(CONFIG_SRC, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["storage"]["base_path"] = tmp
    cfg["backup"]["enabled"] = False
    cfg_path = os.path.join(tmp, "config.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)
    gs = GraphStore(config_path=cfg_path)
    return gs, tmp


@pytest.fixture(scope="function")
def graph():
    gs, tmp = make_graph()
    yield gs
    try:
        gs.close()
    except Exception:
        pass
    shutil.rmtree(tmp, ignore_errors=True)
