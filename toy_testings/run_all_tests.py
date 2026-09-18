#!/usr/bin/env python
"""Master test runner for GLM-X toy_testings.

Runs all toy tests:
1. Graph store API tests (from graph/run_toy_dataset.py)
2. Component unit tests
3. Full pipeline integration tests
4. Configuration validation
"""
from __future__ import annotations

import sys
import os
import subprocess
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_script(script_path: str, args: list = None) -> dict:
    """Run a Python script and return result."""
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)

    # Run from project root (updated_glmx) for proper imports
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Set PYTHONPATH to include project root
    env = os.environ.copy()
    env['PYTHONPATH'] = project_root + os.pathsep + env.get('PYTHONPATH', '')

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=project_root, env=env)
        return {
            "script": os.path.basename(script_path),
            "returncode": result.returncode,
            "stdout": result.stdout[-2000:] if result.stdout else "",
            "stderr": result.stderr[-2000:] if result.stderr else "",
            "success": result.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        return {
            "script": os.path.basename(script_path),
            "returncode": -1,
            "stdout": "",
            "stderr": "TIMEOUT (120s)",
            "success": False,
        }
    except Exception as e:
        return {
            "script": os.path.basename(script_path),
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
            "success": False,
        }


def run_graph_store_tests():
    """Run the existing graph store toy dataset tests."""
    print("\n" + "=" * 70)
    print("RUNNING GRAPH STORE TOY DATASET TESTS")
    print("=" * 70)

    script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "graph", "run_toy_dataset.py"
    )
    return run_script(script)


def run_graph_unit_tests():
    """Run graph unit tests."""
    print("\n" + "=" * 70)
    print("RUNNING GRAPH UNIT TESTS")
    print("=" * 70)

    script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "graph", "tests", "test_unit.py"
    )
    return run_script(script)


def run_pipeline_tests():
    """Run full pipeline tests."""
    print("\n" + "=" * 70)
    print("RUNNING FULL PIPELINE TESTS")
    print("=" * 70)

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_pipeline.py")
    return run_script(script, ["--pipeline"])


def run_component_unit_tests():
    """Run component unit tests."""
    print("\n" + "=" * 70)
    print("RUNNING COMPONENT UNIT TESTS")
    print("=" * 70)

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_pipeline.py")
    return run_script(script, ["--unit"])


def run_config_validation():
    """Validate all config files."""
    print("\n" + "=" * 70)
    print("RUNNING CONFIG VALIDATION")
    print("=" * 70)

    script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts", "validate_configs.py"
    )
    return run_script(script, ["--config-dir", "configs"])


def check_demo_graph():
    """Test the demo graph from graph/demo_graph_data.py."""
    print("\n" + "=" * 70)
    print("TESTING DEMO GRAPH (graph/demo_graph_data.py)")
    print("=" * 70)

    try:
        from graph.demo_graph_data import build_demo_store
        store = build_demo_store()
        print(f"  Demo graph: {store.get_node_count()} nodes, {store.get_edge_count()} edges")
        print(f"  Relations: {store.get_all_relations()}")

        # Test a few queries
        sub = store.get_subgraph_by_embedding_similarity(
            __import__('numpy').zeros(384, dtype=__import__('numpy').float32), top_k=5
        )
        print(f"  Embedding search: {len(sub.nodes)} nodes")
        # sub.nodes is a List[int], not dict
        for nid in sub.nodes:
            node = store.get_node(nid)
            if node:
                print(f"    {node.label} (act={node.activation:.3f})")

        return {"success": True, "nodes": store.get_node_count(), "edges": store.get_edge_count()}
    except Exception as e:
        return {"success": False, "error": str(e)}


def main():
    """Run all toy tests and produce summary report."""
    print("=" * 70)
    print("GLM-X TOY_TESTINGS - MASTER TEST RUNNER")
    print("=" * 70)

    all_results = {
        "timestamp": __import__('time').time(),
        "tests": {},
        "summary": {},
    }

    # Test 1: Demo graph
    demo_result = check_demo_graph()
    all_results["tests"]["demo_graph"] = demo_result
    status = "PASS" if demo_result.get("success") else "FAIL"
    print(f"  {status} Demo graph: {demo_result.get('nodes', 0)} nodes, {demo_result.get('edges', 0)} edges")

    # Test 2: Graph store toy dataset
    graph_result = run_graph_store_tests()
    all_results["tests"]["graph_store_toy"] = graph_result
    status = "PASS" if graph_result["success"] else "FAIL"
    print(f"  {status} Graph store toy dataset: {graph_result['returncode']}")

    # Test 3: Graph unit tests
    unit_result = run_graph_unit_tests()
    all_results["tests"]["graph_unit"] = unit_result
    status = "PASS" if unit_result["success"] else "FAIL"
    print(f"  {status} Graph unit tests: {unit_result['returncode']}")

    # Test 4: Config validation
    config_result = run_config_validation()
    all_results["tests"]["config_validation"] = config_result
    status = "PASS" if config_result["success"] else "FAIL"
    print(f"  {status} Config validation: {config_result['returncode']}")

    # Test 5: Component unit tests
    comp_result = run_component_unit_tests()
    all_results["tests"]["component_unit"] = comp_result
    status = "PASS" if comp_result["success"] else "FAIL"
    print(f"  {status} Component unit tests: {comp_result['returncode']}")

    # Test 6: Full pipeline tests
    pipe_result = run_pipeline_tests()
    all_results["tests"]["full_pipeline"] = pipe_result
    status = "PASS" if pipe_result["success"] else "FAIL"
    print(f"  {status} Full pipeline tests: {pipe_result['returncode']}")

    # Summary
    total = len(all_results["tests"])
    passed = sum(1 for t in all_results["tests"].values() if t.get("success", False))
    all_results["summary"] = {
        "total_test_suites": total,
        "passed": passed,
        "failed": total - passed,
        "overall_success": passed == total,
    }

    print("\n" + "=" * 70)
    print("MASTER TEST SUMMARY")
    print("=" * 70)
    for name, result in all_results["tests"].items():
        status = "PASS" if result.get("success", False) else "FAIL"
        print(f"  {name:25s}: {status}")
    print("-" * 70)
    print(f"  OVERALL: {passed}/{total} test suites passed")
    print("=" * 70)

    # Save report
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "master_test_report.json")
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFull report saved to: {report_path}")

    return 0 if all_results["summary"]["overall_success"] else 1


if __name__ == "__main__":
    sys.exit(main())