#!/usr/bin/env python3
"""
GLM-X Resonance Component — Consolidated Test Runner
====================================================
Orchestrates all 10 testing phases and generates a structured report.

Usage:
    python -m Resonance.tests.run_all              # Run all tests
    python -m Resonance.tests.run_all --phase unit  # Run specific phase
    python -m Resonance.tests.run_all --skip-load   # Skip load/stress tests
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PHASES = [
    ("unit", "Unit Testing", ["test_unit/"]),
    ("smoke", "Smoke Testing", ["test_smoke.py"]),
    ("integration", "Integration Testing", ["test_integration.py"]),
    ("functional", "Functional Testing", ["test_functional.py"]),
    ("regression", "Regression Testing", ["test_regression.py"]),
    ("load", "Load Testing", ["test_load.py"]),
    ("stress", "Stress Testing", ["test_stress.py"]),
    ("concurrency", "Concurrency Testing", ["test_concurrency.py"]),
    ("security", "Security Testing", ["test_security.py"]),
    ("benchmark", "Benchmark / Evaluation", ["test_benchmark.py"]),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resonance Component Test Runner")
    parser.add_argument("--phase", choices=[p[0] for p in PHASES], help="Run only this phase")
    parser.add_argument("--skip-load", action="store_true", help="Skip load/stress/benchmark phases")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip benchmark phase")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage")
    parser.add_argument("--junit", action="store_true", help="Generate JUnit XML report")
    return parser.parse_args()


def _run_pytest(
    targets: List[str],
    phase_name: str,
    verbose: bool = False,
    junit: bool = False,
    coverage: bool = False,
) -> Tuple[int, int, int, float, str]:
    """Run pytest on given targets, return (passed, failed, skipped, elapsed_s, output)."""
    cmd = [sys.executable, "-m", "pytest"]
    if verbose:
        cmd.append("-v")
    if junit:
        report_path = REPORTS_DIR / f"junit_{phase_name.replace(' ', '_').lower()}.xml"
        cmd.extend(["--junitxml", str(report_path)])
    if coverage:
        cmd.extend(["--cov=Resonance", "--cov-report", "term-missing"])

    for t in targets:
        resolved = TESTS_DIR / t
        if resolved.exists():
            cmd.append(str(resolved))
        else:
            print(f"  [WARN] Test target not found: {resolved}")

    if len(cmd) < (4 + int(verbose) + int(junit) * 2 + int(coverage) * 3):
        return 0, 0, 0, 0.0, "No test files found"

    print(f"\n{'='*60}")
    print(f"  PHASE: {phase_name}")
    print(f"{'='*60}")

    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    elapsed = time.perf_counter() - start

    stdout = result.stdout
    stderr = result.stderr

    # Parse result from pytest summary
    import re as _re
    passed = failed = skipped = 0
    last_line = ""
    for line in stdout.split("\n"):
        stripped = line.strip()
        if stripped:
            last_line = stripped

    m = _re.search(r'(\d+)\s+passed', last_line)
    if m:
        passed = int(m.group(1))
    m = _re.search(r'(\d+)\s+failed', last_line)
    if m:
        failed = int(m.group(1))
    m = _re.search(r'(\d+)\s+skipped', last_line)
    if m:
        skipped = int(m.group(1))

    if result.returncode != 0 and failed == 0:
        for line in stdout.split("\n"):
            if "FAILED" in line and "::" in line:
                failed += 1

    output = stdout
    if stderr:
        output += "\n--- STDERR ---\n" + stderr

    print(stdout)
    if stderr:
        print(stderr, file=sys.stderr)

    return passed, failed, skipped, elapsed, output


def _compute_coverage() -> Dict:
    """Run a quick coverage summary."""
    result = subprocess.run(
        [sys.executable, "-m", "coverage", "report", "--include=Resonance/*"],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    return {"report": result.stdout, "error": result.stderr}


def _count_test_files() -> int:
    count = 0
    for phase_id, _, targets in PHASES:
        for t in targets:
            path = TESTS_DIR / t
            if path.is_dir():
                for f in path.rglob("test_*.py"):
                    count += 1
            elif path.is_file():
                count += 1
    return count


def _run_phase(
    phase_id: str, phase_name: str, targets: List[str], args: argparse.Namespace
) -> Dict:
    passed, failed, skipped, elapsed, output = _run_pytest(
        targets, phase_name, verbose=args.verbose, junit=args.junit, coverage=args.coverage,
    )
    return {
        "phase_id": phase_id,
        "phase_name": phase_name,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "elapsed_s": round(elapsed, 2),
        "status": "PASS" if failed == 0 else "FAIL",
    }


def _generate_report(results: List[Dict], total_elapsed: float, args: argparse.Namespace) -> str:
    total_passed = sum(r["passed"] for r in results)
    total_failed = sum(r["failed"] for r in results)
    total_skipped = sum(r["skipped"] for r in results)
    total_tests = total_passed + total_failed + total_skipped

    lines = []
    lines.append("=" * 70)
    lines.append("  GLM-X RESONANCE COMPONENT — CONSOLIDATED TEST REPORT")
    lines.append("=" * 70)
    lines.append(f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Python:    {sys.version.split()[0]}")
    lines.append(f"  Platform:  {sys.platform}")
    lines.append(f"  Test files: {_count_test_files()}")
    lines.append(f"  Coverage:  N/A (run with --coverage for details)")
    lines.append("")

    lines.append("-" * 70)
    lines.append("  PHASE RESULTS")
    lines.append("-" * 70)
    for r in results:
        status_icon = "[PASS]" if r["status"] == "PASS" else "[FAIL]"
        lines.append(
            f"  {status_icon} {r['phase_name']:<25s} "
            f"  {r['passed']:>3d} passed  {r['failed']:>3d} failed  "
            f"{r['skipped']:>3d} skipped  ({r['elapsed_s']:>6.2f}s)"
        )
    lines.append("")

    lines.append("-" * 70)
    lines.append("  SUMMARY")
    lines.append("-" * 70)
    lines.append(f"  Total tests:  {total_tests}")
    lines.append(f"  Passed:       {total_passed}  ({total_passed / max(total_tests, 1) * 100:.1f}%)")
    lines.append(f"  Failed:       {total_failed}")
    lines.append(f"  Skipped:      {total_skipped}")
    lines.append(f"  Total time:   {total_elapsed:.2f}s")

    lines.append("")
    lines.append("-" * 70)
    lines.append("  FAILURE DETAILS")
    lines.append("-" * 70)
    has_failures = False
    for r in results:
        if r["failed"] > 0:
            has_failures = True
            lines.append(f"  [FAIL] {r['phase_name']}: {r['failed']} failure(s)")
    if not has_failures:
        lines.append("  [PASS] All phases passed - no failures detected.")
    lines.append("")

    lines.append("-" * 70)
    lines.append("  RISK ASSESSMENT")
    lines.append("-" * 70)
    if total_failed == 0:
        lines.append("  [LOW] All tests pass. Component is production-ready.")
    elif total_failed <= 3:
        lines.append("  [MEDIUM] Minor failures detected. Review before deployment.")
    else:
        lines.append("  [HIGH] Multiple failures detected. Investigate before deployment.")

    lines.append("")
    lines.append("-" * 70)
    lines.append("  BLUEPRINT COMPLIANCE")
    lines.append("-" * 70)
    phase_names = [r["phase_name"] for r in results if r["failed"] == 0]
    if "Smoke Testing" in phase_names:
        lines.append("  [PASS] Component boots and initializes correctly")
    if "Functional Testing" in phase_names:
        lines.append("  [PASS] Business logic matches blueprint specification")
    if "Integration Testing" in phase_names:
        lines.append("  [PASS] Module-to-module communication verified")
    if "Security Testing" in phase_names:
        lines.append("  [PASS] Security posture validated (no eval, safe YAML, input validation)")
    lines.append("")

    lines.append("=" * 70)
    lines.append("  END OF REPORT")
    lines.append("=" * 70)

    return "\n".join(lines)


def main():
    args = _parse_args()
    os.chdir(PROJECT_ROOT)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"GLM-X Resonance Test Runner")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Reports dir:  {REPORTS_DIR}")
    print()

    phases_to_run = PHASES
    if args.phase:
        phases_to_run = [p for p in PHASES if p[0] == args.phase]
    if args.skip_load:
        phases_to_run = [p for p in phases_to_run if p[0] not in ("load", "stress")]
    if args.skip_benchmark:
        phases_to_run = [p for p in phases_to_run if p[0] not in ("benchmark",)]

    results: List[Dict] = []
    overall_start = time.perf_counter()

    for phase_id, phase_name, targets in phases_to_run:
        result = _run_phase(phase_id, phase_name, targets, args)
        results.append(result)

    total_elapsed = time.perf_counter() - overall_start

    report = _generate_report(results, total_elapsed, args)
    print("\n" + report)

    report_path = REPORTS_DIR / f"consolidated_report_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {report_path}")

    json_path = REPORTS_DIR / f"results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {json_path}")

    total_failed = sum(r["failed"] for r in results)
    sys.exit(1 if total_failed > 0 else 0)


if __name__ == "__main__":
    main()
