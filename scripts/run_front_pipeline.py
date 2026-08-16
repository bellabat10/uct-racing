"""
Front corner pipeline runner - runs Stage A, B and C in sequence exactly as
they would be run from the terminal.

Usage:
    python scripts/run_front_pipeline.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

STAGES = [
    (
        "Stage A: mount_kpi_sweep",
        [sys.executable, str(SCRIPTS_DIR / "mount_kpi_sweep.py"), "--plots"],
    ),
    (
        "Stage B: caster_trail_analysis",
        [
            sys.executable,
            str(SCRIPTS_DIR / "caster_trail_analysis.py"),
            "--geometry",
            "outputs/front/mount_height_kpi_sweep/best_zkpi_geometry.yaml",
            "--out-dir",
            "outputs/front/caster_and_trail",
        ],
    ),
    (
        "Stage C: build_optimal_geometry",
        [sys.executable, str(SCRIPTS_DIR / "build_optimal_geometry.py")],
    ),
]


def main() -> None:
    pipeline_start = time.monotonic()

    for stage_name, cmd in STAGES:
        print(f"[{datetime.now().isoformat(timespec='seconds')}] Starting {stage_name}")
        stage_start = time.monotonic()

        result = subprocess.run(cmd, cwd=REPO_ROOT)

        stage_elapsed = time.monotonic() - stage_start
        print(f"[{datetime.now().isoformat(timespec='seconds')}] Finished {stage_name} in {stage_elapsed:.1f}s")

        if result.returncode != 0:
            print(f"ERROR: {stage_name} failed with exit code {result.returncode}. Aborting pipeline.")
            sys.exit(result.returncode)

    total_elapsed = time.monotonic() - pipeline_start
    print(f"Total elapsed time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
