"""
run_mixture_experiment.py
=========================
Train and evaluate every Sim+Real mixture x training seed, one run at a time.

Order is seed-major (all seven mixtures for seed 0, then seed 1, then seed 2),
so a complete mixture curve exists after the first seven runs.

    python -u scripts/run_mixture_experiment.py

Resumable: a run whose ``results/mixed/<name>/metrics.json`` exists is skipped;
a run that trained but was not evaluated is only evaluated. A run directory
without ``best.pt`` (an interrupted training) stops the driver instead of being
deleted, so nothing is ever overwritten.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from yolo_dataset_splitter import PROJECT_ROOT

SHARES = (0, 20, 40, 50, 60, 80, 100)
SEEDS = (0, 1, 2)
RUNS_ROOT = PROJECT_ROOT / "runs" / "obb" / "UR16e_Labware_Detection"
RESULTS_ROOT = PROJECT_ROOT / "results" / "mixed"
SCRIPTS = PROJECT_ROOT / "scripts"


def run(cmd):
    print(f"\n$ {' '.join(map(str, cmd))}", flush=True)
    subprocess.run([str(c) for c in cmd], cwd=PROJECT_ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run the Sim+Real mixture experiment.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--shares", type=int, nargs="+", default=list(SHARES))
    args = parser.parse_args()
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for seed in args.seeds:
        for share in args.shares:
            name = f"mixed_real{share:02d}_seed{seed}"
            result_dir = RESULTS_ROOT / name
            run_dir = RUNS_ROOT / name
            best = run_dir / "weights" / "best.pt"
            if (result_dir / "metrics.json").is_file():
                print(f"[skip] {name}: already evaluated", flush=True)
                continue
            if result_dir.exists():
                sys.exit(f"{result_dir} exists without metrics.json; inspect it before rerunning")
            t0 = time.time()
            if not run_dir.exists():
                run([sys.executable, "-u", SCRIPTS / "train_mixed.py",
                     "--mixture", PROJECT_ROOT / "mixed_datasets" / f"real{share:02d}",
                     "--train-seed", seed, "--name", name])
            elif not best.is_file():
                sys.exit(f"{run_dir} exists without best.pt (interrupted run); inspect it before rerunning")
            run([sys.executable, SCRIPTS / "eval_real_test.py",
                 "--weights", best, "--dataset", PROJECT_ROOT / "dataset" / "real_v2",
                 "--output", result_dir, "--skip-predict"])
            print(f"[done] {name} in {(time.time() - t0) / 60:.1f} min "
                  f"(total {(time.time() - started) / 3600:.2f} h)", flush=True)
    print("\nAll requested mixture runs are evaluated.", flush=True)


if __name__ == "__main__":
    main()
