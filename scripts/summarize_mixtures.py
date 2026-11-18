"""
summarize_mixtures.py
=====================
Collect ``results/mixed/mixed_real<NN>_seed<N>/metrics.json`` into
``results/mixed/summary.json`` and plot real share against real_v2 test
mAP50-95 (mean +/- std over training seeds) to ``results/mixed/mixture_curve.png``.

    python scripts/summarize_mixtures.py

Incomplete mixtures (fewer seeds than the others) are reported but flagged.
"""

import json
import re
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from yolo_dataset_splitter import CLASS_NAMES, PROJECT_ROOT

RESULTS = PROJECT_ROOT / "results" / "mixed"
NAME = re.compile(r"mixed_real(\d+)_seed(\d+)$")
KEYS = ("precision", "recall", "mAP50", "mAP50-95")


def mean_std(values):
    values = [v * 100 for v in values]
    return {"mean": round(st.mean(values), 2),
            "std": round(st.stdev(values), 2) if len(values) > 1 else None,
            "n": len(values)}


def main():
    runs = {}
    for metrics in sorted(RESULTS.glob("mixed_real*_seed*/metrics.json")):
        match = NAME.match(metrics.parent.name)
        if not match:
            continue
        share, seed = int(match[1]), int(match[2])
        data = json.loads(metrics.read_text(encoding="utf-8"))
        runs.setdefault(share, {})[seed] = data
    if not runs:
        raise SystemExit("no evaluated mixture runs under results/mixed/")
    seed_counts = {share: len(seeds) for share, seeds in runs.items()}
    full = max(seed_counts.values())

    mixtures = []
    for share in sorted(runs):
        seeds = runs[share]
        per_seed = {seed: {k: round(d["overall"][k] * 100, 2) for k in ("mAP50", "mAP50-95")}
                    for seed, d in sorted(seeds.items())}
        per_class = {}
        for name in CLASS_NAMES.values():
            rows = [next(c for c in d["per_class"] if c["name"] == name) for d in seeds.values()]
            per_class[name] = {k: mean_std([r[k] for r in rows]) for k in KEYS}
        mixtures.append({
            "real_share_percent": share,
            "real_images": 800 * share // 100,
            "synthetic_images": 800 - 800 * share // 100,
            "seeds": sorted(seeds),
            "complete": len(seeds) == full,
            "per_seed": per_seed,
            "overall": {k: mean_std([d["overall"][k] for d in seeds.values()]) for k in KEYS},
            "per_class": per_class,
        })

    summary = {
        "experiment": "Sim+Real mixtures, 800 training images each, YOLOv8n-OBB, 30 epochs",
        "synthetic_source": "dataset_controlnet_fg05 train (ControlNet s0.45, fg-keep 0.5)",
        "real_source": "dataset/real_v2 train (captures 0001, 0003, 0004, 0005)",
        "checkpoint_selection": "dataset/real_v2 val (capture 0006)",
        "test": "dataset/real_v2 test (captures 0007, 0009; 500 frames, 2,990 instances)",
        "metrics_note": "percent; mean and sample std over training seeds",
        "mixtures": mixtures,
    }
    controls = {}
    for metrics in sorted(RESULTS.glob("control_real160only_seed*/metrics.json")):
        seed = int(metrics.parent.name.rsplit("seed", 1)[1])
        controls[seed] = json.loads(metrics.read_text(encoding="utf-8"))
    if controls:
        real20 = runs.get(20, {})
        paired = {seed: round((real20[seed]["overall"]["mAP50-95"] - d["overall"]["mAP50-95"]) * 100, 2)
                  for seed, d in sorted(controls.items()) if seed in real20}
        summary_control = {
            "name": "real20_only",
            "description": "the same 160 real images as real20, each listed 5 times (800 entries), no synthetic",
            "seeds": sorted(controls),
            "per_seed": {seed: {k: round(d["overall"][k] * 100, 2) for k in ("mAP50", "mAP50-95")}
                         for seed, d in sorted(controls.items())},
            "overall": {k: mean_std([d["overall"][k] for d in controls.values()]) for k in KEYS},
            "per_class": {name: {k: mean_std([next(c for c in d["per_class"] if c["name"] == name)[k]
                                              for d in controls.values()]) for k in KEYS}
                          for name in CLASS_NAMES.values()},
            "real20_minus_control_mAP50-95_per_seed": paired,
            "real20_minus_control_mAP50-95_mean": round(st.mean(paired.values()), 2) if paired else None,
        }
        summary["real_only_control"] = summary_control
    out = RESULTS / "summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    xs = [m["real_share_percent"] for m in mixtures]
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    for key, label, color in (("mAP50-95", "mAP50–95", "#2a6fdb"), ("mAP50", "mAP50", "#9aa5b1")):
        ys = [m["overall"][key]["mean"] for m in mixtures]
        es = [m["overall"][key]["std"] or 0 for m in mixtures]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=4, linewidth=2, color=color, label=label)
    ax.set_xlabel("Real images in the 800-image training set (%)")
    ax.set_ylabel("Real test score (%)")
    ax.set_title("Sim+Real mixtures on held-out real captures (mean ± std over seeds)")
    ax.set_xticks(xs)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "mixture_curve.png")

    print(f"{'real %':>7} {'seeds':>6} {'mAP50':>14} {'mAP50-95':>14}")
    for m in mixtures:
        o = m["overall"]
        fmt = lambda v: f"{v['mean']:.2f} ± {v['std']:.2f}" if v["std"] is not None else f"{v['mean']:.2f}"
        print(f"{m['real_share_percent']:>7} {len(m['seeds']):>6} {fmt(o['mAP50']):>14} {fmt(o['mAP50-95']):>14}"
              + ("" if m["complete"] else "  (incomplete)"))
    if controls:
        o = summary_control["overall"]
        print(f"control 160 real only ({len(controls)} seeds): mAP50 {o['mAP50']['mean']:.2f} ± {o['mAP50']['std'] or 0:.2f}, "
              f"mAP50-95 {o['mAP50-95']['mean']:.2f} ± {o['mAP50-95']['std'] or 0:.2f}; "
              f"real20 minus control per seed {summary_control['real20_minus_control_mAP50-95_per_seed']}")
    print(f"wrote {out} and {RESULTS / 'mixture_curve.png'}")


if __name__ == "__main__":
    main()
