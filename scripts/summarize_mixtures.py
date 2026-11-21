"""
summarize_mixtures.py
=====================
Collect the Sim+Real mixture runs and their real-only controls into
``results/mixed/summary.json`` and plot real_v2 test mAP50-95 against the number
of real training images to ``results/mixed/mixture_curve.png``.

Run families (all evaluated on the real_v2 test captures 0007/0009):

* ``mixed_real<NN>_seed<S>``      N real + (800 - N) synthetic images
* ``realonly_<NNN>_x800_seed<S>`` the same N real images repeated to 800 lines
                                  (step-matched; ``control_real160only_seed<S>`` is N = 160)
* ``realonly_<NNN>_seed<S>``      the same N real images listed once (raw; fewer steps)

    python scripts/summarize_mixtures.py
"""

import json
import math
import re
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from yolo_dataset_splitter import CLASS_NAMES, PROJECT_ROOT

RESULTS = PROJECT_ROOT / "results" / "mixed"
KEYS = ("precision", "recall", "mAP50", "mAP50-95")
PATTERNS = (
    ("mixture", re.compile(r"mixed_real(\d+)_seed(\d+)$"), lambda g: 800 * int(g) // 100),
    ("real_only_steps_matched", re.compile(r"realonly_(\d+)_x800_seed(\d+)$"), int),
    ("real_only_steps_matched", re.compile(r"control_real160only_seed(\d+)$"), None),
    ("real_only_raw", re.compile(r"realonly_(\d+)_seed(\d+)$"), int),
)
LABELS = {
    "mixture": "Sim+Real mixture (800 images: N real + synthetic)",
    "real_only_steps_matched": "Real only, step-matched (N real repeated to 800 lines)",
    "real_only_raw": "Real only, raw (N real listed once; fewer training steps)",
}


def mean_std(values):
    values = [v * 100 for v in values]
    return {"mean": round(st.mean(values), 2),
            "std": round(st.stdev(values), 2) if len(values) > 1 else None,
            "n": len(values)}


def t_two_sided_p(t, df, steps=200000):
    """Two-sided p-value of Student's t by numeric integration (no scipy here)."""
    if not math.isfinite(t):
        return 0.0
    c = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))
    a, h = abs(t), 200.0 / steps
    return min(1.0, 2 * sum(c * (1 + (a + (i + 0.5) * h) ** 2 / df) ** (-(df + 1) / 2) for i in range(steps)) * h)


def collect():
    runs = {family: {} for family in LABELS}
    for metrics in sorted(RESULTS.glob("*/metrics.json")):
        name = metrics.parent.name
        for family, pattern, to_n in PATTERNS:
            match = pattern.match(name)
            if not match:
                continue
            if to_n is None:
                n, seed = 160, int(match[1])
            else:
                n, seed = to_n(match[1]), int(match[2])
            runs[family].setdefault(n, {})[seed] = json.loads(metrics.read_text(encoding="utf-8"))
            break
    # 100 % real is real-only with 800 distinct images: it closes both real-only curves.
    if 800 in runs["mixture"]:
        for family in ("real_only_steps_matched", "real_only_raw"):
            runs[family].setdefault(800, runs["mixture"][800])
    return runs


def describe(seeds):
    return {
        "seeds": sorted(seeds),
        "per_seed": {seed: {k: round(d["overall"][k] * 100, 2) for k in ("mAP50", "mAP50-95")}
                     for seed, d in sorted(seeds.items())},
        "overall": {k: mean_std([d["overall"][k] for d in seeds.values()]) for k in KEYS},
        "per_class": {name: {k: mean_std([next(c for c in d["per_class"] if c["name"] == name)[k]
                                          for d in seeds.values()]) for k in KEYS}
                      for name in CLASS_NAMES.values()},
    }


def paired(a, b):
    """Mixture minus control, paired by training seed, in mAP50-95 points."""
    seeds = sorted(set(a) & set(b))
    diffs = [(a[s]["overall"]["mAP50-95"] - b[s]["overall"]["mAP50-95"]) * 100 for s in seeds]
    out = {"seeds": seeds, "diff_per_seed": {s: round(d, 2) for s, d in zip(seeds, diffs)}}
    if diffs:
        out["mean_diff"] = round(st.mean(diffs), 2)
        out["wins"] = f"{sum(d > 0 for d in diffs)}/{len(diffs)}"
    if len(diffs) > 1:
        sd = st.stdev(diffs)
        t = st.mean(diffs) / (sd / math.sqrt(len(diffs))) if sd > 0 else math.copysign(math.inf, st.mean(diffs))
        out["std_diff"] = round(sd, 2)
        out["paired_t"] = round(t, 2) if math.isfinite(t) else None
        out["paired_t_p_two_sided"] = round(t_two_sided_p(t, len(diffs) - 1), 4)
    return out


def main():
    runs = collect()
    if not runs["mixture"]:
        raise SystemExit("no evaluated mixture runs under results/mixed/")
    summary = {
        "experiment": "Sim+Real mixtures vs real-only controls, YOLOv8n-OBB, 30 epochs, batch 16",
        "synthetic_source": "dataset_controlnet_fg05 train (ControlNet s0.45, fg-keep 0.5)",
        "real_source": "dataset/real_v2 train (captures 0001, 0003, 0004, 0005); nested seed-42 order, equal per capture",
        "checkpoint_selection": "dataset/real_v2 val (capture 0006)",
        "test": "dataset/real_v2 test (captures 0007, 0009; 500 frames, 2,990 instances)",
        "metrics_note": "percent; mean and sample std over training seeds; paired t-tests are two-sided",
        "families": {family: {"description": LABELS[family],
                              "by_real_images": {n: describe(seeds) for n, seeds in sorted(runs[family].items())}}
                     for family in LABELS if runs[family]},
        "mixture_minus_real_only": {},
    }
    for family in ("real_only_steps_matched", "real_only_raw"):
        summary["mixture_minus_real_only"][family] = {
            n: paired(runs["mixture"][n], runs[family][n])
            for n in sorted(set(runs["mixture"]) & set(runs[family])) if n != 800}
    out = RESULTS / "summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    xs_all = sorted(set().union(*[set(r) for r in runs.values()]))
    pos = {n: i for i, n in enumerate(xs_all)}
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    styles = {"mixture": ("#2a6fdb", "o", "-"), "real_only_steps_matched": ("#d9822b", "s", "-"),
              "real_only_raw": ("#9aa5b1", "^", "--")}
    for family, (color, marker, line) in styles.items():
        if not runs[family]:
            continue
        ns = sorted(runs[family])
        stats = [describe(runs[family][n])["overall"]["mAP50-95"] for n in ns]
        ax.errorbar([pos[n] for n in ns], [s["mean"] for s in stats], yerr=[s["std"] or 0 for s in stats],
                    marker=marker, linestyle=line, capsize=4, linewidth=2, color=color, label=LABELS[family])
    ax.set_xticks(range(len(xs_all)))
    ax.set_xticklabels([str(n) for n in xs_all])
    ax.set_xlabel("Real training images (captures 0001/0003/0004/0005; not to scale)")
    ax.set_ylabel("Real test mAP50–95 (%)")
    ax.set_title("Held-out real captures 0007/0009 (mean ± std over 3 training seeds)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS / "mixture_curve.png")

    fmt = lambda v: f"{v['mean']:.2f} ± {v['std']:.2f}" if v["std"] is not None else f"{v['mean']:.2f}"
    for family in LABELS:
        if not runs[family]:
            continue
        print(f"\n{LABELS[family]}")
        print(f"{'real imgs':>9} {'seeds':>5} {'mAP50':>14} {'mAP50-95':>14}")
        for n, seeds in sorted(runs[family].items()):
            o = describe(seeds)["overall"]
            print(f"{n:>9} {len(seeds):>5} {fmt(o['mAP50']):>14} {fmt(o['mAP50-95']):>14}")
    for family, rows in summary["mixture_minus_real_only"].items():
        print(f"\nmixture minus {family} (mAP50-95 points, paired by seed)")
        for n, p in rows.items():
            print(f"{n:>9}  per seed {p['diff_per_seed']}  mean {p.get('mean_diff')}  wins {p.get('wins')}"
                  + (f"  t={p['paired_t']} p={p['paired_t_p_two_sided']}" if "paired_t" in p else ""))
    print(f"\nwrote {out} and {RESULTS / 'mixture_curve.png'}")


if __name__ == "__main__":
    main()
