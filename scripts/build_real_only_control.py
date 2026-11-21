"""
build_real_only_control.py
==========================
Build real-only controls for the Sim+Real mixtures: the SAME N real images as the
mixture with N real images, and no synthetic images.

Two variants per N:

* raw (``realonly_<NNN>/``): each image listed once.  A 30-epoch run then takes far
  fewer training steps than an 800-image mixture (and, below 100 batches in total,
  never leaves the learning-rate warm-up), so it confounds data with training length.
* step-matched (``realonly_<NNN>_x800/``): the N images are repeated in order until
  the list has 800 lines, so a 30-epoch run takes the same number of training steps
  as every 800-image mixture.  When 800 is not a multiple of N, the final partial
  pass uses the first images of the fixed order (which is round-robin over the
  four captures, so captures stay balanced whenever the remainder is a multiple of 4).

``mixed_datasets/real20_only/`` (the first control: 160 images x 5) is the
step-matched control for N = 160 and is built with ``--legacy-real20``.

    python scripts/build_real_only_control.py --n 16 40 80 160 320          # raw
    python scripts/build_real_only_control.py --n 16 40 80 320 --steps-matched
    python scripts/build_real_only_control.py --legacy-real20
"""

import argparse
import json
from pathlib import Path

from yolo_dataset_splitter import PROJECT_ROOT

TOTAL = 800
MIXED = PROJECT_ROOT / "mixed_datasets"
# The mixture whose real part supplies the N real images.
SOURCE_FOR_N = {16: "real02", 40: "real05", 80: "real10", 160: "real20", 320: "real40"}


def build(n, steps_matched, out):
    source_dir = MIXED / SOURCE_FOR_N[n]
    source = json.loads((source_dir / "mixture_manifest.json").read_text(encoding="utf-8"))
    real = [e for e in source["images"] if e["source"] == "real"]
    if len(real) != n or source["real_images"] != n:
        raise ValueError(f"{source_dir.name} does not hold exactly {n} real images")
    entries = [real[i % n] for i in range(TOTAL)] if steps_matched else list(real)
    out.mkdir(exist_ok=False)
    (out / "train.txt").write_text("".join(f"./../../{e['image']}\n" for e in entries),
                                   encoding="utf-8", newline="\n")
    yaml = (source_dir / "data.yaml").read_text(encoding="utf-8").splitlines()
    yaml[0] = (f"# Real-only control: the {n} real images of {source_dir.name}, no synthetic"
               + (f", repeated to {TOTAL} lines (step-matched)." if steps_matched else ", each listed once."))
    (out / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8", newline="\n")
    per_capture = {}
    for e in entries:
        per_capture[e["capture"]] = per_capture.get(e["capture"], 0) + 1
    manifest = {
        "mixture": out.name,
        "control_for": source_dir.name,
        "real_share_percent": 100,
        "total_images": len(entries),
        "distinct_images": n,
        "steps_matched": steps_matched,
        "repeat": round(len(entries) / n, 3),
        "real_images": len(entries),
        "synthetic_images": 0,
        "order_seed": source["order_seed"],
        "real_entries_per_capture": per_capture,
        "sources": {"real": source["sources"]["real"]},
        "val": source["val"],
        "test": source["test"],
        "class_names": source["class_names"],
        "images": entries,
    }
    with (out / "mixture_manifest.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(f"{out.name}: {n} distinct real images, {len(entries)} lines, per capture {per_capture}")


def main():
    parser = argparse.ArgumentParser(description="Build real-only controls for the Sim+Real mixtures.")
    parser.add_argument("--n", type=int, nargs="+", choices=sorted(SOURCE_FOR_N), default=[])
    parser.add_argument("--steps-matched", action="store_true",
                        help=f"repeat the N images to {TOTAL} lines")
    parser.add_argument("--legacy-real20", action="store_true",
                        help="build mixed_datasets/real20_only (160 images x 5)")
    args = parser.parse_args()
    if args.legacy_real20:
        build(160, True, MIXED / "real20_only")
    for n in args.n:
        build(n, args.steps_matched, MIXED / (f"realonly_{n:03d}" + ("_x800" if args.steps_matched else "")))


if __name__ == "__main__":
    main()
