"""
build_mixed_datasets.py
=======================
Build the Sim+Real mixture training sets for the mixture experiment.

Every mixture has exactly 800 training images.  The real share is 0, 20, 40,
50, 60, 80 or 100 percent; the remainder is synthetic:

    synthetic  dataset_controlnet_fg05/images/train   (800 stylized renders)
    real       dataset/real_v2/images/train           (captures 0001/0003/0004/0005)

Each source has ONE fixed shuffled order (seed 42), so the subsets are nested:
a larger real share contains every real image of a smaller one, and a larger
synthetic share contains every synthetic image of a smaller one.  The real
order interleaves the four training captures round-robin, so any real count
divisible by four takes the same number of frames from each capture.

No image or label is copied.  Each mixture is a folder

    mixed_datasets/real<NN>/
        train.txt              one image path per line, relative to this folder
        data.yaml              train = train.txt, val/test = dataset/real_v2 val/test
        mixture_manifest.json  every image, its source and capture, class counts

Ultralytics finds each label by replacing the last ``/images/`` path component
with ``/labels/``, so the labels stay in their source datasets as well.

Usage:
    python scripts/build_mixed_datasets.py        # refuses existing mixture folders
"""

import json
import random
from pathlib import Path

from yolo_dataset_splitter import CLASS_NAMES, PROJECT_ROOT

SEED = 42
TOTAL = 800
REAL_SHARES = (0, 20, 40, 50, 60, 80, 100)
SYNTH_DIR = PROJECT_ROOT / "dataset_controlnet_fg05"
REAL_DIR = PROJECT_ROOT / "dataset" / "real_v2"
OUT_ROOT = PROJECT_ROOT / "mixed_datasets"
TRAIN_CAPTURES = ("0001", "0003", "0004", "0005")


def count_classes(label_path):
    counts = [0] * len(CLASS_NAMES)
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            counts[int(line.split()[0])] += 1
    return counts


def ordered_sources():
    """Fixed seed-42 orders: (synthetic image paths, real image paths)."""
    synth = sorted((SYNTH_DIR / "images" / "train").glob("*.png"))
    if len(synth) != TOTAL:
        raise ValueError(f"expected {TOTAL} synthetic train images, found {len(synth)}")
    random.Random(SEED).shuffle(synth)

    real_manifest = json.loads((REAL_DIR / "real_v2_manifest.json").read_text(encoding="utf-8"))
    if tuple(real_manifest["splits"]["train"]["captures"]) != TRAIN_CAPTURES:
        raise ValueError("real_v2 train captures differ from the expected four")
    rng = random.Random(SEED)
    per_capture = []
    for capture in TRAIN_CAPTURES:
        frames = sorted((REAL_DIR / "images" / "train").glob(f"{capture}_*.png"))
        if len(frames) < TOTAL // len(TRAIN_CAPTURES):
            raise ValueError(f"capture {capture} has only {len(frames)} frames")
        rng.shuffle(frames)
        per_capture.append(frames)
    depth = min(map(len, per_capture))
    real = [frames[i] for i in range(depth) for frames in per_capture]
    return synth, real


def main():
    synth, real = ordered_sources()
    OUT_ROOT.mkdir(exist_ok=True)
    for share in REAL_SHARES:
        n_real = TOTAL * share // 100
        n_synth = TOTAL - n_real
        if n_real % len(TRAIN_CAPTURES):
            raise ValueError(f"{n_real} real images cannot be split equally over the captures")
        out = OUT_ROOT / f"real{share:02d}"
        out.mkdir(exist_ok=False)

        entries, class_counts = [], {"synthetic": [0] * len(CLASS_NAMES), "real": [0] * len(CLASS_NAMES)}
        per_capture = {capture: 0 for capture in TRAIN_CAPTURES}
        for source, images in (("real", real[:n_real]), ("synthetic", synth[:n_synth])):
            for image in images:
                label = image.parent.parent.parent / "labels" / image.parent.name / f"{image.stem}.txt"
                if not label.is_file():
                    raise FileNotFoundError(label)
                counts = count_classes(label)
                class_counts[source] = [a + b for a, b in zip(class_counts[source], counts)]
                capture = image.stem.split("_")[0] if source == "real" else None
                if capture:
                    per_capture[capture] += 1
                entries.append({
                    "image": image.relative_to(PROJECT_ROOT).as_posix(),
                    "label": label.relative_to(PROJECT_ROOT).as_posix(),
                    "source": source,
                    "source_dataset": (REAL_DIR if source == "real" else SYNTH_DIR).relative_to(PROJECT_ROOT).as_posix(),
                    "capture": capture,
                    "instances_by_class": counts,
                })
        if len(entries) != TOTAL or len({e["image"] for e in entries}) != TOTAL:
            raise ValueError("mixture must contain exactly 800 distinct images")
        if n_real and len(set(per_capture.values())) != 1:
            raise ValueError(f"unequal capture counts: {per_capture}")

        # "./" lines are resolved by Ultralytics relative to the folder holding train.txt.
        (out / "train.txt").write_text(
            "".join(f"./../../{e['image']}\n" for e in entries), encoding="utf-8", newline="\n")
        (out / "data.yaml").write_text("\n".join([
            f"# Sim+Real mixture: {n_real} real + {n_synth} synthetic training images ({share}% real).",
            "# `path` intentionally omitted: Ultralytics resolves entries relative to this file.",
            "# Validation and test are always the real_v2 capture-level splits (val 0006, test 0007/0009).",
            "train: train.txt",
            "val: ../../dataset/real_v2/images/val",
            "test: ../../dataset/real_v2/images/test",
            "",
            "names:",
            *[f"  {k}: {CLASS_NAMES[k]}" for k in sorted(CLASS_NAMES)],
        ]) + "\n", encoding="utf-8", newline="\n")
        manifest = {
            "mixture": out.name,
            "real_share_percent": share,
            "total_images": TOTAL,
            "real_images": n_real,
            "synthetic_images": n_synth,
            "order_seed": SEED,
            "nesting": "first N of one fixed seed-42 order per source; real order interleaves the captures round-robin",
            "real_images_per_capture": per_capture,
            "sources": {"synthetic": "dataset_controlnet_fg05/images/train", "real": "dataset/real_v2/images/train"},
            "val": "dataset/real_v2/images/val (capture 0006)",
            "test": "dataset/real_v2/images/test (captures 0007, 0009)",
            "class_names": [CLASS_NAMES[k] for k in sorted(CLASS_NAMES)],
            "instances_by_class": class_counts,
            "images": entries,
        }
        with (out / "mixture_manifest.json").open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
        print(f"{out.name}: real={n_real} {per_capture} synthetic={n_synth} "
              f"instances real={class_counts['real']} synthetic={class_counts['synthetic']}")


if __name__ == "__main__":
    main()
