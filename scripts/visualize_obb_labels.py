"""
visualize_obb_labels.py
=======================
Draw YOLOv8-OBB label polygons on top of their images so a synthetic dataset can be
sanity-checked by eye before spending GPU hours on training.

    python scripts/visualize_obb_labels.py --dataset dataset --num 12
    python scripts/visualize_obb_labels.py --dataset dataset_smoke --all

Writes annotated copies to <dataset>/preview/ (never touches the originals).
Label format expected per line: `cls x1 y1 x2 y2 x3 y3 x4 y4` (normalised corners).
"""

import argparse
import random
import re
from pathlib import Path

import cv2
import numpy as np

COLORS = [(0, 200, 255), (0, 255, 0), (255, 0, 255), (255, 160, 0), (0, 0, 255), (255, 255, 0)]


def read_class_names(dataset):
    """Parse the `names:` block of data.yaml without needing PyYAML."""
    names, in_names = {}, False
    yaml_path = dataset / "data.yaml"
    if not yaml_path.exists():
        return names
    for line in yaml_path.read_text().splitlines():
        if line.startswith("names:"):
            in_names = True
            continue
        m = re.match(r"\s+(\d+):\s*(\S+)", line) if in_names else None
        if m:
            names[int(m.group(1))] = m.group(2)
        elif in_names and line.strip() and not line.startswith(" "):
            break
    return names


def main():
    ap = argparse.ArgumentParser(description="Overlay YOLOv8-OBB labels on dataset images.")
    ap.add_argument("--dataset", default="dataset", help="dataset root containing images/ and labels/")
    ap.add_argument("--split", default="train")
    ap.add_argument("--num", type=int, default=12, help="how many random images to annotate")
    ap.add_argument("--all", action="store_true", help="annotate every image in the split")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.dataset)
    names = read_class_names(root)
    images = sorted((root / "images" / args.split).glob("*.*"))
    if not images:
        raise SystemExit(f"no images found under {root / 'images' / args.split}")
    if not args.all:
        random.seed(args.seed)
        images = random.sample(images, min(args.num, len(images)))
    out_dir = root / "preview"
    out_dir.mkdir(exist_ok=True)

    for img_path in images:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        lbl_path = root / "labels" / args.split / f"{img_path.stem}.txt"
        n = 0
        for line in (lbl_path.read_text().splitlines() if lbl_path.exists() else []):
            parts = line.split()
            if len(parts) != 9:
                continue
            cls = int(parts[0])
            pts = (np.array(parts[1:], dtype=float).reshape(4, 2) * [w, h]).astype(int)
            color = COLORS[cls % len(COLORS)]
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
            cv2.putText(img, names.get(cls, str(cls)), tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color, 1, cv2.LINE_AA)
            n += 1
        cv2.imwrite(str(out_dir / img_path.name), img)
        print(f"{img_path.name}: {n} boxes -> {out_dir / img_path.name}")


if __name__ == "__main__":
    main()
