"""
make_contact_sheet.py
=====================
Tile rendered frames with their ground-truth YOLOv8-OBB boxes into one grid image for
visual QA of a generator change.

    python scripts/make_contact_sheet.py --dataset dataset_geometry_test --out results/geometry_test_grid.jpg
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from visualize_obb_labels import COLORS, read_class_names


def annotate(img_path, lbl_path, names):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    for line in (lbl_path.read_text().splitlines() if lbl_path.exists() else []):
        parts = line.split()
        if len(parts) != 9:
            continue
        cls = int(parts[0])
        pts = (np.array(parts[1:], dtype=float).reshape(4, 2) * [w, h]).round().astype(np.int32)
        color = COLORS[cls % len(COLORS)]
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(img, names.get(cls, str(cls)), tuple(int(v) for v in pts[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.putText(img, img_path.stem, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description="Grid of frames with ground-truth OBB overlays")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--tile", type=int, default=640, help="tile edge in pixels")
    args = ap.parse_args()

    names = read_class_names(args.dataset)
    images = sorted((args.dataset / "images" / args.split).glob("*.png"))[: args.cols * args.rows]
    if not images:
        raise SystemExit("no images found")
    tiles = [cv2.resize(annotate(p, args.dataset / "labels" / args.split / f"{p.stem}.txt", names),
                        (args.tile, args.tile), interpolation=cv2.INTER_AREA) for p in images]
    blank = np.zeros_like(tiles[0])
    tiles += [blank] * (args.cols * args.rows - len(tiles))
    grid = np.vstack([np.hstack(tiles[r * args.cols:(r + 1) * args.cols]) for r in range(args.rows)])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"{len(images)} frames -> {args.out} ({grid.shape[1]}x{grid.shape[0]})")


if __name__ == "__main__":
    main()
