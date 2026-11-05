"""
prep_real_test_set.py
=====================
Convert the real UR16e camera captures from the Hugging Face dataset
`ABC-iRobotics/LabWare_Items` (config ``instance_segmentation``) into our
YOLOv8-OBB layout::

    dataset/real_test/
        images/<split>/*.png      real 1280x800 RGB frames
        labels/<split>/*.txt      cls x1 y1 x2 y2 x3 y3 x4 y4 (normalised)
        preview/<split>/*.png     ground-truth OBB overlays for inspection
        data.yaml                 Ultralytics dataset descriptor (no `path:`)
        real_test_manifest.json   provenance, class counts, per-frame detail

Each annotation's binary mask is taken from ``instance_mask == instance_id`` and
the tight oriented rectangle is fitted with cv2.minAreaRect over the *visible*
mask pixels -- the same visibility-aware convention our synthetic labels use.
Class mapping (Károly's object_name -> our class id):

    TubeHolder      -> 0  Tube_Holder
    Tube            -> 1  Glass_Tube
    Slide_holder    -> 2  Slide_Holder
    eppendorf_tarto -> 3  Eppendorf_Rack

Anything else is treated as background (skipped and counted in the manifest).

Usage:
    python scripts/prep_real_test_set.py                      # official test split
    python scripts/prep_real_test_set.py --splits test validation
"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from datasets import load_dataset

from yolo_dataset_splitter import CLASS_NAMES, PROJECT_ROOT

HF_REPO = "ABC-iRobotics/LabWare_Items"
HF_CONFIG = "instance_segmentation"
IMAGE_EXT = ".png"

# Károly's Label Factory object_name -> our YOLOv8-OBB class id.
OBJECT_NAME_TO_CLASS = {
    "TubeHolder": 0,
    "Tube": 1,
    "Slide_holder": 2,
    "eppendorf_tarto": 3,
}

MIN_INSTANCE_AREA_PX = 100      # skip specks / heavy-occlusion slivers
MIN_BOX_SIDE_PX = 4.0           # degenerate minAreaRect guard
COLORS = [(0, 200, 255), (0, 255, 0), (255, 0, 255), (255, 160, 0)]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mask_to_obb(instance_mask, instance_id, width, height):
    """Visible-pixel min-area rectangle as normalised corner coords, or None."""
    ys, xs = np.nonzero(instance_mask == instance_id)
    area = int(xs.size)
    if area < MIN_INSTANCE_AREA_PX:
        return None, area
    points = np.stack([xs, ys], axis=1).astype(np.float32)
    rect = cv2.minAreaRect(points)
    if min(rect[1]) < MIN_BOX_SIDE_PX:
        return None, area
    corners = cv2.boxPoints(rect) / [width, height]
    return np.clip(corners, 0.0, 1.0), area


def draw_overlays(image_bgr, entries, names):
    for entry in entries:
        pts = (np.array(entry["corners_norm"]).reshape(4, 2)
               * [image_bgr.shape[1], image_bgr.shape[0]]).astype(int)
        color = COLORS[entry["class_id"] % len(COLORS)]
        cv2.polylines(image_bgr, [pts], isClosed=True, color=color, thickness=2)
        cv2.putText(image_bgr, names[entry["class_id"]], tuple(pts[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    return image_bgr


def main():
    ap = argparse.ArgumentParser(description="HF LabWare_Items -> YOLOv8-OBB real test set")
    ap.add_argument("--splits", nargs="+", default=["test"],
                    choices=("train", "validation", "test"),
                    help="which official HF splits to export (default: test only)")
    ap.add_argument("--output", type=Path, default=PROJECT_ROOT / "dataset" / "real_test")
    ap.add_argument("--num-preview", type=int, default=8,
                    help="GT overlays per split written to preview/<split>/")
    args = ap.parse_args()
    out = args.output.resolve()

    print(f"loading {HF_REPO} config={HF_CONFIG} (HF cache is reused on reruns)")
    hf = load_dataset(HF_REPO, HF_CONFIG)

    manifest = {
        "source": {
            "repo": HF_REPO,
            "config": HF_CONFIG,
            "demo": "instance_mask + annotations_json + instance_names fields",
            "license": "mit",
            "url": "https://huggingface.co/datasets/ABC-iRobotics/LabWare_Items",
        },
        "class_mapping": {k: {"class_id": v, "yolo_name": CLASS_NAMES[v]}
                          for k, v in OBJECT_NAME_TO_CLASS.items()},
        "filters": {"min_instance_area_px": MIN_INSTANCE_AREA_PX,
                    "min_box_side_px": MIN_BOX_SIDE_PX},
        "splits": {},
        "skipped_instances": [],
        "skipped_object_names": {},
    }
    names_by_id = CLASS_NAMES

    for split in args.splits:
        images_dir = out / "images" / split
        labels_dir = out / "labels" / split
        preview_dir = out / "preview" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        frames, class_counts, per_frame = [], {i: 0 for i in CLASS_NAMES}, []
        previews_left = args.num_preview

        for row in hf[split]:
            dataset_id = row["dataset_id"].rstrip("_")
            stem = f"{dataset_id}_{row['frame_id']:04d}"
            image = row["image"].convert("RGB")
            width, height = image.size
            instance_mask = np.asarray(row["instance_mask"])
            annotations = json.loads(row["annotations_json"])

            entries = []
            for ann in annotations:
                object_name = ann["object_name"]
                if object_name not in OBJECT_NAME_TO_CLASS:
                    manifest["skipped_object_names"][object_name] = (
                        manifest["skipped_object_names"].get(object_name, 0) + 1)
                    continue
                corners, area = mask_to_obb(
                    instance_mask, ann["instance_id"], width, height)
                if corners is None:
                    manifest["skipped_instances"].append({
                        "frame": stem, "instance": ann["instance_name"],
                        "object_name": object_name, "visible_area_px": area})
                    continue
                class_id = OBJECT_NAME_TO_CLASS[object_name]
                entries.append({
                    "class_id": class_id,
                    "object_name": object_name,
                    "instance_name": ann["instance_name"],
                    "visible_area_px": area,
                    "corners_norm": [[round(float(c), 6) for c in p] for p in corners],
                })
                class_counts[class_id] += 1

            if not entries:
                continue
            img_path = images_dir / f"{stem}{IMAGE_EXT}"
            lbl_path = labels_dir / f"{stem}.txt"
            image.save(img_path)
            lbl_path.write_text("".join(
                f"{e['class_id']} " + " ".join(f"{c:.6f}" for c in sum(e['corners_norm'], [])) + "\n"
                for e in entries), encoding="utf-8")
            frames.append(stem)
            per_frame.append({"file": img_path.name,
                              "dataset_id": row["dataset_id"],
                              "frame_id": row["frame_id"],
                              "source_image": row["image_file_name"],
                              "instances": entries,
                              "image_sha256": sha256(img_path)})
            if previews_left > 0:
                preview_dir.mkdir(parents=True, exist_ok=True)
                bgr = draw_overlays(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR),
                                    entries, CLASS_NAMES)
                cv2.imwrite(str(preview_dir / img_path.name), bgr)
                previews_left -= 1

        manifest["splits"][split] = {
            "images": len(frames),
            "instances_by_class": {CLASS_NAMES[k]: class_counts[k] for k in sorted(class_counts)},
            "frames": per_frame,
        }
        print(f"{split}: {len(frames)} images, "
              + ", ".join(f"{names_by_id[k]}={class_counts[k]}" for k in sorted(class_counts)))

    yaml_lines = [
        "# Real UR16e camera frames from ABC-iRobotics/LabWare_Items (instance_segmentation).",
        "# `path` intentionally omitted: Ultralytics resolves split dirs relative to this file.",
        "# This is an evaluation-only set: never train on it. Ultralytics requires",
        "# `train:`/`val:` keys, so unused ones alias the first exported split.",
    ]
    first_split = next(iter(manifest["splits"]))
    emitted = {}
    for split in manifest["splits"]:
        key = "val" if split == "validation" else split
        emitted[key] = split
        yaml_lines.append(f"{key}: images/{split}")
    for key in ("train", "val"):
        if key not in emitted:
            yaml_lines.append(f"{key}: images/{first_split}  # unused alias (eval-only set)")
    yaml_lines.append("")
    yaml_lines.append("names:")
    for k in sorted(CLASS_NAMES):
        yaml_lines.append(f"  {k}: {CLASS_NAMES[k]}")
    (out / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    manifest_path = out / "real_test_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest -> {manifest_path}")
    print(f"skipped unknown object_names: {manifest['skipped_object_names'] or 'none'}")
    print(f"skipped degenerate/tiny instances: {len(manifest['skipped_instances'])}")


if __name__ == "__main__":
    main()
