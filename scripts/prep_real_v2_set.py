"""
prep_real_v2_set.py
===================
Capture-level real dataset v2 for the Sim2Real experiments.

Loads *all three* official splits of ``ABC-iRobotics/LabWare_Items``
(config ``instance_segmentation``), groups frames by ``dataset_id`` (the
capture), keeps only captures whose dominant per-frame annotation
composition is exactly

    1 TubeHolder, 2 Tube, 2 Slide_holder, 1 eppendorf_tarto

and writes ``dataset/real_v2/`` in the same YOLOv8-OBB layout and with the
same mask -> ``cv2.minAreaRect`` visible-pixel label conversion as
``scripts/prep_real_test_set.py``.  Splits are assigned **by capture**, never
by frame::

    train = captures 0001, 0003, 0004, 0005
    val   = capture  0006
    test  = captures 0007, 0009

Every frame of each kept capture is exported (frames whose surviving
instances convert to zero OBB labels are written with an empty label file,
matching YOLO background-image semantics).  ``real_v2_manifest.json``
records the capture assignments, per-split image/instance counts, per-frame
detail and image SHA256 hashes; the script asserts that no ``dataset_id``
appears in more than one split.

Usage:
    python scripts/prep_real_v2_set.py
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from datasets import load_dataset

from prep_real_test_set import (OBJECT_NAME_TO_CLASS, HF_REPO, HF_CONFIG,
                                IMAGE_EXT, sha256, mask_to_obb, draw_overlays)
from yolo_dataset_splitter import CLASS_NAMES, PROJECT_ROOT

# Supervisor's capture rule: dominant composition must be 1/2/2/1.
REQUIRED_COMPOSITION = {"TubeHolder": 1, "Tube": 2, "Slide_holder": 2,
                        "eppendorf_tarto": 1}
EXPECTED_KEEP = {"0001", "0003", "0004", "0005", "0006", "0007", "0009"}

CAPTURE_SPLITS = {
    "train": ["0001", "0003", "0004", "0005"],
    "val": ["0006"],
    "test": ["0007", "0009"],
}
CONTACT_FRAMES_PER_CAPTURE = 3


def main():
    out = (PROJECT_ROOT / "dataset" / "real_v2").resolve()
    if out.exists():
        raise FileExistsError(f"refusing to write into existing directory: {out}")
    sheet_path = (PROJECT_ROOT / "results" / "real_v2" / "gt_contact_sheet.jpg")
    if sheet_path.exists():
        raise FileExistsError(f"refusing to overwrite {sheet_path}")

    print(f"loading {HF_REPO} config={HF_CONFIG} (all splits)")
    hf = load_dataset(HF_REPO, HF_CONFIG)

    # ---- group every frame by capture ------------------------------------
    captures = defaultdict(list)
    for official_split in hf:
        for row in hf[official_split]:
            captures[row["dataset_id"].rstrip("_")].append(
                (official_split, row))

    # ---- per-capture report + composition filter --------------------------
    report = {}
    keep = set()
    for cid in sorted(captures):
        rows = captures[cid]
        total_objs = Counter()
        frame_sigs = Counter()
        for _, row in rows:
            cnt = Counter(a["object_name"] for a in json.loads(row["annotations_json"]))
            total_objs.update(cnt)
            frame_sigs[tuple(sorted(cnt.items()))] += 1
        modal_sig = dict(frame_sigs.most_common(1)[0][0])
        passes = modal_sig == REQUIRED_COMPOSITION
        report[cid] = {"frames": len(rows), "instances_per_object_name": dict(sorted(total_objs.items())),
                       "modal_frame_composition": modal_sig, "kept": passes}
        print(f"{cid}: {len(rows)} frames, objects={dict(sorted(total_objs.items()))}, "
              f"modal={modal_sig} -> {'KEEP' if passes else 'drop'}")
        if passes:
            keep.add(cid)

    if keep != EXPECTED_KEEP:
        raise SystemExit(f"capture filter result {sorted(keep)} differs from expected "
                         f"{sorted(EXPECTED_KEEP)} — stopping, no files written")

    # ---- split assignment; assert disjointness ----------------------------
    assignment = {}
    for split, cids in CAPTURE_SPLITS.items():
        for cid in cids:
            assert cid in keep, f"assigned capture {cid} was dropped by the filter"
            assert cid not in assignment, f"capture {cid} in two splits"
            assignment[cid] = split
    assert set(assignment) == keep, "capture split assignment must cover all kept captures"

    # ---- export ------------------------------------------------------------
    manifest = {
        "source": {
            "repo": HF_REPO, "config": HF_CONFIG, "license": "mit",
            "url": "https://huggingface.co/datasets/ABC-iRobotics/LabWare_Items",
            "note": "capture-level export over all three official splits; "
                    "splits assigned by dataset_id (capture), never by frame",
        },
        "class_mapping": {k: {"class_id": v, "yolo_name": CLASS_NAMES[v]}
                          for k, v in OBJECT_NAME_TO_CLASS.items()},
        "capture_rule": {
            "required_modal_composition": REQUIRED_COMPOSITION,
            "expected_keep": sorted(EXPECTED_KEEP),
        },
        "capture_report": report,
        "capture_split_assignment": assignment,
        "splits": {},
        "skipped_instances": [],
        "skipped_object_names": {},
    }

    tiles = []
    for split, cids in CAPTURE_SPLITS.items():
        images_dir = out / "images" / split
        labels_dir = out / "labels" / split
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        class_counts = {i: 0 for i in CLASS_NAMES}
        per_frame = []
        n_images = 0
        for cid in cids:
            contact_left = CONTACT_FRAMES_PER_CAPTURE
            for official_split, row in sorted(captures[cid],
                                              key=lambda t: (t[0], t[1]["frame_id"])):
                stem = f"{cid}_{row['frame_id']:04d}"
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
                    corners, area = mask_to_obb(instance_mask, ann["instance_id"],
                                              width, height)
                    if corners is None:
                        manifest["skipped_instances"].append({
                            "frame": stem, "instance": ann["instance_name"],
                            "object_name": object_name, "visible_area_px": area})
                        continue
                    entries.append({
                        "class_id": OBJECT_NAME_TO_CLASS[object_name],
                        "object_name": object_name,
                        "instance_name": ann["instance_name"],
                        "visible_area_px": area,
                        "corners_norm": [[round(float(c), 6) for c in p] for p in corners],
                    })
                    class_counts[OBJECT_NAME_TO_CLASS[object_name]] += 1

                img_path = images_dir / f"{stem}{IMAGE_EXT}"
                image.save(img_path)
                (labels_dir / f"{stem}.txt").write_text(
                    "".join(f"{e['class_id']} "
                            + " ".join(f"{c:.6f}" for c in sum(e["corners_norm"], [])) + "\n"
                            for e in entries), encoding="utf-8")
                n_images += 1
                per_frame.append({"file": img_path.name,
                                  "capture": cid,
                                  "source_official_split": official_split,
                                  "frame_id": row["frame_id"],
                                  "instances": entries,
                                  "image_sha256": sha256(img_path)})
                if contact_left > 0:
                    bgr = draw_overlays(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR),
                                        entries, CLASS_NAMES)
                    cv2.putText(bgr, f"{split}/{cid}", (8, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    tiles.append(cv2.resize(bgr, (640, 400)))
                    contact_left -= 1

        manifest["splits"][split] = {
            "captures": cids,
            "images": n_images,
            "instances_by_class": {CLASS_NAMES[k]: class_counts[k] for k in sorted(class_counts)},
            "frames": per_frame,
        }
        print(f"{split}: {n_images} images, "
              + ", ".join(f"{CLASS_NAMES[k]}={class_counts[k]}" for k in sorted(class_counts)))

    # ---- data.yaml ----------------------------------------------------------
    yaml_lines = [
        "# Capture-level real UR16e frames from ABC-iRobotics/LabWare_Items (instance_segmentation).",
        "# `path` intentionally omitted: Ultralytics resolves split dirs relative to this file.",
        "# Splits are by capture (dataset_id): train=0001,0003,0004,0005 val=0006 test=0007,0009.",
    ]
    for split in CAPTURE_SPLITS:
        yaml_lines.append(f"{split}: images/{split}")
    yaml_lines.append("")
    yaml_lines.append("names:")
    for k in sorted(CLASS_NAMES):
        yaml_lines.append(f"  {k}: {CLASS_NAMES[k]}")
    (out / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    manifest_path = out / "real_v2_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest -> {manifest_path}")
    print(f"skipped unknown object_names: {manifest['skipped_object_names'] or 'none'}")
    print(f"skipped degenerate/tiny instances: {len(manifest['skipped_instances'])}")

    # ---- GT contact sheet ----------------------------------------------------
    if tiles:
        sheet_path.parent.mkdir(parents=True, exist_ok=True)
        cols = CONTACT_FRAMES_PER_CAPTURE
        rows = (len(tiles) + cols - 1) // cols
        sheet = np.zeros((rows * 400, cols * 640, 3), np.uint8)
        for i, tile in enumerate(tiles):
            r, c = divmod(i, cols)
            sheet[r*400:(r+1)*400, c*640:(c+1)*640] = tile
        cv2.imwrite(str(sheet_path), sheet)
        print(f"contact sheet -> {sheet_path} ({len(tiles)} tiles)")


if __name__ == "__main__":
    main()
