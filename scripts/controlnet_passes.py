"""
controlnet_passes.py
====================
Read the Depth / Normal / Object-Index EXR passes written by
``blender/generate_synthetic_dataset.py --export-passes``, turn them into ControlNet
conditioning images, and QA them against the YOLOv8-OBB labels.

    python scripts/controlnet_passes.py --dataset smoke_controlnet --out results/controlnet/smoke_passes_grid.jpg

Contact sheet: one row per frame -> RGB + GT OBB | ControlNet depth | world normal | object index.
Checks per frame: depth finite and metric, index values are integers that all map to an object
in the scene JSON, and every labelled object's index-mask pixels lie inside its GT OBB.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import OpenEXR

BACKGROUND_DEPTH_M = 1e3      # Cycles writes ~1e10 where a ray hits only the world background
INDEX_COLORS = [(0, 0, 0)] + [tuple(int(c) for c in cv2.cvtColor(
    np.uint8([[[int(h), 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]) for h in np.linspace(0, 179, 24, endpoint=False)[
    np.random.default_rng(3).permutation(24)]]


def read_exr(path):
    """(H, W) float32 for single-channel passes, (H, W, 3) for XYZ normals."""
    with OpenEXR.File(str(path)) as f:
        ch = f.channels()
        if "V" in ch:
            return ch["V"].pixels.astype(np.float32)
        return np.stack([ch[k].pixels for k in ("X", "Y", "Z")], axis=-1).astype(np.float32)


def load_passes(dataset, stem):
    root = Path(dataset) / "passes"
    return {name: read_exr(root / name / f"{stem}.exr") for name in ("depth", "normal", "index")}


def depth_to_control(depth, lo_pct=1.0, hi_pct=99.0):
    """Metric Z -> 8-bit relative inverse depth (near = white), the MiDaS-style encoding the
    SD1.5 depth ControlNet was trained on. Normalised per image; background -> black."""
    valid = depth < BACKGROUND_DEPTH_M
    inv = np.zeros_like(depth)
    inv[valid] = 1.0 / np.maximum(depth[valid], 1e-6)
    if valid.any():
        lo, hi = np.percentile(inv[valid], (lo_pct, hi_pct))
        inv = np.clip((inv - lo) / max(hi - lo, 1e-9), 0.0, 1.0) * valid
    return (inv * 255.0 + 0.5).astype(np.uint8)


def normal_to_rgb(normal):
    return (np.clip(normal * 0.5 + 0.5, 0.0, 1.0)[..., ::-1] * 255.0 + 0.5).astype(np.uint8)


def index_to_rgb(index):
    ids = np.rint(index).astype(np.int32)
    out = np.zeros((*ids.shape, 3), np.uint8)
    for i in np.unique(ids):
        out[ids == i] = INDEX_COLORS[i % len(INDEX_COLORS)] if i else (0, 0, 0)
    return out


def read_obbs(label_path, w, h):
    obbs = []
    for line in (label_path.read_text().splitlines() if label_path.exists() else []):
        p = line.split()
        obbs.append((int(p[0]), (np.array(p[1:], float).reshape(4, 2) * [w, h]).astype(np.float32)))
    return obbs


def check_frame(dataset, stem, passes):
    """Return a dict of QA numbers for one frame."""
    scene = json.loads((Path(dataset) / "scene" / f"{stem}.json").read_text())
    by_index = {o["pass_index"]: o for o in scene["objects"]}
    depth, index = passes["depth"], passes["index"]
    h, w = index.shape
    ids = np.unique(index)
    ids_int = np.rint(ids).astype(int)
    obbs = read_obbs(Path(dataset) / "labels" / "train" / f"{stem}.txt", w, h)
    worst, per_object = 1.0, []
    for pid in ids_int:
        obj = by_index.get(int(pid))
        if not pid or obj is None or not obj["labelled"]:
            continue
        ys, xs = np.nonzero(np.rint(index) == pid)
        # fraction of this object's index pixels within 1.5 px of its best-matching GT OBB
        best = max(np.mean([cv2.pointPolygonTest(poly, (float(x) + 0.5, float(y) + 0.5), True) >= -1.5
                            for x, y in zip(xs[::5], ys[::5])])
                   for cls, poly in obbs if cls == obj["class_id"])
        per_object.append((obj["name"], round(float(best), 4)))
        worst = min(worst, best)
    finite = depth[depth < BACKGROUND_DEPTH_M]
    return {
        "frame": stem,
        "depth_m_min": round(float(finite.min()), 4), "depth_m_max": round(float(finite.max()), 4),
        "background_fraction": round(float((depth >= BACKGROUND_DEPTH_M).mean()), 4),
        "index_ids": [int(i) for i in ids_int],
        "index_max_non_integer": round(float(np.abs(ids - ids_int).max()), 6),
        "index_ids_unknown": [int(i) for i in ids_int if i and int(i) not in by_index],
        "labelled_objects_in_index": len(per_object),
        "worst_mask_in_obb_fraction": round(float(worst), 4),
        "per_object_mask_in_obb": per_object,
    }


def tile(img, size, title):
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    cv2.putText(img, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description="ControlNet pass QA contact sheet")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tile", type=int, default=320)
    args = ap.parse_args()
    stems = sorted(p.stem for p in (args.dataset / "passes" / "depth").glob("*.exr"))
    rows, report = [], []
    for stem in stems:
        passes = load_passes(args.dataset, stem)
        rgb = cv2.imread(str(args.dataset / "images" / "train" / f"{stem}.png"))
        for cls, poly in read_obbs(args.dataset / "labels" / "train" / f"{stem}.txt", *rgb.shape[1::-1]):
            cv2.polylines(rgb, [poly.round().astype(np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA)
        q = check_frame(args.dataset, stem, passes)
        report.append(q)
        d = q["depth_m_min"], q["depth_m_max"]
        rows.append(np.hstack([
            tile(rgb, args.tile, f"{stem} RGB + GT OBB"),
            tile(cv2.cvtColor(depth_to_control(passes["depth"]), cv2.COLOR_GRAY2BGR), args.tile,
                 f"depth ctrl ({d[0]:.2f}-{d[1]:.2f} m)"),
            tile(normal_to_rgb(passes["normal"]), args.tile, "world normal"),
            tile(index_to_rgb(passes["index"]), args.tile, f"object index ({len(q['index_ids']) - 1} objs)"),
        ]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    summary = {
        "frames": len(report),
        "all_index_values_integer": all(q["index_max_non_integer"] < 1e-4 for q in report),
        "unknown_index_ids": sum(len(q["index_ids_unknown"]) for q in report),
        "min_mask_in_obb_fraction": min(q["worst_mask_in_obb_fraction"] for q in report),
        "frames_report": report,
    }
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    for q in report:
        print(f"{q['frame']}: depth {q['depth_m_min']:.3f}-{q['depth_m_max']:.3f} m, bg {q['background_fraction']:.2f}, "
              f"{len(q['index_ids']) - 1} index ids, unknown {q['index_ids_unknown']}, "
              f"labelled masks in OBB >= {q['worst_mask_in_obb_fraction']:.3f}")
    print({k: v for k, v in summary.items() if k != "frames_report"})
    print(f"contact sheet -> {args.out}")


if __name__ == "__main__":
    main()
