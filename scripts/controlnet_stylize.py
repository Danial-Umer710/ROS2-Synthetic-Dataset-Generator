"""
controlnet_stylize.py
=====================
Stylize rendered frames with Multi-ControlNet img2img (SD1.5 Realistic Vision +
depth + canny ControlNets, fp16) and composite the original labelled-object pixels
back inside the object-index mask so the YOLOv8-OBB labels stay exactly valid.

    # single configuration
    python scripts/controlnet_stylize.py --dataset smoke_controlnet --strength 0.45

    # parameter grid + comparison sheet
    python scripts/controlnet_stylize.py --dataset smoke_controlnet --grid 0.35 0.45 0.55 \
        --sheet results/controlnet/stylize_grid_comparison.jpg

Layout of an output config dir: <out>/<config>/images/<stem>.png plus a
<config>_manifest.json recording every parameter and the per-frame seeds.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from controlnet_passes import load_passes, depth_to_control, read_obbs
from controlnet_setup import load_pipeline, local_paths, VRAM_BUDGET_GIB

PROMPT = ("photo of a laboratory bench with plastic centrifuge tubes, a tube rack and labware, "
          "translucent glass and polypropylene, lab lighting, DSLR photo, sharp focus")
NEGATIVE = ("cartoon, render, cgi, illustration, blurry, deformed, extra objects, missing objects, "
            "text, watermark, logo")
MASK_DILATE_PX = 3
MASK_FEATHER_PX = 2


def index_edges(index):
    """Geometry-only Canny-style edge map (white lines on black) for the canny ControlNet:
    edges of the object-index map carry the true silhouettes without shading/texture edges."""
    ids = np.rint(index).astype(np.uint8)
    edges = np.zeros(index.shape, np.uint8)
    for i in np.unique(ids):
        if i == 0:
            continue
        m = (ids == i).astype(np.uint8)
        edges |= (cv2.dilate(m, np.ones((3, 3), np.uint8)) - m).astype(np.uint8)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    return np.repeat((edges * 255)[..., None], 3, axis=2)


def fg_alpha(dataset, stem, index):
    """Feathered alpha mask of all labelled objects from the object-index pass."""
    scene = json.loads((Path(dataset) / "scene" / f"{stem}.json").read_text())
    labelled = {o["pass_index"] for o in scene["objects"] if o["labelled"]}
    m = np.isin(np.rint(index).astype(np.int32), list(labelled)).astype(np.float32)
    m = cv2.dilate(m, np.ones((2 * MASK_DILATE_PX + 1,) * 2, np.uint8))
    return cv2.GaussianBlur(m, (2 * MASK_FEATHER_PX + 1,) * 2, 0)


def stylize_frame(pipe, dataset, stem, strength, cfg):
    rgb = np.asarray(Image.open(Path(dataset) / "images" / "train" / f"{stem}.png").convert("RGB"))
    passes = load_passes(dataset, stem)
    depth_ctrl = Image.fromarray(np.repeat(depth_to_control(passes["depth"])[..., None], 3, axis=2))
    edge_ctrl = Image.fromarray(index_edges(passes["index"]))
    h, w = rgb.shape[:2]
    gen = torch.Generator("cuda").manual_seed(cfg["seed_base"] * 1_000_003 + int(stem))
    out = pipe(prompt=cfg["prompt"], negative_prompt=cfg["negative"],
               image=Image.fromarray(rgb), control_image=[depth_ctrl, edge_ctrl],
               strength=strength, num_inference_steps=cfg["steps"],
               guidance_scale=cfg["guidance_scale"],
               controlnet_conditioning_scale=cfg["control_scale"],
               control_guidance_end=cfg["control_end"],
               width=w, height=h, generator=gen).images[0]
    styl = np.asarray(out, dtype=np.float32)
    alpha = fg_alpha(dataset, stem, passes["index"])[..., None] * cfg["fg_keep"]
    return np.clip(alpha * rgb + (1.0 - alpha) * styl + 0.5, 0, 255).astype(np.uint8)


def run_config(pipe, dataset, stems, cfg, out_root):
    name = f"s{cfg['strength']:.2f}_fg{cfg['fg_keep']:.2f}".replace(".", "")
    images_dir = out_root / name / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    times, records = [], []
    for stem in stems:
        t = time.time()
        img = stylize_frame(pipe, dataset, stem, cfg["strength"], cfg)
        times.append(time.time() - t)
        cv2.imwrite(str(images_dir / f"{stem}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        records.append({"stem": stem, "seed": cfg["seed_base"] * 1_000_003 + int(stem)})
    manifest = {**cfg, "models": {k: f"{r[0]}@{r[1][:10]}" for k, r in
                                  __import__("controlnet_setup").MODELS.items()},
                "frames": records,
                "seconds_per_image": round(float(np.mean(times)), 2),
                "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2)}
    manifest["peak_reserved_gib_fits_budget"] = manifest["peak_reserved_gib"] <= VRAM_BUDGET_GIB
    (out_root / f"{name}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{name}: {len(stems)} frames, {manifest['seconds_per_image']} s/img, "
          f"peak {manifest['peak_reserved_gib']} GiB")
    return name


def comparison_sheet(dataset, stems, names, out_root, sheet_path, tile=480):
    from visualize_obb_labels import COLORS, read_class_names
    class_names = read_class_names(dataset)
    rows = []
    for stem in stems:
        cells = []
        for col, name in enumerate([None] + names):
            if name is None:
                img = cv2.imread(str(dataset / "images" / "train" / f"{stem}.png"))
                title = f"{stem} original"
            else:
                img = cv2.imread(str(out_root / name / "images" / f"{stem}.png"))
                title = f"strength {name.split('_')[0][1:3]}.{name.split('_')[0][3:]} fg-keep {name.split('_fg')[1][0]}.{name.split('_fg')[1][1:]}"
            h, w = img.shape[:2]
            for cls, poly in read_obbs(dataset / "labels" / "train" / f"{stem}.txt", w, h):
                cv2.polylines(img, [poly.round().astype(np.int32)], True,
                              COLORS[cls % len(COLORS)], 2, cv2.LINE_AA)
            img = cv2.resize(img, (tile, tile), interpolation=cv2.INTER_AREA)
            cv2.putText(img, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cells.append(img)
        rows.append(np.hstack(cells))
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(sheet_path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"comparison sheet -> {sheet_path}")


def main():
    ap = argparse.ArgumentParser(description="Multi-ControlNet img2img stylization with label-safe compositing")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--strength", type=float, default=0.45)
    ap.add_argument("--grid", type=float, nargs="*", default=None)
    ap.add_argument("--fg-keep", type=float, default=1.0,
                    help="original-pixel blend inside labelled masks (1.0 = hard label guarantee)")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance", type=float, default=6.0)
    ap.add_argument("--depth-scale", type=float, default=1.0)
    ap.add_argument("--edge-scale", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None,
                    help="output root (default <dataset>/stylized)")
    ap.add_argument("--sheet", type=Path, default=None, help="write a comparison contact sheet")
    args = ap.parse_args()
    dataset = args.dataset.resolve()
    out_root = (args.out or dataset / "stylized").resolve()
    stems = sorted(p.stem for p in (dataset / "images" / "train").glob("*.png"))
    cfg = {"prompt": PROMPT, "negative": NEGATIVE, "steps": args.steps,
           "guidance_scale": args.guidance, "control_scale": [args.depth_scale, args.edge_scale],
           "control_end": [1.0, 0.8], "fg_keep": args.fg_keep, "seed_base": args.seed}
    pipe = load_pipeline(local_paths())
    strengths = args.grid if args.grid else [args.strength]
    names = [run_config(pipe, dataset, stems, {**cfg, "strength": s}, out_root) for s in strengths]
    if args.sheet:
        comparison_sheet(dataset, stems, names, out_root, args.sheet)


if __name__ == "__main__":
    main()
