# UR16e Sim2Real Labware Detection

**A diffusion-stylized synthetic-data pipeline for oriented detection of laboratory objects — narrowing the Sim2Real gap by repainting geometry-true Blender renders with a frozen ControlNet model.**

> **Branch Structure**
> - **`controlnet` (this branch):** The `blenderproc` pipeline **plus** the diffusion realism stage described in [ControlNet realism stage](#controlnet-realism-stage-this-branch): EXR geometry passes exported from Blender (`--export-passes`), Multi-ControlNet img2img stylization (`scripts/controlnet_stylize.py`), label-safe foreground compositing, two stylized 1,000-image datasets (`dataset_controlnet/`, `dataset_controlnet_fg05/`), two trained models (`weights/controlnet/`, `weights/controlnet_fg05/`), and their synthetic + real-camera evaluations (`results/controlnet/`). Best real-world score to date: **56.56% mAP50–95** on 282 real UR16e frames.
> - **`blenderproc`:** The `main` baseline **plus** distractor/hindrance objects and loose tubes (see [Distractor and loose-tube extension](#distractor-and-loose-tube-extension-inherited-from-blenderproc)), a second dataset (`dataset_blenderproc/`), a second model (`weights/blenderproc/`), and the real-camera Sim2Real benchmark (`dataset/real_test/`).
> - **`main`:** Clean v2 CAD/STL procedural Blender baseline, 1,000-image 80/10/10 split (`dataset/`), trained YOLOv8n-OBB weights (`weights/`), complete results (`results/baseline/`).
> - **`dev`:** Active development branch for upcoming Sim2Real improvements (Docker reproducibility, further scene variation).
> - **`legacy-v1`:** Archived v1 Blender HTTP Server pipeline, the 4,000-image `YOLO_Main_Dataset`, the 50-epoch `ur16e_main_run_4000-3` training run (`99.5% mAP@50`), and `predict-12` test predictions.

This project connects STL geometry, procedural Blender rendering, geometry-derived oriented bounding-box (OBB) annotations, diffusion-based appearance transfer, and YOLOv8n-OBB training. Its intended application is laboratory perception for a UR16e robot. Each published experiment uses **1,000 synthetic images**, a fixed **800/100/100 train/validation/test split**, and a fresh **30-epoch** fine-tuning run from the original DOTA-pretrained `yolov8n-obb.pt`.

**Real-camera results on the 282-frame real test set** ([`dataset/real_test/`](dataset/real_test/), all models evaluated identically — the number that matters on this branch):

| Model | Training data | Real mAP50 | Real mAP50–95 |
|---|---|---:|---:|
| `weights/best.pt` | 800 clean synthetic | 49.38% | 37.28% |
| `weights/blenderproc/best.pt` | 800 synthetic + distractors/loose tubes | 66.30% | 48.91% |
| `weights/controlnet/best.pt` | 800 ControlNet-stylized, fg-keep 1.0 | 63.43% | 48.04% |
| `weights/controlnet_fg05/best.pt` | 800 ControlNet-stylized, fg-keep 0.5 | **73.93%** | **56.56%** |

Keeping 50% of the original rendered pixels inside the labelled-object masks (`fg-keep 0.5`) beats both the fully stylized variant and the previous best model: **+7.65 mAP50–95 points** over the distractor model, and **+19.3 points** over the clean baseline. `Glass_Tube`, the class that drove most of the gap, recovers from 2.0% (baseline) / 29.3% (distractor) to **38.4%** real mAP50–95.

> **Scope:** real-camera scores measure detection against auto-generated instance-mask ground truth on one benchmark session, not robot grasp success. Synthetic holdout scores (~88%) remain same-renderer-domain numbers. An image-space OBB is not a calibrated 3D pose or a guaranteed grasp point.

![Original render vs ControlNet-stylized variants with unchanged GT boxes](results/controlnet/stylize_grid_comparison.jpg)

*Left to right: original Cycles render, then img2img at strength 0.35 / 0.45 / 0.55 with fg-keep 1.0 — ground-truth OBBs (drawn on top) are identical in every column.*

## Contents

- [ControlNet realism stage (this branch)](#controlnet-realism-stage-this-branch)
  - [Why diffusion stylization](#why-diffusion-stylization)
  - [Pipeline at a glance](#pipeline-at-a-glance)
  - [Step 1 — geometry passes from the same render](#step-1--geometry-passes-from-the-same-render)
  - [Step 2 — frozen diffusion stylization](#step-2--frozen-diffusion-stylization)
  - [Step 3 — label-safe compositing (fg-keep)](#step-3--label-safe-compositing-fg-keep)
  - [Step 4 — split, train, evaluate](#step-4--split-train-evaluate)
  - [Published stylized datasets](#published-stylized-datasets)
  - [Results](#results)
  - [Reproducing this stage](#reproducing-this-stage)
- [Distractor and loose-tube extension (inherited from `blenderproc`)](#distractor-and-loose-tube-extension-inherited-from-blenderproc)
- [Repository and published assets](#repository-and-published-assets)
- [Classes, geometry, and assembly](#classes-geometry-and-assembly)
- [Dataset and split integrity](#dataset-and-split-integrity)
- [Implemented mathematical methods](#implemented-mathematical-methods)
- [Training and evaluation protocol](#training-and-evaluation-protocol)
- [Experimental results](#experimental-results)
- [Qualitative results](#qualitative-results)
- [Reproducibility](#reproducibility)
- [Limitations and next experiments](#limitations-and-next-experiments)
- [Provenance and licensing](#provenance-and-licensing)

## ControlNet realism stage (this branch)

### Why diffusion stylization

The real-camera evaluation on `blenderproc` showed that the remaining Sim2Real gap is dominated by **appearance, not geometry**: the distractor model already detects loose tubes and ignores look-alike objects, yet still drops from 87.92% (its own synthetic test split) to 48.91% real mAP50–95. Cycles renders are geometrically correct but visually recognisable as renders — clean procedural materials, perfect edges, synthetic lighting.

This branch attacks the appearance gap directly. Each synthetic frame is repainted by a **frozen diffusion model** into a photographic style, while the labelled-object geometry is held fixed, so the exact same YOLOv8-OBB labels remain valid. The idea follows the img2img-stylization approach used in YOLO-sd-style Sim2Real pipelines, with one difference that matters for labels: conditioning and compositing are driven by **exact renderer geometry** (depth + per-object index), not by an estimated edge map.

### Pipeline at a glance

```text
blender/generate_synthetic_dataset.py --export-passes        (Cycles, seed 42, 1000 frames)
  ├─ images/train/*.png   labels/train/*.txt   data.yaml     <- unchanged baseline outputs
  ├─ passes/depth/*.exr   passes/normal/*.exr  passes/index/*.exr
  └─ scene/*.json                                            <- index -> object/class mapping
        │
        ▼
scripts/controlnet_passes.py     QA: metric depth, integer ids, labelled masks inside GT OBBs
scripts/controlnet_setup.py      pinned HF downloads + VRAM probe (5.5 GiB budget on 6 GiB GPU)
        │
        ▼
scripts/controlnet_stylize.py    SD1.5 Realistic Vision + depth & geometry-edge ControlNets
        │                        img2img (strength 0.45) -> index-mask composite (fg-keep)
        ▼
<config>/images/*.png + unchanged labels -> yolo_dataset_splitter.py (seed 42)
        │
        ▼
train_yolo.py (fresh yolov8n-obb.pt, 30 ep) -> evaluate_yolo.py (stylized holdout)
                                           -> eval_real_test.py  (282 real frames)
```

### Step 1 — geometry passes from the same render

The generator's `--export-passes` flag (Cycles only) enables the Z-depth, world-normal, and object-index view-layer passes and routes them through a compositor File Output node into `passes/{depth,normal,index}/<frame>.exr`. Every scene object is assigned a `pass_index`; the per-frame `scene/<frame>.json` (written by `--scene-json`) maps each index to its object, class, and `labelled` flag. The Combined image still goes to the Composite node, so **the saved RGB and labels are bit-identical to a run without pass export**.

[`scripts/controlnet_passes.py`](scripts/controlnet_passes.py) turns the passes into conditioning inputs and QA's them against the labels: depth must be finite and metric, index values must be integers that all resolve in the scene JSON, and every labelled object's index-mask pixels must lie inside its ground-truth OBB. The published QA contact sheet is [`results/controlnet/smoke_passes_grid.jpg`](results/controlnet/smoke_passes_grid.jpg) ([report](results/controlnet/smoke_passes_grid.json)).

### Step 2 — frozen diffusion stylization

[`scripts/controlnet_stylize.py`](scripts/controlnet_stylize.py) runs a Multi-ControlNet img2img pipeline. Two conditioning maps are derived **from the EXR passes, not from the rendered image**:

- **Depth ControlNet** input: metric Z converted to percentile-normalised inverse depth (near = white, background = black) — the MiDaS-style encoding `control_v11f1p_sd15_depth` was trained on.
- **Edge ControlNet** input: the boundary mask of the object-index map — white silhouette edges on black, dilated 3 px. This feeds the *canny* ControlNet a **geometry-only edge map**: true object silhouettes with no shading, texture, or reflection edges, which is why the stylized frame keeps the render's exact object layout.

Frozen configuration (recorded per run in `results/controlnet/s045_*_manifest.json`):

| Component | Pinned value |
|---|---|
| Base model | `SG161222/Realistic_Vision_V5.1_noVAE` @ `1e9f017a7b` |
| VAE | `stabilityai/sd-vae-ft-mse` @ `31f26fdeee` |
| ControlNets | `lllyasviel/control_v11f1p_sd15_depth` @ `539f99181d`, `lllyasviel/control_v11p_sd15_canny` @ `115a470d54` |
| Scheduler | DPM++ 2M midpoint, Karras sigmas |
| Steps / strength | 25 / **0.45** (≈ 11 effective denoising steps) |
| Guidance / ControlNet scales | 6.0 / depth 1.0, edge 0.6 (edge guidance ends at 0.8) |
| Precision / attention | fp16, PyTorch SDPA, no CPU offload |
| Per-frame seed | `42 × 1,000,003 + frame_index` |

[`scripts/controlnet_setup.py`](scripts/controlnet_setup.py) downloads the four pinned snapshot revisions (`--download`) and probes peak VRAM of the exact pipeline (`--probe`). Measured on the RTX 4050 Laptop 6 GB: **3.38 GiB** of weights, **4.74 GiB** peak reserved (fits the 5.5 GiB budget), **~4.4–5.9 s/frame** → roughly **1.5 hours per 1,000 frames** at 640×640. Report: [`smoke_controlnet/vram_probe.json`](smoke_controlnet/vram_probe.json).

### Step 3 — label-safe compositing (fg-keep)

Diffusion adds photographic texture but can also move edges slightly. The OBB labels were computed on the *rendered* pixels, so each stylized frame is composited back against the original render inside the labelled-object mask:

$$
\mathrm{out} = (\alpha \cdot k)\,\mathrm{rgb} + (1 - \alpha \cdot k)\,\mathrm{stylized},
$$

where $\alpha$ is the feathered union of all **labelled** objects' index-mask pixels (dilate 3 px, Gaussian feather 2 px) and $k$ is `fg-keep`:

- **`fg-keep 1.0`** (`dataset_controlnet/`): original pixels inside every labelled mask — a *hard* label guarantee; only background, distractors, and shading are diffused.
- **`fg-keep 0.5`** (`dataset_controlnet_fg05/`): a 50/50 blend inside labelled masks — labelled objects get half the diffusion texture while geometry stays render-true; a softer guarantee that buys noticeably more realism.

Because the mask comes from the object-index pass, it matches the visible-silhouette label convention exactly — including transparent objects and occlusion boundaries.

### Step 4 — split, train, evaluate

The stylized images plus the **unchanged** source labels form a standard YOLO dataset. The splitter was run with seed 42, so `dataset_controlnet/` and `dataset_controlnet_fg05/` have **the same train/val/test stem membership** as `dataset_blenderproc/` (the underlying scenes differ — the source was re-rendered with the current generator, i.e. Falcon tubes and ~50% hole fill). Both models then trained with the *unchanged* recipe: fresh `yolov8n-obb.pt`, 800 images, 30 epochs, 640/batch 16, seed 42, deterministic, validation-selected `best.pt`.

Runs: `controlnet_s045_800_30ep` → `weights/controlnet/`, `controlnet_s045_fg05_800_30ep` → `weights/controlnet_fg05/`. Selected checkpoint SHA256:

```text
weights/controlnet/best.pt      75ab3144d60b2fad8e1caf7c58d09a9c537eddfc9cd0e624383f12cd042ccaff
weights/controlnet_fg05/best.pt 6f2b6ef144fe712daf0b0b2b0a39e4c13b59f2b7d40aa3a390d9d9f2ac331b39
```

### Published stylized datasets

| Directory | Stylization | Size | Train | Val | Test | Labelled instances |
|---|---|---:|---:|---:|---:|---:|
| [`dataset_controlnet/`](dataset_controlnet/) | strength 0.45, fg-keep 1.0 | ~525 MiB | 800 | 100 | 100 | 6,362 |
| [`dataset_controlnet_fg05/`](dataset_controlnet_fg05/) | strength 0.45, fg-keep 0.5 | ~523 MiB | 800 | 100 | 100 | 6,362 |

Labels are identical between the two datasets (copied from the same source render); only the pixels differ. Per-frame diffusion seeds and every pipeline parameter are frozen in [`s045_fg100_manifest.json`](results/controlnet/s045_fg100_manifest.json) and [`s045_fg050_manifest.json`](results/controlnet/s045_fg050_manifest.json). The ~5 GiB source render (`dataset_controlnet_src/`, images + EXR passes + scene JSON) is **not** published — regenerate it with `--export-passes` (commands below).

### Results

All four models were evaluated on the **same real test set** ([`dataset/real_test/`](dataset/real_test/), 282 real UR16e frames, 1,592 instances; imgsz 640, conf 0.001, NMS IoU 0.7 — see [Real-camera Sim2Real evaluation](#real-camera-sim2real-evaluation-datasetreal_test)):

| Model | Real P | Real R | Real mAP50 | Real mAP50–95 | Metrics |
|---|---:|---:|---:|---:|---|
| Baseline `weights/best.pt` | 54.02 | 46.44 | 49.38 | 37.28 | [json](results/baseline/real_world_eval/metrics.json) |
| Distractor `weights/blenderproc/best.pt` | 65.67 | 61.42 | 66.30 | 48.91 | [json](results/blenderproc/real_world_eval/metrics.json) |
| ControlNet fg-keep 1.0 | 62.96 | 62.70 | 63.43 | 48.04 | [json](results/controlnet/real_world_eval/metrics.json) |
| **ControlNet fg-keep 0.5** | **73.46** | **65.64** | **73.93** | **56.56** | [json](results/controlnet/real_world_eval_fg05/metrics.json) |

Per-class real mAP50–95 for the two ControlNet models:

| Class | fg-keep 1.0 | fg-keep 0.5 |
|---|---:|---:|
| Tube_Holder | 78.51 | **81.72** |
| Glass_Tube | 13.35 | **38.39** |
| Slide_Holder | 44.73 | **45.41** |
| Eppendorf_Rack | 55.55 | **60.71** |

On their **own stylized test splits** (same-domain holdouts, 100 images each): fg-keep 1.0 model **88.54%** mAP50–95, fg-keep 0.5 model **87.28%** ([fg1.0 metrics](results/controlnet/stylized_test_eval/metrics.json), [fg0.5 metrics](results/controlnet/stylized_test_eval_fg05/metrics.json)) — in line with the ~88% of the earlier synthetic holdouts.

Findings:

1. **Full stylization alone did not beat the distractor model.** fg-keep 1.0 scores 48.04% real mAP50–95 (vs 48.91%): photorealistic backgrounds help `Tube_Holder` (+3.5 pts) but the untouched-render foreground keeps `Glass_Tube` at 13.35% — the domain gap on the *object itself* is what was hurting.
2. **Blending diffusion into the foreground is the win.** fg-keep 0.5 lifts `Glass_Tube` to 38.39% (+25 pts over fg-keep 1.0, +9.1 over the distractor model) and every other class too, for +7.65 pts overall. Half-rendered texture on the object is apparently close enough to the Falcon-tube appearance that the detector finally transfers — while still differing enough from both domains to avoid over-fitting either.
3. **`Slide_Holder` recall (0.42) and `Glass_Tube` precision (0.48) remain the weakest points**; residual causes include the 15 ml vs 50 ml real/synthetic tube size mismatch and unannotated real clutter counted as false positives.

Per-frame prediction overlays for all 282 real frames are published under [`results/controlnet/real_world_eval*/predictions/`](results/controlnet/).

### Reproducing this stage

```powershell
# 0. install the diffusion stack on top of the base environment
python -m pip install -r requirements.txt -r requirements-controlnet.txt
python scripts/controlnet_setup.py --download
python scripts/controlnet_setup.py --probe --image dataset/images/train/000000.png

# 1. render the source dataset with passes into a NEW directory
#    (previous renders ran ~3.8 s/frame; pass EXR writes add a little more)
& "C:\Program Files\Blender Foundation\Blender 4.3\blender.exe" -b -P "blender/generate_synthetic_dataset.py" -- --num-images 1000 --samples 64 --seed 42 --engine cycles --scene-json --export-passes --out-dir "dataset_controlnet_src_repro"

# 2. QA the passes against the labels (contact sheet + JSON report)
python scripts/controlnet_passes.py --dataset dataset_controlnet_src_repro --out results/controlnet_repro/smoke_passes_grid.jpg

# 3. stylize all 1,000 frames (~1.5 h per config); writes <out>/s045_fg050/images/*.png + manifest
python scripts/controlnet_stylize.py --dataset dataset_controlnet_src_repro --strength 0.45 --fg-keep 0.5 --out dataset_controlnet_stylized
# for the fg-keep 1.0 variant: --fg-keep 1.0 ; for a strength grid + comparison sheet:
#   --grid 0.35 0.45 0.55 --sheet results/controlnet_repro/stylize_grid.jpg

# 4. promote the config dir to a dataset: add the UNCHANGED labels + data.yaml from the
#    source render next to the stylized images, then apply the frozen seed-42 split
python scripts/yolo_dataset_splitter.py --dataset <assembled_dir> --seed 42 --apply
python scripts/yolo_dataset_splitter.py --dataset <assembled_dir> --seed 42 --check

# 5. train + evaluate exactly like every other stage
python -u scripts/train_yolo.py --dataset <assembled_dir> --weights yolov8n-obb.pt --epochs 30 --imgsz 640 --batch 16 --device 0 --name controlnet_repro
python scripts/eval_real_test.py --weights runs/obb/UR16e_Labware_Detection/controlnet_repro/weights/best.pt --output results/controlnet_repro/real_world_eval
```

To skip rendering and reuse the published stylized datasets, go straight to step 4 with `--dataset dataset_controlnet` or `dataset_controlnet_fg05` (use `--check` only; their splits are already frozen).

## Distractor and loose-tube extension (inherited from `blenderproc`)

This section documents the `blenderproc`-branch stage this branch builds on. It implements two items of reviewer feedback on the baseline: (1) **hindrance / distractor objects** that look like labware but must *not* be detected, and (2) **loose `Glass_Tube` instances outside the holder**, lying on the bench as well as standing, so the detector does not learn that tubes only ever appear upright inside a rack. The `blenderproc` branch is named for the planned BlenderProc migration; the implemented generator is still the headless `bpy` pipeline in [`blender/generate_synthetic_dataset.py`](blender/generate_synthetic_dataset.py), extended as described here. The BlenderProc port itself remains future work ([`blenderproc/README.md`](blenderproc/README.md)). Everything in the later sections of this README (camera, projection, OBB, lighting, split, training protocol) applies unchanged to the extension; only the scene-composition additions are documented in this section.

### Added assets

| File | Role | Label | Imported bounding box (mm) |
|---|---|---|---|
| `stl_models/distractors/eppendorf_big.stl` | Distractor (hindrance) | **none** | 21.0 × 21.0 × 133.0 |
| `stl_models/distractors/eppendorf0p5ml.stl` | Distractor (hindrance) | **none** | 10.0 × 17.6 × 31.3 |
| `stl_models/Tube.stl` (existing) | Additional **loose** instances on the bench | `1: Glass_Tube` | 116.4 long, 30 mm body |

*Note:* after `dataset_blenderproc/` was frozen, `Tube.stl` was replaced by the procedural `Falcon_Tube.stl` (see [Generator fixes](#generator-fixes-for-the-real-camera-failure-modes)); current renders — including the ControlNet source dataset — use the Falcon mesh for both holder and loose tubes.

Both distractor meshes are imported with the same millimetre-to-metre scale, Y-up fix, welding, and ground re-centring as the target meshes, and their longest bounding-box axis is aligned with $z$. The class list is unchanged: `0: Tube_Holder`, `1: Glass_Tube`, `2: Slide_Holder`, `3: Eppendorf_Rack`.

### Label policy: distractors are pure negatives

Distractor objects carry no `class_id` property and are never appended to the generator's `S.labelled` list; `compute_labels()` iterates only over that list, and `build_scene()` raises if a distractor is ever found in it. Consequently no OBB line is written for a distractor, in any pose, even when it is in front of a target. The loose tubes are linked duplicates of the tube mesh with `class_id = 1` and **are** in `S.labelled`, so they receive the same visible-silhouette minimum-area OBB as the tubes inside the holder.

For the visibility ray-cast used by `--label-mode visible`, distractors are flagged see-through (`material_kind = "glass"`) exactly like the glass targets: a target standing behind a translucent distractor is boxed on its own visible silhouette rather than being cut by the distractor's outline.

With `--scene-json` the generator additionally writes `scene/<frame>.json` per frame listing every rendered mesh with its `class_id` (`null` for distractors), pose, location, and whether it produced a label line. These files are published in [`dataset_blenderproc/scene/`](dataset_blenderproc/scene/) and are the machine-checkable evidence for the policy: over the 1,000 frames, **3,993 distractor instances** were rendered and **0** were labelled; **1,993 loose tubes** (994 upright, 999 lying) were rendered and labelled whenever their visible fraction passed the existing filters.

### Resting poses on the bench

Let $V=\{v_i\}$ be the 3-D convex-hull vertices of a mesh (standing on $z=0$, centred on the $z$ axis). A per-instance rotation is drawn as

$$
R_{\text{upright}}=R_z(\psi),\qquad
R_{\text{lying}}=R_z(\psi)\,R_x\!\left(\tfrac{\pi}{2}\right)R_z(\varphi),\qquad
\psi,\varphi\sim\mathcal U[0,2\pi),
$$

so a lying object has its long axis horizontal, a uniformly random yaw $\psi$ about the vertical, and a uniformly random spin $\varphi$ about its own axis. Loose tubes lie with probability **0.5**, distractors with probability **0.7**. The object is then rested on the bench analytically rather than by physics simulation:

$$
z_0=-\min_i\,(Rv_i)_z,\qquad
c=\tfrac12\bigl(\min_i (Rv_i)_{xy}+\max_i (Rv_i)_{xy}\bigr),\qquad
r=\max_i\bigl\|(Rv_i)_{xy}-c\bigr\|.
$$

The instance is placed at $(x-c_x,\;y-c_y,\;z_0)$, so its lowest hull vertex touches $z=0$ and the centre of its horizontal footprint lands on the sampled bench point $(x,y)$; $r$ is the radius of the circle enclosing that footprint. A conical tube lying on its side therefore rests on its cap rim and tip, as in the previews below.

### Collision-free placement

Every object on the bench is represented by its footprint circle $(x,y,r)$ in a per-frame list. A candidate position is accepted only if

$$
\sqrt{(x-x_j)^2+(y-y_j)^2}>r+r_j+\varepsilon\quad\text{for all placed }j,
$$

with $\varepsilon = 3\,\text{mm}$ for loose tubes and distractors (the inherited loose rack/stand placement uses 10 mm). The holder contributes its own footprint circle at the origin; tubes inside the holder are inside that circle and are not added separately. Because enclosing circles are conservative, meshes never interpenetrate. Up to 60 candidates are drawn per object; an object whose candidates all fail is hidden for that frame (this happened **0** times in the published 1,000 frames).

Placement order per frame is: holder → loose rack and stand → loose tubes → camera → distractors.

- **Loose tubes:** $n\sim\mathcal U\{1,2,3\}$ instances, each at polar angle $\theta\sim\mathcal U[0,2\pi)$ and distance $d\sim\mathcal U\,[\,r_h+r+0.005,\; r_h+r+0.08\,]$ metres from the holder centre, where $r_h$ is the holder footprint radius. They are "next to the holder", not scattered across the bench.
- **Distractors:** $n\sim\mathcal U\{2,\dots,6\}$ instances in total, each drawing one of the two meshes uniformly. Each instance is placed in one of two modes:
  - *Hindering* (probability **0.35**): a target $t$ currently visible in the scene (holder, loose rack/stand, or any tube) is chosen uniformly; with $\hat d$ the unit horizontal vector from the target centre toward the camera position, $\hat d_\perp$ its perpendicular, the distractor is placed at
    $$
    p = t + \hat d\,(r_t + r + g) + \hat d_\perp\,\lambda\, r_t,\qquad g\sim\mathcal U[2\,\text{mm},30\,\text{mm}],\ \ \lambda\sim\mathcal U[-0.5,0.5],
    $$
    i.e. just in front of the target as seen from the camera, so it partially occludes it. If that position violates the collision test the instance falls back to random mode.
  - *Random* (otherwise): $\theta\sim\mathcal U[0,2\pi)$, $d\sim\mathcal U\,[\,r_h+r+0.005,\; r_h+r+0.20\,]$.

The camera is sampled before the distractors so the hindering direction is known; the camera model itself is unchanged.

### Distractor material

Distractors use a Principled BSDF with transmission weight 1.0 and IOR 1.49 (polypropylene), re-randomised per frame per mesh:

| Case | Probability | Roughness | Base colour |
|---|---:|---|---|
| Clear, glass-like | 0.3 | $\mathcal U[0,0.1]$ | per-channel $\mathcal U[0.9,1.0]$ |
| Frosted / translucent polypropylene | 0.7 | $\mathcal U[0.25,0.6]$ | HSV with $h\sim\mathcal U[0,1)$, $s\sim\mathcal U[0.05,0.3]$, $v\sim\mathcal U[0.85,1.0]$ |

Target materials, bench, world, lighting and exposure randomisation are those of the baseline.

### Dataset `dataset_blenderproc/`

Generated with seed 42, 64 Cycles samples, 640 × 640, visible-silhouette labels, in **3,787 s (3.79 s/frame)** on the RTX 4050 (OptiX). The exposure guard re-rendered 204 of the 1,000 frames. The directory was split with the same `scripts/yolo_dataset_splitter.py` (seed 42) and passes the same integrity checks (1,000 pairs, no shared stems, no duplicate decoded pixels, all classes in every split); memberships and hashes are frozen in [`dataset_blenderproc/split_manifest.json`](dataset_blenderproc/split_manifest.json). The published directory is approximately **484 MiB**.

| Split | Images | Tube_Holder | Glass_Tube | Slide_Holder | Eppendorf_Rack | Total labelled instances | Distractor instances (unlabelled) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 800 | 800 | 3,962 | 667 | 667 | **6,096** | — |
| Validation | 100 | 100 | 498 | 79 | 82 | **759** | — |
| Test | 100 | 100 | 497 | 88 | 79 | **764** | — |
| **Total** | **1,000** | **1,000** | **4,957** | **834** | **828** | **7,619** | **3,993** |

`Glass_Tube` instances rise from 2,985 in the baseline to 4,957 because of the loose tubes. Distractor counts per split are not tracked by the splitter (they have no labels); the per-frame totals are in the scene JSON files.

### Smoke test (8 frames, ground-truth overlays)

Before the full render an 8-frame smoke test (`--num-images 8 --samples 32 --seed 7 --scene-json`) was rendered and its labels drawn with `scripts/visualize_obb_labels.py`. The overlays and scene JSONs are published in [`results/blenderproc/smoke/`](results/blenderproc/smoke/). Boxes appear on the holder, the tubes in the holder, the loose tubes (upright and lying), the slide stand and the rack; none appear on the Eppendorf distractors, including the frosted one in the foreground of frame 7.

| Smoke frame 000002 (three lying loose tubes, two small distractors) | Smoke frame 000007 (foreground distractor overlapping the holder, unboxed) |
|---|---|
| ![Smoke 000002](results/blenderproc/smoke/previews/000002.png) | ![Smoke 000007](results/blenderproc/smoke/previews/000007.png) |

### Training

Run **`stl_synthetic_distractors_800_30ep`**: identical recipe to the baseline (original DOTA-pretrained `yolov8n-obb.pt`, 800 training images, 30 epochs, 640 / batch 16, seed 42, deterministic, `patience=0`, device 0, workers 0), completed in 0.25 h. Checkpoint selection used the distractor validation split only. The selected `weights/blenderproc/best.pt` SHA256 is

```text
1de4a36fe87b550b24d5ef71cda83fb8dc18cfe653607d880853f6f3f50d066f
```

![Distractor-model training and validation curves across 30 epochs](results/blenderproc/training/results.png)

### Results compared to the baseline

All four combinations of {baseline model, distractor model} × {baseline test split, distractor test split} were evaluated with the same frozen-checkpoint evaluator (batch 16, 640 px, NMS IoU 0.7, confidence cutoff 0.001). Values are percentages; exact values are in the linked `metrics.json` files.

| Model | Evaluation split | Images | Instances | Precision | Recall | mAP50 | mAP50–95 | Metrics |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Distractor model | Distractor **validation** | 100 | 759 | 98.97 | 94.39 | 97.44 | **88.18** | [json](results/blenderproc/validation/metrics.json) |
| Distractor model | Distractor **test** | 100 | 764 | 98.39 | 94.95 | 97.58 | **87.92** | [json](results/blenderproc/test/metrics.json) |
| Distractor model | Baseline test | 100 | 566 | 98.38 | 96.56 | 97.49 | **88.27** | [json](results/blenderproc/cross_evaluation/new_model_on_baseline_test/metrics.json) |
| Baseline model | Baseline test | 100 | 566 | 97.04 | 95.76 | 97.16 | **88.29** | [json](results/baseline/test/metrics.json) |
| Baseline model | Distractor test | 100 | 764 | 86.06 | 82.85 | 88.82 | **78.57** | [json](results/blenderproc/cross_evaluation/baseline_model_on_distractors_test/metrics.json) |
| Baseline model | Distractor validation | 100 | 759 | 85.61 | 83.32 | 88.79 | **79.69** | [json](results/blenderproc/cross_evaluation/baseline_model_on_distractors_val/metrics.json) |

Per-class AP50–95 on the **distractor test split**, and on the **baseline test split** for the new model:

| Class | Distractor model, distractor test | Baseline model, distractor test | Distractor model, baseline test | Baseline model, baseline test |
|---|---:|---:|---:|---:|
| Tube_Holder | 95.75 | 96.65 | 95.51 | 96.23 |
| Glass_Tube | **86.56** | **62.15** | 86.34 | 85.86 |
| Slide_Holder | **85.07** | **71.16** | 87.56 | 87.05 |
| Eppendorf_Rack | 84.30 | 84.31 | 83.67 | 84.00 |

Observations:

1. **The baseline model does not generalise to the new scene composition.** On the distractor test split its `Glass_Tube` recall falls to 55.4% (it misses most lying/loose tubes, which it never saw) and its `Slide_Holder` precision falls to 52.6%: in the annotated batches ([example](results/blenderproc/cross_evaluation/baseline_model_on_distractors_test/val_batch0_pred.jpg)) it labels lying tubes and the large Eppendorf distractor as `Slide_Holder` with confidence ≈ 0.9. Its holder and rack scores are unchanged.
2. **The distractor model does not regress on the clean domain.** 88.27% versus 88.29% mAP50–95 on the baseline test split, with all per-class scores within ±0.5 points; its `Glass_Tube` AP50–95 on that split is slightly higher (86.34 vs 85.86).
3. **Distractors are ignored, loose tubes are detected.** In the annotated distractor-test batches ([example](results/blenderproc/test/val_batch0_pred.jpg)) the large and small Eppendorf tubes receive no boxes while upright and lying loose tubes are boxed as `Glass_Tube`.
4. **The harder split costs about 0.4 points.** 87.92% on the distractor test versus 88.27% on the baseline test for the same model; `Slide_Holder` and `Eppendorf_Rack` remain the weakest classes, as in the baseline. Both splits are 100 images, so differences below roughly one point should not be over-interpreted.

As with the baseline, these are same-renderer synthetic holdouts. The distractor result shows robustness to *rendered* look-alike objects, not to real laboratory clutter.

| Distractor validation PR curve | Distractor test PR curve |
|---|---|
| ![Distractor validation PR curve](results/blenderproc/validation/BoxPR_curve.png) | ![Distractor test PR curve](results/blenderproc/test/BoxPR_curve.png) |

| Ground truth: distractor validation batch 0 | Predictions: distractor validation batch 0 |
|---|---|
| ![Distractor validation batch 0 labels](results/blenderproc/validation/val_batch0_labels.jpg) | ![Distractor validation batch 0 predictions](results/blenderproc/validation/val_batch0_pred.jpg) |

Ten fixed-seed test previews (`random.Random(42).sample(sorted(test_stems), 10)`, confidence 0.25, NMS IoU 0.7), predictions only:

| Test example | Test example |
|---|---|
| ![Test 000030](results/blenderproc/test/previews/000030.png) | ![Test 000099](results/blenderproc/test/previews/000099.png) |
| ![Test 000103](results/blenderproc/test/previews/000103.png) | ![Test 000127](results/blenderproc/test/previews/000127.png) |
| ![Test 000228](results/blenderproc/test/previews/000228.png) | ![Test 000250](results/blenderproc/test/previews/000250.png) |
| ![Test 000296](results/blenderproc/test/previews/000296.png) | ![Test 000825](results/blenderproc/test/previews/000825.png) |
| ![Test 000867](results/blenderproc/test/previews/000867.png) | ![Test 000954](results/blenderproc/test/previews/000954.png) |

The full 100-image prediction record is [`results/blenderproc/test/predictions.json`](results/blenderproc/test/predictions.json); SHA256 values of every published extension artifact are in [`results/blenderproc/checksums.sha256`](results/blenderproc/checksums.sha256).

### Real-camera Sim2Real evaluation (`dataset/real_test/`)

The repository's synthetic holdouts cannot show real-world transfer, so both checkpoints were also evaluated on **real UR16e camera frames**: the [`ABC-iRobotics/LabWare_Items`](https://huggingface.co/datasets/ABC-iRobotics/LabWare_Items) dataset (MIT license), `instance_segmentation` configuration, official **test** split — 282 images at 1280×800 with 1,592 annotated instances of the same four lab items. [`scripts/prep_real_test_set.py`](scripts/prep_real_test_set.py) converts each instance mask to a tight oriented box (`cv2.minAreaRect` over visible pixels — the same visibility convention as the synthetic labels), maps `TubeHolder/Tube/Slide_holder/eppendorf_tarto` to our class ids 0–3, and writes `dataset/real_test/` plus a provenance manifest. [`scripts/eval_real_test.py`](scripts/eval_real_test.py) runs the frozen checkpoints (imgsz 640, conf 0.001, NMS IoU 0.7); overlays for every frame are under `results/*/real_world_eval/predictions/`.

| Model | Real P | Real R | Real mAP50 | Real mAP50–95 | Own synthetic test mAP50–95 |
|---|---:|---:|---:|---:|---:|
| Baseline `weights/best.pt` | 54.02 | 46.44 | 49.38 | 37.28 | 88.29 (clean test) |
| Distractor `weights/blenderproc/best.pt` | 65.67 | 61.42 | 66.30 | 48.91 | 87.92 (distractor test) |
| ControlNet fg-keep 1.0 `weights/controlnet/best.pt` | 62.96 | 62.70 | 63.43 | 48.04 | 88.54 (stylized test) |
| **ControlNet fg-keep 0.5 `weights/controlnet_fg05/best.pt`** | **73.46** | **65.64** | **73.93** | **56.56** | 87.28 (stylized test) |

Full numbers, per-class tables, and the report: [`results/baseline/real_world_eval/sim2real_report.json`](results/baseline/real_world_eval/sim2real_report.json) and [`SIM2REAL_REPORT.md`](results/baseline/real_world_eval/SIM2REAL_REPORT.md); the ControlNet-stage evaluations are under [`results/controlnet/real_world_eval*/`](results/controlnet/). "Own synthetic test" above refers to each model's same-domain holdout — it is not comparable across rows.

The Sim2Real gap is large but structured: `Glass_Tube` accounts for most of it — the real `Tube` items are Falcon-style conical tubes lying loose (the baseline only ever saw thin glass tubes upright inside the holder), and its real `Glass_Tube` mAP50 is 4.8%. The distractor model's loose-tube training lifts this to 43.4% and gains 11.6 mAP50–95 points overall. Remaining error modes: the distractor model hallucinates `Glass_Tube` inside the real rack's empty holes (in synthetic scenes holes always contained tubes), the baseline labels real loose tubes `Slide_Holder`, and unannotated scene clutter counts as false positives. The ControlNet stage on this branch takes these two models' numbers as its reference and improves them to **56.56%** real mAP50–95 — see [Results](#results).

```powershell
# rebuild the real test set (HF cache reused), then evaluate a checkpoint on it
python scripts/prep_real_test_set.py
python scripts/eval_real_test.py --weights weights/best.pt --output results/baseline/real_world_eval_v2
```

### Generator fixes for the real-camera failure modes

The real-camera evaluation exposed two scene-composition gaps, fixed in the generator after `dataset_blenderproc/` was frozen (that dataset and its model were made with the previous generator and `Tube.stl`; re-running the current generator produces different scenes):

1. **Empty rack holes ("ghost tubes").** Each of the four tube holes is now filled independently with probability `HOLE_FILL_PROB = 0.5` (previously 0.75 with at least one tube forced), so about half the holes are empty and some racks are completely empty — the distractor model had learnt that a hole always contains a tube.
2. **Falcon tube geometry.** The real tubes are unskirted 50 ml conical tubes with screw caps; `Tube.stl` has a skirt that hides its conical bottom. [`scripts/generate_falcon_tube.py`](scripts/generate_falcon_tube.py) revolves a radial profile into a watertight mesh, `stl_models/Falcon_Tube.stl` (pointed conical tip over 21 mm, 29 mm body, 34 mm ribbed cap, 114.5 mm overall), which replaces `Tube.stl` as class `1: Glass_Tube`. The cap (faces above the radius step) gets its own opaque plastic material whose colour varies per tube instance (Object Info → Random → constant ColorRamp, palette redrawn per frame); visibility ray-casts treat cap faces as opaque and the clear body as see-through. A pointed tube cannot stand on its tip, so loose tubes lie on their side (80%) or stand inverted on the cap (20%). Because cap and tip have different radii, a lying tube is tilted to its physical rest angle $\varphi^* = \arg\min_\varphi \big(z_{\text{com}}(\varphi) - \min_i z_i(\varphi)\big)$ over the hull vertices $z_i$ and the shell centroid (also applied to lying distractors). The OBB is the minimum-area rectangle of the projected visible surface points, so cap and tip are enclosed automatically.

A 16-frame QA render with ground-truth overlays ([`results/geometry_test_grid.jpg`](results/geometry_test_grid.jpg), `scripts/make_contact_sheet.py`) has 46.9% empty holes, 3 fully empty racks, every loose tube labelled and no distractor labelled.

![Geometry QA contact sheet](results/geometry_test_grid.jpg)

### Reproducing the extension

```powershell
# 8-frame smoke test with ground-truth overlays and per-frame scene JSON
& "C:\Program Files\Blender Foundation\Blender 4.3\blender.exe" -b -P "blender/generate_synthetic_dataset.py" -- --num-images 8 --samples 32 --seed 7 --scene-json --out-dir "dataset_blenderproc_smoke"
python scripts/visualize_obb_labels.py --dataset dataset_blenderproc_smoke --all

# full 1,000-frame render into a NEW directory, then the frozen 800/100/100 split
& "C:\Program Files\Blender Foundation\Blender 4.3\blender.exe" -b -P "blender/generate_synthetic_dataset.py" -- --num-images 1000 --samples 64 --seed 42 --scene-json --out-dir "dataset_blenderproc_reproduced"
python scripts/yolo_dataset_splitter.py --dataset dataset_blenderproc_reproduced --seed 42 --apply
python scripts/yolo_dataset_splitter.py --dataset dataset_blenderproc_reproduced --seed 42 --check

# fresh 30-epoch run (validates best.pt, predicts the test split, saves ten previews)
python -u scripts/train_yolo.py --dataset dataset_blenderproc --weights yolov8n-obb.pt --epochs 30 --imgsz 640 --batch 16 --device 0 --name stl_synthetic_distractors_800_30ep_reproduction

# frozen-checkpoint evaluations, including the cross-evaluation against the baseline
python scripts/evaluate_yolo.py --weights weights/blenderproc/best.pt --dataset dataset_blenderproc --split test --device 0 --output runs/obb/reproduce_distractor_test
python scripts/evaluate_yolo.py --weights weights/blenderproc/best.pt --dataset dataset --split test --device 0 --output runs/obb/reproduce_new_model_on_baseline_test
python scripts/evaluate_yolo.py --weights weights/best.pt --dataset dataset_blenderproc --split test --device 0 --output runs/obb/reproduce_baseline_model_on_distractor_test
```

The generator on this branch always adds distractors and loose tubes; the published `dataset/` baseline was rendered with the `main` version of the script. Verify the published extension dataset with `python scripts/yolo_dataset_splitter.py --dataset dataset_blenderproc --check`.

---

The remainder of this README documents the inherited **baseline** (`main`) pipeline, dataset, and results, which this branch reuses unchanged.

## Repository and published assets

```text
blender/
  generate_synthetic_dataset.py      Headless Blender generator (+ distractors, loose tubes,
                                     --scene-json, --export-passes for ControlNet conditioning)
stl_models/
  TubeHolder.stl
  Falcon_Tube.stl                    Procedural Falcon tube (class 1; replaces Tube.stl on this branch)
  Tube.stl                           Legacy skirted tube (kept for dataset_blenderproc provenance)
  Slide_holder.stl
  eppendorf_tarto.stl
  distractors/                       Unlabelled hindrance meshes (blenderproc stage)
    eppendorf_big.stl
    eppendorf0p5ml.stl
scripts/
  yolo_dataset_splitter.py           Split creation and integrity checks
  train_yolo.py                      Fresh training, validation, test previews
  evaluate_yolo.py                   Frozen-checkpoint validation/test evaluation
  prep_real_test_set.py              Build dataset/real_test from HF LabWare_Items
  eval_real_test.py                  Frozen-checkpoint evaluation on the real test set
  generate_falcon_tube.py            Procedural Falcon_Tube.stl generator
  make_contact_sheet.py              GT-OBB overlay contact sheets for render QA
  visualize_obb_labels.py            Ground-truth label overlays
  prepare_publication.py             Artifact copying and publication checksums
  controlnet_setup.py                Pinned HF model downloads + VRAM probe (this branch)
  controlnet_passes.py               EXR pass loading, conditioning maps, pass-vs-label QA
  controlnet_stylize.py              Multi-ControlNet img2img stylization + fg-keep compositing
weights/
  best.pt  last.pt                   Baseline model (main)
  blenderproc/best.pt  last.pt       Distractor model (blenderproc)
  controlnet/best.pt  last.pt        Stylized fg-keep 1.0 model (this branch)
  controlnet_fg05/best.pt  last.pt   Stylized fg-keep 0.5 model (this branch, best real score)
dataset/
  data.yaml  split_manifest.json
  images/{train,val,test}/  labels/{train,val,test}/     800 / 100 / 100
  real_test/                       282 real UR16e frames + manifest (evaluation only)
dataset_blenderproc/               Distractor + loose-tube dataset, same layout + scene/
dataset_controlnet/                Stylized s=0.45 fg-keep 1.0 dataset (this branch)
dataset_controlnet_fg05/           Stylized s=0.45 fg-keep 0.5 dataset (this branch)
results/baseline/                  Baseline training/validation/test + real_world_eval + SIM2REAL_REPORT
results/blenderproc/               Distractor model results + cross_evaluation + real_world_eval
results/controlnet/                This branch:
  training/  training_fg05/        Curves, held-out val, test previews for both stylized models
  stylized_test_eval{,_fg05}/      Same-domain stylized holdout evaluations
  real_world_eval{,_fg05}/         Real-test metrics + all 282 prediction overlays
  smoke_passes_grid.jpg + .json    Pass-vs-label QA contact sheet
  stylize_grid_comparison.jpg      Strength grid 0.35/0.45/0.55 at fg-keep 1.0 (shown above)
  s045_fg100_manifest.json  s045_fg050_manifest.json   Frozen stylization params + per-frame seeds
smoke_controlnet/vram_probe.json   GPU memory/speed probe of the diffusion pipeline
requirements.txt                   Base ML environment (torch 2.5.1+cu121, ultralytics, ...)
requirements-controlnet.txt        Diffusion pins: diffusers 0.40.0 / transformers 5.16.1 / OpenEXR
.gitattributes
.gitignore
```

The **complete dataset**, not merely a sample, is published in [`dataset/`](dataset/). It is approximately **467 MiB** including labels and metadata; the largest image is below 1 MiB. Each trained checkpoint is approximately **6.2 MiB**. The publication fits GitHub's normal Git file and push limits, so no Git LFS pointers or separate Release download are required.

Historical v1 datasets (`YOLO_Main_Dataset/`, `Blender Dataset/`), `.blend` scene files, and the 4,000-image training/test runs (`ur16e_main_run_4000-3`, `predict-12`) have been moved to the dedicated `legacy-v1` branch so that `main` contains only the clean v2 STL baseline.

Only the current baseline's dataset, weights, STL files, and results have explicit `.gitignore` publication exceptions. Other local render runs, caches, and generated previews remain ignored. `.gitattributes` prevents newline conversion in frozen dataset/result artifacts so their byte-level checksums survive cross-platform Git checkouts.

## Classes, geometry, and assembly

| ID | Class | Mesh file (this branch) | Scene role | Assigned material |
|---:|---|---|---|---|
| 0 | `Tube_Holder` | `TubeHolder.stl` | Main rack | Matte/plastic |
| 1 | `Glass_Tube` | `Falcon_Tube.stl` | Occupied holder holes + loose instances | Principled glass (opaque cap) |
| 2 | `Slide_Holder` | `Slide_holder.stl` | Loose object on the bench | Principled glass |
| 3 | `Eppendorf_Rack` | `eppendorf_tarto.stl` | Loose object on the bench | Principled glass |

The class names are defined by `OBJECT_SPECS` in the [generator](blender/generate_synthetic_dataset.py). STL coordinates are interpreted as millimetres and scaled by **0.001** into Blender metres. Import applies the configured Y-up to Z-up transform, welds nearby vertices, recalculates normals, and recentres the meshes on the ground plane. The frozen `dataset/` and `dataset_blenderproc/` were rendered with the legacy `Tube.stl` (skirted, ~116.4 mm); on this branch class 1 is the procedural **`Falcon_Tube.stl`** — an unskirted 50 ml conical tube (29 mm body, pointed conical tip over 21 mm, 34 mm ribbed screw cap, 114.5 mm overall) generated by [`scripts/generate_falcon_tube.py`](scripts/generate_falcon_tube.py) to match the real centrifuge tubes. Its cap is opaque plastic with a per-instance random colour; loose instances lie on their side (80%) or stand inverted on the cap (20%). See [Generator fixes](#generator-fixes-for-the-real-camera-failure-modes).

The generator orients the holder so the face with the most fitting holes points up. For these meshes, the winning orientation is `-Y` in the orientation search, exposing **four holes** with approximately **14.7 mm radius**. The tube body has approximately **14.5 mm radius** and its tip rests on a hole floor about **9 mm** above the bench.

Hole detection uses a 1 mm downward-ray grid, a highest sufficiently populated height band as the rack deck, four-connected depressed regions, and 24-direction horizontal radius probes. The fitting test is

$$
r_{\mathrm{insert}}(d)=\max_{v_i:z_i\leq d+10^{-6}}\sqrt{x_i^2+y_i^2},
\qquad
r_{\mathrm{insert}}(d)\leq r_{\mathrm{hole}}+0.001.
$$

Here $d$ is insertion depth in metres. This tests the inserted body rather than rejecting a tube because its cap is wider than the hole. In the frozen baseline datasets, each fitting hole was occupied independently with probability **0.75**, with a fallback ensuring at least one tube. **On this branch the fill probability is `HOLE_FILL_PROB = 0.5` with no forced minimum** (see [Generator fixes](#generator-fixes-for-the-real-camera-failure-modes)) — the ControlNet source dataset contains empty holes and fully empty racks. Each loose object is included with probability **0.85**. Holder, tube, and loose-object yaw angles are randomized.

**Material realism caveat:** the two loose rack/stand meshes are deliberately rendered as glass in this baseline. Their assigned material is not evidence that the corresponding physical parts are transparent. Correct this in a future dataset if the real components are opaque plastic.

## Dataset and split integrity

Each image is a 640 × 640 RGB PNG. Each label is a text file with the same stem. The generator produces an unsplit `images/train` and `labels/train` tree; the splitter then assigns complete image/label pairs to disjoint partitions.

| Split | Images | Label files | Tube_Holder instances | Glass_Tube instances | Slide_Holder instances | Eppendorf_Rack instances | Total instances |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 800 | 800 | 800 | 2,378 | 668 | 672 | **4,518** |
| Validation | 100 | 100 | 100 | 305 | 78 | 83 | **566** |
| Test | 100 | 100 | 100 | 302 | 86 | 78 | **566** |
| **Total** | **1,000** | **1,000** | **1,000** | **2,985** | **832** | **833** | **5,650** |

The split is constructed by sorting all stems, shuffling once with `random.Random(42)`, and taking indices `[0:800]`, `[800:900]`, and `[900:1000]`. It is not re-sampled when a saved manifest exists.

The integrity checks enforce:

- Exactly 1,000 matching image/label pairs and the expected four-class mapping.
- No stem shared across splits and no duplicated decoded image content.
- Readable 640 × 640 RGB images; nine fields per label row; finite normalized coordinates and nonzero polygon area.
- All four classes represented in each split.
- File contents matching the saved image, decoded-pixel, and label SHA256 values.
- Separate training, validation, and test paths in the YAML.

[`dataset/split_manifest.json`](dataset/split_manifest.json) records the memberships and hashes. [`dataset/data.yaml`](dataset/data.yaml) deliberately omits an absolute `path`: Ultralytics resolves the dataset root from the YAML file's directory when the scripts pass its absolute filename.

A frame's rendering random-number generator is initialized with

$$
s_k=1{,}000{,}003\,s+k,\qquad s=42,
$$

where $k$ is its zero-based image index. Visibility mesh-vertex subsampling uses NumPy's generator with seed $s$. These seeds make the sampling reproducible under the same code and environment; they do not guarantee bit-identical GPU renders or training outcomes across different Blender, driver, library, or hardware versions. The published files and checksums are the exact baseline artifacts.

## Implemented mathematical methods

The formulas below describe the actual implementation, including its approximations. Blender evaluates the underlying shader and ray-tracing algorithms; the Python script configures them rather than implementing a new optical renderer.

### 1. Camera hemisphere sampling and look-at rotation

Let $\mathbf{t}$ be the randomized look-at target, $\rho$ the camera-target distance, $\theta$ azimuth, and $\phi$ elevation above the XY plane:

$$
\begin{aligned}
t_x,t_y&\sim\mathcal U(-0.04,0.04), & t_z&\sim\mathcal U(0,0.06),\\
\rho&\sim\mathcal U(0.50,0.95), & \theta&\sim\mathcal U(0,2\pi),\\
\phi&\sim\mathcal U(12^\circ,80^\circ).
\end{aligned}
$$

Angles are converted to radians before trigonometric evaluation. The camera position is

$$
\mathbf{c}=\mathbf{t}+\rho
\begin{bmatrix}
\cos\phi\cos\theta\\
\cos\phi\sin\theta\\
\sin\phi
\end{bmatrix}.
$$

This is **uniform in elevation angle**, not uniform over hemisphere surface area. It samples a restricted dome, not the full hemisphere. Uniform-area sampling would require a different elevation distribution and is not used in the reported dataset.

Blender cameras look along local $-Z$ with local $+Y$ as the up axis. Define

$$
\mathbf{f}=\frac{\mathbf{t}-\mathbf{c}}{\|\mathbf{t}-\mathbf{c}\|},\qquad
\mathbf{k}=\begin{bmatrix}0&0&1\end{bmatrix}^{\mathsf T},
$$

$$
\mathbf{r}=\frac{\mathbf{f}\times\mathbf{k}}{\|\mathbf{f}\times\mathbf{k}\|},\qquad
\mathbf{u}=\mathbf{r}\times\mathbf{f},\qquad
R_0=\begin{bmatrix}\mathbf{r}&\mathbf{u}&-\mathbf{f}\end{bmatrix}.
$$

Within the sampled non-polar elevation range, this is the basis produced by `look.to_track_quat("-Z", "Y")`. A local-camera-Z roll $\beta\sim\mathcal U(-8^\circ,8^\circ)$ is then applied:

$$
R=R_0R_z(\beta),\qquad
R_z(\beta)=
\begin{bmatrix}
\cos\beta&-\sin\beta&0\\
\sin\beta&\cos\beta&0\\
0&0&1
\end{bmatrix},\qquad
M_c=\begin{bmatrix}R&\mathbf{c}\\\mathbf{0}^{\mathsf T}&1\end{bmatrix}.
$$

The code implements this as quaternion multiplication `track_quaternion @ roll_quaternion`, then converts to Euler angles. Focal length is sampled uniformly from **35–55 mm**, with Blender's `AUTO` sensor fit. Camera clipping distances are **0.01–50 m**.

### 2. World-to-camera and perspective projection

For an object-local vertex $\tilde{\mathbf{p}}_o=[x,y,z,1]^{\mathsf T}$ and object transform $M_o$,

$$
\tilde{\mathbf{p}}_w=M_o\tilde{\mathbf{p}}_o,\qquad
\tilde{\mathbf{p}}_c=V\tilde{\mathbf{p}}_w,\qquad
V=\operatorname{normalize}(M_c)^{-1}.
$$

`normalize` here is Blender's matrix normalization, removing camera-axis scale. With the unit-scale camera above, $V=[R^{\mathsf T},-R^{\mathsf T}\mathbf c;\,\mathbf0^{\mathsf T},1]$.

Write camera coordinates as $(X_c,Y_c,Z_c)$ and forward depth as $d=-Z_c$. `Camera.view_frame(scene=...)` supplies reference-plane boundaries $x_0,x_1,y_0,y_1$ at depth $d_0>0$. The vectorized projector uses

$$
\lambda=\frac{d}{d_0},\qquad
u=\frac{X_c-x_0\lambda}{(x_1-x_0)\lambda},\qquad
v=1-\frac{Y_c-y_0\lambda}{(y_1-y_0)\lambda}.
$$

Thus $\nu$ increases rightward and $v$ increases downward. Pixel coordinates are $x_{\mathrm{px}}=W\nu$ and $y_{\mathrm{px}}=Hv$, with $W=H=640$ for this dataset. Points with $d\leq10^{-6}$ are excluded before forming a label hull.

The generator checks the vectorized implementation against `bpy_extras.object_utils.world_to_camera_view` and aborts if the maximum coordinate difference exceeds $10^{-5}$. The recorded baseline scene check reported a maximum difference of **$2.06\times10^{-8}$**; this is a projection consistency check, not a bound on annotation visibility error.

### 3. Visibility, convex hull, and minimum-area OBB

Two label modes are supported:

- **`visible` (published default):** transform the 3D convex-hull vertices plus up to **800 sampled mesh vertices**, then test their visibility with camera-to-point rays. An opaque object blocks the ray; glass surfaces are stepped through, up to eight intersections. A hit on the target object itself counts as visible.
- **`amodal`:** project the object's 3D convex-hull vertices without opaque-occlusion filtering.

The default is a finite geometric visibility approximation. The additional samples are mesh vertices, not an area-uniform surface sample. The visibility rays do not model refractive bending, attenuation, or perceptual contrast.

For the retained projected points $P=\{\mathbf p_i\}$, compute their 2D convex hull $H=\operatorname{conv}(P)$ using Blender's `convex_hull_2d`. Clip that polygon to $[0,1]^2$ using Sutherland–Hodgman clipping. For an edge from $\mathbf p$ to $\mathbf q$ crossing boundary coordinate $b$ on axis $a$, the intersection is

$$
t=\frac{b-p_a}{q_a-p_a},\qquad
\mathbf p_{\mathrm{int}}=\mathbf p+t(\mathbf q-\mathbf p).
$$

Polygon area is computed with the shoelace formula:

$$
A(H)=\frac12\left|\sum_i x_i y_{i+1}-x_{i+1}y_i\right|.
$$

A label is discarded when fewer than three points remain, or when

$$
\frac{A(H\cap[0,1]^2)}{A(H)}<0.25.
$$

This ratio measures **in-frame hull area after the visibility step**, not the percentage of the original physical object visible through occlusion.

For every nondegenerate edge $\mathbf e_j=\mathbf h_{j+1}-\mathbf h_j$ of the clipped hull, form an orthonormal basis:

$$
\mathbf a_j=\frac{\mathbf e_j}{\|\mathbf e_j\|},\qquad
\mathbf b_j=\begin{bmatrix}-a_{j,y}&a_{j,x}\end{bmatrix}^{\mathsf T}.
$$

Project all hull vertices onto these axes and take their extrema:

$$
\begin{aligned}
a_j^-&=\min_i\mathbf h_i\cdot\mathbf a_j,& a_j^+&=\max_i\mathbf h_i\cdot\mathbf a_j,\\
b_j^-&=\min_i\mathbf h_i\cdot\mathbf b_j,& b_j^+&=\max_i\mathbf h_i\cdot\mathbf b_j,\\
A_j&=(a_j^+-a_j^-)(b_j^+-b_j^-),& j^*&=\arg\min_j A_j.
\end{aligned}
$$

The four corners are

$$
\begin{aligned}
\mathbf q_1&=a^-\mathbf a+b^-\mathbf b,&\mathbf q_2&=a^+\mathbf a+b^-\mathbf b,\\
\mathbf q_3&=a^+\mathbf a+b^+\mathbf b,&\mathbf q_4&=a^-\mathbf a+b^+\mathbf b,
\end{aligned}
$$

using the minimizing basis. The edge-alignment theorem underlying rotating calipers ensures that a minimum-area enclosing rectangle has a side aligned with a hull edge. **This implementation enumerates all edge orientations**, taking $O(h^2)$ work for $h$ hull vertices; it is not the optimized $O(h)$ rotating-calipers algorithm.

Rectangles below normalized area **$2\times10^{-5}$**, approximately **8.192 pixel²** at 640 × 640, are discarded. Finally, each corner coordinate is clamped to $[0,1]$ and written to six decimal places:

```text
class_index x1 y1 x2 y2 x3 y3 x4 y4
```

Equivalently, normalized coordinates are $x_i=x_{i,\mathrm{px}}/W$ and $y_i=y_{i,\mathrm{px}}/H$. The implementation already operates in normalized image coordinates. Because images are square, minimizing normalized area is equivalent to minimizing pixel area. Corner order is cyclic in the selected edge basis; the starting corner need not be the image-space top-left corner.

**Boundary limitation:** a minimum enclosing rectangle can extend beyond the clipped hull's image boundary. The final per-coordinate clamp satisfies the YOLO range constraint but can turn that rectangle into a nonrectangular quadrilateral and change enclosure/tightness. Border-truncated objects therefore need particular annotation QA. The baseline does not claim pixel-exact visible silhouettes at every viewpoint.

### 4. Optics and material parameters

For each glass mesh, the Principled BSDF is configured with

$$
T=1.0,\qquad \eta=1.5,\qquad s_{\mathrm{IOR}}=0.5,\qquad
r_{\mathrm{glass}}\sim\mathcal U(0,0.1).
$$

The initial roughness before per-frame randomization is **0.03**. Glass base colour is white with probability **0.7**; otherwise each RGB channel is independently sampled from $\mathcal U(0.9,1.0)$. Linked tube instances share their material.

For the holder,

$$
r_{\mathrm{plastic}}\sim\mathcal U(0.35,0.75),\qquad s_{\mathrm{IOR}}=0.4.
$$

Its base colour comes from `PLASTIC_PALETTE` with probability **0.8**, otherwise from the random surface-colour sampler. Plastic transmission remains at the shader default rather than being assigned the glass transmission weight.

As physical context, an air/material interface with $\eta=1.5$ has normal-incidence dielectric Fresnel reflectance

$$
F_0=\left(\frac{\eta-1}{\eta+1}\right)^2=0.04.
$$

Consequently, transmission weight 1 does **not** mean every incident ray transmits all its energy: Fresnel reflection still occurs. This Fresnel expression explains the material choice; the script does not independently implement it, a full Principled BRDF/BTDF, or a custom roughness-to-microfacet conversion. Those calculations belong to Blender's shader implementation.

The baseline uses **Cycles**, **64 samples**, adaptive sampling, denoising, AgX colour management, 12 total bounces, 12 transmission bounces, and 8 transparent bounces. Reflective and refractive caustics are disabled; glossy blur is 1.0. The recorded rendering backend was **OptiX on the RTX 4050 Laptop GPU**.

### 5. Bounded lighting and exposure correction

Each frame creates $N\in\{1,2,3\}$ point lights. A reference-distance power budget and positive weights are sampled:

$$
B\sim\mathcal U(20,75)\ \mathrm W,\qquad
w_i\sim\mathcal U(0.3,1.0),\qquad
\alpha_i=\frac{w_i}{\sum_jw_j}.
$$

For light distance $d_i\sim\mathcal U(0.5,1.3)$ metres, the assigned Blender energy is

$$
E_i=B\alpha_i\left(\frac{d_i}{0.8}\right)^2.
$$

Lights are placed using the same spherical-to-Cartesian form around the origin, with elevation **20–85°** and azimuth **0–360°**. Source radius is **0.01–0.25 m**. HSV saturation is sampled from **0–0.12** with probability 0.8, otherwise **0.25–0.6**; hue is uniform and value is 1.

The reference-distance normalization satisfies

$$
\sum_i E_i\left(\frac{0.8}{d_i}\right)^2=B\leq75\ \mathrm W.
$$

**The 75 W bound is not a literal cap on the sum of raw Blender lamp energies.** Because of distance compensation, an individual energy and the sum of energies can be as high as $75(1.3/0.8)^2\approx198.05$ W. Nor is this equation a strict irradiance bound at every object surface; orientation, source radius, occlusion, and surface position also matter.

The default procedural world strength is **0.04–0.20**. If external HDRIs are supplied, the HDRI branch is selected with probability 0.6 and uses strength **0.3–1.0**. The published baseline used **no external HDRIs or photographic floor textures**. The `MAX_ALBEDO=0.75` cap applies to generated background HSV values; it is not a cap on glass colour, every plastic palette entry, supplied images, or final rendered pixel intensity.

Initial exposure is $e\sim\mathcal U(-1.5,0)$ EV. After rendering, let $I_{p,c}$ be the loaded PNG RGB values returned by Blender's image API. The guard computes

$$
\mu=\frac{1}{3N_p}\sum_{p,c}I_{p,c},\qquad
f_{\mathrm{clip}}=\frac1{N_p}\sum_p\mathbf1[\min_c I_{p,c}>0.95].
$$

It accepts the frame when $0.28\leq\mu\leq0.60$ and $f_{\mathrm{clip}}\leq0.10$. Otherwise, the correction magnitude is

$$
\Delta e=\operatorname{clip}\left(2.2\left|\log_2\frac{\max(\mu,0.02)}{0.44}\right|,\ 0.5,\ 2.0\right).
$$

Exposure decreases by $\Delta e$ if the mean is above 0.60 or clipping exceeds 0.10; otherwise it increases by $\Delta e$. The frame is re-rendered, with at most **three corrective renders** after the initial render. Final EV is not clamped to the initial interval. This is a heuristic image-brightness guard, not a calibrated exposure model or a guarantee that every final image meets the acceptance bounds.

### 6. Procedural floor and background randomization

A **4 m × 4 m** bench plane sits at $z=-0.0002$ m. Generated coordinates feed a Mapping node, then procedural texture nodes, a three-stop colour ramp where applicable, and a bump node. Mapping rotation is uniform over **0–360°**. The selected texture is rewired and randomized for every frame; there is no flat-colour floor branch.

| Family | Sampling weight | Principal randomized parameters |
|---|---:|---|
| Wood | 2.0 | Wave X bands; SIN or SAW; scale 15–60; distortion 3–12; detail 2–8; detail scale 1–3; phase 0–100; Y mapping scale 1–3; BSDF roughness 0.30–0.65 |
| Brushed metal | 1.5 | Noise scale 5–30; detail 10–16; noise roughness 0.6–0.9; Y mapping scale 30–80; metallic 0.7–1.0; BSDF roughness 0.15–0.45 |
| Noise / terrazzo | 2.0 | Equal branch probability: Noise scale 10–120, detail 5–15, roughness 0.4–0.9, distortion 0–3; or Voronoi scale 20–150, randomness 0.6–1.0, F1 / SMOOTH_F1 / DISTANCE_TO_EDGE |
| Tiles / brick | 1.5 | Scale 5–25; mortar size 0.01–0.05; mortar smoothness 0–0.5; bias −0.3–0.3; width 0.3–0.8; row height 0.15–0.5; offset 0 or 0.5; squash sampled from [1, 1, 2]; BSDF roughness 0.20–0.60 |
| Fine grain | 1.5 | Noise scale 60–400; detail 3–8; noise roughness 0.4–0.8; colour variation amplitude 0.05–0.20 |
| Optional image texture | 3.0, only when images exist | Random supplied image; XY mapping scale 1–6; noise bump scale 100–300 |

Without supplied images, weights sum to **8.5**; selection probabilities are each weight divided by 8.5. With images, the total becomes **11.5**. BSDF roughness starts at a random **0.30–0.80** and is overridden by the families listed above. Bump strength is **0.05–0.35**, and bump distance is **0.01–0.10** in Blender node units. Noise-node roughness is a different parameter from BSDF surface roughness.

The default world uses a colour-ramped noise background with scale **1–6**, detail **2–6**, and distortion **0–1**, rather than a sterile constant colour. Texture and HDRI internals are Blender shader-node algorithms; the Python code selects parameters and connections, not a separate hand-written noise or BRDF implementation.

## Training and evaluation protocol

The reported run is **`stl_synthetic_800_30ep_holdout`**.

| Setting | Value |
|---|---|
| Initialization | Original DOTA-pretrained `yolov8n-obb.pt` |
| Training images | 800 only |
| Epochs | 30, all completed |
| Input size / batch | 640 / 16 |
| Training seed | 42 |
| Device | CUDA device 0, RTX 4050 Laptop GPU |
| Workers | 0 |
| Resume | False |
| Deterministic option | True |
| Early stopping | Disabled (`patience=0`) |
| Optimizer selection | Ultralytics `auto` |
| Training AMP | Enabled |
| Mosaic / MixUp | 1.0 / 0.0; mosaic closed for the last 10 epochs |
| Checkpoint selection | Validation mAP50–95 fitness; test not used for selection |
| Standalone evaluation | Batch 16, 640 pixels, workers 0, NMS IoU 0.7, confidence cutoff 0.001 |
| Test preview inference | Confidence 0.25, NMS IoU 0.7; ten fixed-seed selections |

“Fresh” or “from scratch” in the reproduction commands means a **new fine-tuning run from the original pretrained Nano checkpoint**, not random parameter initialization and not resuming the earlier model that saw all 1,000 images.

The original full arguments and environment-specific paths are preserved in [`results/baseline/training/args.yaml`](results/baseline/training/args.yaml) and [`training_provenance.json`](results/baseline/training/training_provenance.json). They are historical records, not portable configuration files; use the commands below to reproduce the run in another checkout.

### Metric definitions and interpretation

For a class and a specified matching criterion,

$$
P=\frac{TP}{TP+FP},\qquad R=\frac{TP}{TP+FN},\qquad F_1=\frac{2PR}{P+R}.
$$

Ultralytics computes interpolated precision–recall AP using a 101-point recall grid and trapezoidal integration. With $C=4$ classes and thresholds $\mathcal T=\{0.50,0.55,\ldots,0.95\}$,

$$
\operatorname{mAP50}=\frac1C\sum_{c=1}^C AP_c(0.50),\qquad
\operatorname{mAP50\text{–}95}=\frac1{10C}\sum_{\tau\in\mathcal T}\sum_{c=1}^C AP_c(\tau).
$$

The installed Ultralytics OBB validator uses **`batch_probiou`** for oriented-box matching. These library metrics should not be silently treated as an independently implemented polygon-IoU benchmark. NMS IoU 0.7 is a suppression parameter, distinct from the AP matching thresholds.

The reported Precision and Recall are Ultralytics' class-mean values at the split's smoothed mean-F1 maximum. They are **not** precision/recall at the preview confidence cutoff of 0.25, nor measurements of a preselected real-world deployment threshold. AP summarizes the confidence ranking; mAP is not the fraction of successful robot picks.

The standalone validation/test confusion matrices also use the explicitly requested low confidence cutoff **0.001**. Their raw false-positive counts include very low-confidence detections and must not be interpreted as deployment error counts at 0.25 or a future operating threshold.

## Experimental results

### Aggregate held-out results

All quality metrics below are percentages; timings are milliseconds per image as reported by Ultralytics on the **NVIDIA GeForce RTX 4050 Laptop GPU**.

| Split | Images | Instances | Precision | Recall | mAP50 | mAP50–95 | Inference ms/image |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | 100 | 566 | **98.42** | **97.16** | **98.47** | **89.79** | **8.44** |
| Test | 100 | 566 | **97.04** | **95.76** | **97.16** | **88.29** | **5.49** |

| Split | Preprocess ms/image | Model inference ms/image | Postprocess ms/image |
|---|---:|---:|---:|
| Validation | 1.51 | 8.44 | 4.90 |
| Test | 0.76 | 5.49 | 5.66 |

Timings come from separate 100-image, batch-16 evaluation passes. GPU warm-up, utilization, thermals, and execution context can affect them. They are **not single-frame end-to-end robot latency**, and the validation/test timing difference is not evidence of an algorithmic speed improvement.

Exact unrounded values are preserved in [validation metrics](results/baseline/validation/metrics.json) and [test metrics](results/baseline/test/metrics.json). The selected `best.pt` SHA256 is:

```text
8bd74b415a4873487b253b1996d0ba4b5b84d88398fa012f8dec22d3c0662ea1
```

### Per-class results

| Class | Validation AP50 | Validation AP50–95 | Test AP50 | Test AP50–95 |
|---|---:|---:|---:|---:|
| Tube_Holder | 99.50 | 96.78 | 99.50 | 96.23 |
| Glass_Tube | 97.94 | 84.62 | 97.96 | 85.86 |
| Slide_Holder | 98.44 | 89.41 | 97.27 | 87.05 |
| Eppendorf_Rack | 98.00 | 88.36 | 93.91 | 84.00 |

The test checkpoint was frozen before quantitative test evaluation; it was not retrained or selected using the test score. Ten fixed-seed test predictions had previously been inspected for visual QA, so this is not claimed to be a preregistered, fully blinded study. The shared CAD assets and renderer also make these same-domain synthetic holdouts, not unseen-object or real-camera benchmarks.

### Learning curves and diagnostic plots

![Training and validation learning curves across 30 epochs](results/baseline/training/results.png)

| Validation precision–recall | Test precision–recall |
|---|---|
| ![Validation PR curve](results/baseline/validation/BoxPR_curve.png) | ![Test PR curve](results/baseline/test/BoxPR_curve.png) |

| Validation F1–confidence | Test F1–confidence |
|---|---|
| ![Validation F1 curve](results/baseline/validation/BoxF1_curve.png) | ![Test F1 curve](results/baseline/test/BoxF1_curve.png) |

All original training-batch images, validation-batch images, precision/recall curves, confusion matrices, and CSV records are included in the corresponding result directories. Standalone [validation confusion matrices](results/baseline/validation/confusion_matrix.png) and [test confusion matrices](results/baseline/test/confusion_matrix.png) use confidence **0.001**, as explained above. The per-epoch CSV and standalone checkpoint evaluation are separate passes; the headline scores are taken from the standalone metric JSON files.

## Qualitative results

### Validation ground truth and predictions

| Ground truth: validation batch 0 | Predictions: validation batch 0 |
|---|---|
| ![Validation batch 0 labels](results/baseline/validation/val_batch0_labels.jpg) | ![Validation batch 0 predictions](results/baseline/validation/val_batch0_pred.jpg) |

![Validation batch 1 predictions](results/baseline/validation/val_batch1_pred.jpg)

![Validation batch 2 predictions](results/baseline/validation/val_batch2_pred.jpg)

Ground-truth overlays are supplied to distinguish annotation conventions from model errors. Confidence filtering used for plots and preview inference is not the definition of AP.

### Ten fixed-seed test previews

Inference processed all **100** test images. These ten previews were selected with `random.Random(42).sample(sorted(test_stems), 10)`, independently of prediction confidence or visual quality. Their boxes, class names, and confidence scores are predictions, not ground truth. Preview inference uses **confidence 0.25** and **NMS IoU 0.7**.

| Test example | Test example |
|---|---|
| ![Test 000030](results/baseline/test/previews/000030.png) | ![Test 000099](results/baseline/test/previews/000099.png) |
| ![Test 000103](results/baseline/test/previews/000103.png) | ![Test 000127](results/baseline/test/previews/000127.png) |
| ![Test 000228](results/baseline/test/previews/000228.png) | ![Test 000250](results/baseline/test/previews/000250.png) |
| ![Test 000296](results/baseline/test/previews/000296.png) | ![Test 000825](results/baseline/test/previews/000825.png) |
| ![Test 000867](results/baseline/test/previews/000867.png) | ![Test 000954](results/baseline/test/previews/000954.png) |

The complete 100-image prediction record, including polygon coordinates in pixels, is available in [`results/baseline/test/predictions.json`](results/baseline/test/predictions.json).

## Reproducibility

### 1. Clone and install the recorded ML environment

The recorded environment used **Windows, Python 3.12.9, Blender 4.3.2, Ultralytics 8.4.127, PyTorch 2.5.1+cu121, and torchvision 0.20.1+cu121**. Blender runs its own Python interpreter and includes `bpy`, `mathutils`, and NumPy; do not install `bpy` into the ML virtual environment for headless generation.

Run the following in PowerShell:

```powershell
git clone https://github.com/Danial-Umer710/ROS2-Synthetic-Dataset-Generator.git
cd ROS2-Synthetic-Dataset-Generator
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
# optional, only for the ControlNet stylization stage on this branch:
python -m pip install -r requirements-controlnet.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Use a CUDA-capable NVIDIA driver compatible with the installed PyTorch wheel. The final command should report CUDA availability and the selected GPU. `requirements.txt` pins the measured package versions; the CUDA wheel installation is intentionally shown separately. `requirements-controlnet.txt` adds the diffusion stack (diffusers 0.40.0, transformers 5.16.1, OpenEXR) pinned to the newest releases that still run on torch 2.5.1 — the file header explains the constraint chain. The four HF model snapshots are fetched on demand by `scripts/controlnet_setup.py --download`; they are not vendored in the repository.

The complete dataset and trained weights are already included in the repository. Regeneration and retraining are optional if the objective is only to inspect or evaluate the published baseline.

### 2. Verify the published dataset

From the repository root:

```powershell
python scripts/yolo_dataset_splitter.py --dataset dataset --check
```

This checks the saved split rather than generating new random memberships. It should report 800/100/100 images and labels, zero overlapping stems, and zero duplicate pixel hashes. Do not change the dataset or manifest to accommodate a failed integrity check without first identifying the cause.

### 3. Generate a new dataset in headless Blender

Use a **new output directory** so the published split remains untouched:

```powershell
& "C:\Program Files\Blender Foundation\Blender 4.3\blender.exe" -b -P "blender/generate_synthetic_dataset.py" -- --stl-dir "stl_models" --out-dir "dataset_reproduced" --num-images 1000 --resolution 640 --samples 64 --engine cycles --label-mode visible --seed 42
python scripts/yolo_dataset_splitter.py --dataset dataset_reproduced --seed 42 --apply
python scripts/yolo_dataset_splitter.py --dataset dataset_reproduced --seed 42 --check
python scripts/visualize_obb_labels.py --dataset dataset_reproduced --split val --num 12
```

Blender must be installed separately; adjust its executable path if necessary. Everything after `--` is a generator argument. On Linux/macOS, substitute the installed `blender` executable for the Windows path.

For a small installation smoke test, use `--num-images 4 --samples 4 --out-dir dataset_smoke_new` and visualize its `train` labels. Do not run the fixed 1,000-image splitter on that small smoke dataset.

The generator refuses an output directory containing a frozen split manifest. Within an unsplit render directory it can resume by skipping existing image/label pairs; `--overwrite` is an explicit request to re-render those unsplit frames. The initially generated YAML uses the training folder as a validation placeholder, so **split before training**. Optional `--textures-dir` and `--hdri-dir` inputs change the experiment and were not used for the reported baseline.

### 4. Start a fresh 30-epoch training run

To reproduce training using the **published** dataset:

```powershell
python -u scripts/train_yolo.py --dataset dataset --weights yolov8n-obb.pt --epochs 30 --imgsz 640 --batch 16 --device 0 --name stl_synthetic_800_30ep_reproduction
```

To use newly rendered images instead, substitute `--dataset dataset_reproduced`. The default original pretrained checkpoint is retained at the repository root; Ultralytics can also download the named pretrained model if missing. The script rejects a labware checkpoint as the initialization for this fresh baseline workflow.

The script checks data integrity, trains on 800 images, validates the selected `best.pt` on 100 validation images, predicts the 100 test images, and saves ten preview images. New runs go under `runs/obb/UR16e_Labware_Detection/`; existing published artifacts are not overwritten. Running this command launches training, not just a preview viewer.

For controlled experiments, use validation data to choose settings. Do not repeatedly use the test set to select epochs, augmentations, thresholds, or checkpoints.

### 5. Evaluate a published frozen checkpoint without training

Each evaluation requires a new output directory so previous measurements remain intact:

```powershell
python scripts/evaluate_yolo.py --weights weights/best.pt --dataset dataset --split val --imgsz 640 --batch 16 --device 0 --output runs/obb/reproduce_validation
python scripts/evaluate_yolo.py --weights weights/best.pt --dataset dataset --split test --imgsz 640 --batch 16 --device 0 --output runs/obb/reproduce_test
```

The evaluator verifies the four-class mapping and split integrity, checks evaluated instance counts, and saves `metrics.json`, PR/F1/P/R curves, confusion matrices, and annotated image batches. It records checkpoint and split-manifest hashes. `--device cpu` is available for functional evaluation, but CPU timings are not comparable with the published RTX 4050 timings.

### 6. Inspect labels and publication provenance

```powershell
python scripts/visualize_obb_labels.py --dataset dataset --split val --num 12
python scripts/prepare_publication.py
python -m py_compile blender/generate_synthetic_dataset.py scripts/yolo_dataset_splitter.py scripts/train_yolo.py scripts/evaluate_yolo.py scripts/prepare_publication.py
```

The publication audit reports dataset integrity and asset sizes. `prepare_publication.py --copy-assets` is the local maintainer's packaging operation: it expects the original baseline run and original STL input directory, copies artifacts into the publication layout, and refuses to overwrite differing files. It is not needed by a fresh clone, and it does not retrain or recompute reported metrics.

[`publication_manifest.json`](results/baseline/publication_manifest.json) maps the published assets to their original local-run sources and records SHA256 values. Absolute Windows paths embedded in frozen training/evaluation metadata are retained as provenance, not required paths for a new checkout.

## Limitations and next experiments

1. **One real benchmark session.** Real-camera transfer is measured on a single 282-frame session from `ABC-iRobotics/LabWare_Items` with auto-generated mask annotations (which the dataset card says "must be independently reviewed"). A labelled, session-separated set from the intended robot camera, bench, and lighting is still needed.
2. **Shared assets and renderer.** All splits use the same four CAD models and procedural rendering system, and the stylized datasets all pass through one frozen diffusion configuration. This is not an unseen-geometry or unseen-appearance benchmark.
3. **Limited sample size.** Each synthetic holdout contains only 100 images (the real set 282). Multiple instances within an image are correlated; the tables are point estimates, not statistical certification of a deployment failure rate.
4. **Approximate annotations.** Vertex-sampled visibility, see-through rather than refracted visibility rays, frame-area filtering, and coordinate clamping at image borders can differ from a manually defined visible silhouette.
5. **Geometry and materials vs. the physical labware.** `Falcon_Tube.stl` models a 50 ml tube; the real benchmark's tubes are ~15 ml conical tubes — a real size mismatch remains. Glass assignments for the loose stand and rack, mesh wall thickness, liquids, printing, and surface finish need comparison with actual objects.
6. **fg-keep < 1.0 relaxes the label guarantee.** At fg-keep 0.5, half of each labelled object's pixels are diffusion output; if the img2img step drifts an edge inside the mask (rare under depth+edge conditioning at strength 0.45, but possible), the OBB no longer describes it exactly. fg-keep 1.0 exists for the strict guarantee; the sweep over fg-keep and strength values is incomplete — only 1.0 and 0.5 at strength 0.45 have published scores.
7. **Context bias and negatives.** The images always include the main holder. Distractors and loose tubes appear only near the holder; empty scenes, tubes far from the holder, and real clutter are still absent. The splitter intentionally validates a specific 1,000-image, nonempty-label snapshot; a broader dataset needs a versioned configuration rather than bypassing its checks.
8. **OBB is not robot pose.** Visible-extent tube boxes can cover only the exposed part above an opaque holder. Camera calibration, depth/pose estimation, grasp-point validation, temporal stability, and supervised pick tests remain necessary.
9. **Future robotics observability.** ROS2/UR16e integration and Prometheus/Grafana monitoring of inference latency, detection failures, and robot outcomes are planned work, not evaluated components of this release.

A controlled longer-training run may improve synthetic validation performance, but it should be judged against the frozen validation split and followed by an independently maintained final test set. More rendering volume alone is not evidence of improved transfer to the real laboratory.

## Provenance and licensing

- The four source STL files are published unchanged under their original filenames. Original mesh authorship and license notices were not supplied with the local STL directory; no new mesh-license grant is asserted here.
- A standalone project-wide `LICENSE` file was not present in the inherited repository. This README does not invent an MIT, CC0, or other blanket license for the code, CAD assets, dataset, or weights.
- Ultralytics software and its pretrained/model-derived artifacts remain subject to the applicable [upstream licensing terms](https://github.com/ultralytics/ultralytics). Review those terms and obtain the necessary asset permissions for your intended use.
- The diffusion models used by the ControlNet stage (Realistic Vision V5.1, `sd-vae-ft-mse`, and the two `lllyasviel` SD1.5 ControlNets) are downloaded from Hugging Face at pinned revisions and are **not vendored** in this repository. Stable Diffusion 1.5 derivatives are governed by the [CreativeML OpenRAIL-M license](https://huggingface.co/spaces/CompVis/stable-diffusion-license); review each model card for its terms before redistribution or commercial use.
- The published raw metric records, split manifest, and checksums are the evidence for this baseline. Historical experiments elsewhere in the repository are not interchangeable with these artifacts.

### References and tools

- [Blender](https://www.blender.org/) — rendering, mesh geometry, camera projection, and Principled materials.
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — pretrained Nano OBB model, training, inference, and evaluation.
- [diffusers](https://github.com/huggingface/diffusers) — Multi-ControlNet img2img pipeline (SD1.5 + depth + canny).
- [ControlNet (Zhang et al.)](https://arxiv.org/abs/2302.05543) — adding spatial conditioning controls to diffusion models.
- [ABC-iRobotics/LabWare_Items](https://huggingface.co/datasets/ABC-iRobotics/LabWare_Items) — real UR16e labware benchmark (MIT license).
- [GitHub large-file documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github) — publication limits and storage guidance.

**Project maintainer:** Danial Umer.

**Repository:** [ROS2-Synthetic-Dataset-Generator](https://github.com/Danial-Umer710/ROS2-Synthetic-Dataset-Generator).
