# BlenderProc Pipeline (`blenderproc` Branch)

This branch is dedicated to migrating the UR16e labware synthetic dataset pipeline from raw headless `bpy` (`blender/generate_synthetic_dataset.py`) to **[BlenderProc](https://github.com/DLR-RM/BlenderProc)**.

## Inherited Baseline Assets on This Branch

- **`stl_models/`:** The 4 CAD meshes (`TubeHolder.stl`, `Tube.stl`, `Slide_holder.stl`, `eppendorf_tarto.stl`).
- **`dataset/`:** The 1,000-image 80/10/10 split dataset and `split_manifest.json` from the `bpy` baseline for direct benchmark comparison.
- **`weights/` & `results/baseline/`:** The YOLOv8n-OBB baseline checkpoints and validation/test metrics (**89.79% val mAP50-95**, **88.29% test mAP50-95**).

## BlenderProc Migration Scope

1. **Loader & Physics Positioning (`bproc.loader` / `bproc.object`):**
   - Load `stl_models/*.stl` with physically simulated rigid-body placement for loose tubes next to the holder and distractor objects on the bench.
2. **Material & Texture Randomization (`bproc.material` / `cc_textures`):**
   - Randomize Principled BSDF transmission/roughness for glass labware and randomize PBR bench/world textures.
3. **Camera & Pose Sampling (`bproc.sampler.shell`):**
   - Sample camera poses on the hemisphere around the `Tube_Holder` with obstacle/occlusion checks.
4. **Annotation & Export (`bproc.writer` -> YOLOv8-OBB):**
   - Extract instance segmentation masks / 3D-to-2D projected contours and convert minimum-area rotated bounding boxes to normalized YOLOv8-OBB labels (`class_index x1 y1 x2 y2 x3 y3 x4 y4`).
