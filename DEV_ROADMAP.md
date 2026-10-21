# Development Roadmap (`dev` Branch)

This branch tracks active Sim2Real experiments building on the clean v2 STL baseline on `main`.

## Active Tasks (from Review Feedback)

1. **Add Loose Glass Tube Next to Holder:**
   - Extend `blender/generate_synthetic_dataset.py` (`place_objects`) so `Glass_Tube` (`Tube.stl`) instances are also placed loose on the bench beside `Tube_Holder` (both upright and lying horizontally), in addition to the tubes seated inside the rack holes.
2. **Distractor / Hindrance Models:**
   - Source and import visually similar transparent/matte labware and clutter meshes as negative/distractor objects (`role="distractor"`, unlabelled) to harden YOLOv8-OBB against false positives.
3. **Reproducibility with Docker:**
   - Add a `Dockerfile` and `docker-compose.yml` encapsulating headless Blender 4.3 + CUDA PyTorch/Ultralytics so dataset generation, splitting, training, and evaluation run identically in a container.

## Included Experimental Runs

- **`results/baseline/`:** Held-out 80/10/10 split baseline (`stl_synthetic_800_30ep_holdout`: **89.79% validation mAP50-95**, **88.29% test mAP50-95**).
- **`results/experiments/stl_synthetic_1000_30ep/`:** Initial 30-epoch full-1,000-image validation run (`92.69% in-sample mAP50-95`, `98.69% mAP50`), including weights (`best.pt`, `last.pt`), metric curves, and validation batch predictions.
