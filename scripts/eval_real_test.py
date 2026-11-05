"""
eval_real_test.py
=================
Evaluate a frozen YOLOv8-OBB checkpoint on the real UR16e camera test set
produced by ``scripts/prep_real_test_set.py`` (``dataset/real_test/``).

Saves Ultralytics val artifacts (curves, confusion matrix, batch mosaics),
a ``metrics.json`` summary, prediction overlays for every real frame, and a
``predictions.json`` detection dump -- the Sim2Real baseline artefacts.

    python scripts/eval_real_test.py --weights weights/best.pt --output results/baseline/real_world_eval
"""

import argparse
import json
import random
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO

from train_yolo import summarize_metrics, write_json
from yolo_dataset_splitter import CLASS_NAMES, PROJECT_ROOT, file_hash

PREVIEW_CONF = 0.25
PREVIEW_COUNT = 10
SEED = 42


def export_predictions(model, images_dir, out_dir, device, batch, imgsz):
    """Run OBB inference on every real frame; save overlays + detections JSON."""
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True)
    images = sorted(images_dir.glob("*.png"))
    selected = set(random.Random(SEED).sample([p.stem for p in images],
                                            min(PREVIEW_COUNT, len(images))))
    predictions = []
    for result in model.predict(source=str(images_dir), stream=True, save=False,
                                imgsz=imgsz, batch=batch, device=device,
                                conf=PREVIEW_CONF, iou=0.7, verbose=False):
        stem = Path(result.path).stem
        predictions.append({"image": Path(result.path).name,
                            "detections": result.summary(normalize=False, decimals=6)})
        # Ultralytics names saves <stem>.jpg for non-JPEG sources.
        result.save(filename=str(pred_dir / f"{stem}.jpg"),
                    conf=True, labels=True, line_width=2)
    write_json(out_dir / "predictions.json", {
        "split": "test", "images": len(predictions),
        "confidence_threshold": PREVIEW_CONF, "nms_iou": 0.7,
        "coordinate_space": "image_pixels",
        "preview_stems_fixed_seed": sorted(selected),
        "predictions": predictions,
    })
    return len(predictions)


def main():
    parser = argparse.ArgumentParser(description="Evaluate an OBB checkpoint on the real LabWare test set.")
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "weights" / "best.pt")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "dataset" / "real_test")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-predict", action="store_true")
    args = parser.parse_args()
    dataset, weights, output = args.dataset.resolve(), args.weights.resolve(), args.output.resolve()
    manifest_path = dataset / "real_test_manifest.json"
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run scripts/prep_real_test_set.py first: {manifest_path}")
    if output.exists():
        raise FileExistsError(f"Choose a new --output directory: {output}")
    if args.device != "cpu" and (not args.device.isdecimal() or not torch.cuda.is_available()):
        raise RuntimeError("Use one available CUDA device index or --device cpu")

    manifest = json.loads(manifest_path.read_text())
    expected = manifest["splits"]["test"]["instances_by_class"]
    model = YOLO(str(weights))
    if model.task != "obb" or model.names != CLASS_NAMES:
        raise ValueError("Checkpoint must contain the four STL labware classes in their original order")

    output.mkdir(parents=True)
    metrics = model.val(
        data=str(dataset / "data.yaml"), split="test",
        imgsz=args.imgsz, batch=args.batch, device=args.device,
        workers=0, conf=0.001, iou=0.7, half=False, plots=True,
        project=str(output), name="yolo_val", exist_ok=True,
    )
    actual = {CLASS_NAMES[i]: int(v) for i, v in enumerate(metrics.nt_per_class)}
    if actual != {k: expected[k] for k in expected if expected[k]}:
        raise ValueError(f"instance counts differ from real_test manifest: {actual} vs {expected}")

    n_pred = 0
    if not args.skip_predict:
        n_pred = export_predictions(model, dataset / "images" / "test",
                                    output, args.device, args.batch, args.imgsz)
    report = {
        "dataset": "ABC-iRobotics/LabWare_Items instance_segmentation test split (real UR16e camera)",
        "split": "test",
        "images": manifest["splits"]["test"]["images"],
        "instances": sum(expected.values()),
        "checkpoint": str(weights),
        "checkpoint_sha256": file_hash(weights),
        "manifest_sha256": file_hash(manifest_path),
        "prediction_overlays": n_pred,
        "configuration": {
            "imgsz": args.imgsz, "batch": args.batch, "device": args.device,
            "workers": 0, "confidence_cutoff": 0.001, "nms_iou": 0.7,
            "half": False, "ultralytics": ultralytics.__version__, "torch": torch.__version__,
            "native_resolution": "1280x800 letterboxed to imgsz",
        },
        **summarize_metrics(metrics),
    }
    write_json(output / "metrics.json", report)
    print(f"real-test evaluation saved to {output / 'metrics.json'}")
    print(report["overall"])


if __name__ == "__main__":
    main()
