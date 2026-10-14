import argparse
import json
import random
from pathlib import Path

import torch
from ultralytics import YOLO

from yolo_dataset_splitter import CLASS_NAMES, PROJECT_ROOT, SPLIT_COUNTS, file_hash, inspect_dataset

SEED = 42
PREVIEW_COUNT = 10
PREVIEW_CONF = 0.25


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def summarize_metrics(metrics):
    keys = ("precision", "recall", "mAP50", "mAP50-95")
    return {
        "overall": dict(zip(keys, map(float, metrics.box.mean_results()))),
        "per_class": [
            {
                "class_id": int(class_id),
                "name": metrics.names[int(class_id)],
                "instances": int(metrics.nt_per_class[int(class_id)]),
                **dict(zip(keys, map(float, metrics.box.class_result(index)))),
            }
            for index, class_id in enumerate(metrics.box.ap_class_index)
        ],
        "speed_ms_per_image": {key: float(value) for key, value in metrics.speed.items()},
    }


def export_test_predictions(model, dataset, run_dir, manifest, device="0", batch=16, imgsz=640):
    test_dir = dataset / "images" / "test"
    expected = set(manifest["splits"]["test"])
    images = sorted(test_dir.glob("*.png"))
    if len(images) != 100 or {path.stem for path in images} != expected:
        raise ValueError("Test image paths differ from the saved 100-image holdout")
    selected = set(random.Random(SEED).sample(sorted(expected), PREVIEW_COUNT))
    preview_dir = PROJECT_ROOT / f"test_predictions_preview_{run_dir.name}"
    preview_dir.mkdir(exist_ok=False)
    predictions, seen = [], set()
    results = model.predict(
        source=str(test_dir),
        stream=True,
        save=False,
        save_txt=False,
        imgsz=imgsz,
        batch=batch,
        device=device,
        conf=PREVIEW_CONF,
        iou=0.7,
        verbose=False,
    )
    for result in results:
        path = Path(result.path)
        if path.stem not in expected or path.stem in seen or result.obb is None:
            raise ValueError(f"Unexpected or duplicate OBB prediction result: {path}")
        seen.add(path.stem)
        predictions.append({
            "image": path.name,
            "detections": result.summary(normalize=False, decimals=6),
        })
        if path.stem in selected:
            result.save(
                filename=str(preview_dir / path.name),
                conf=True,
                labels=True,
                line_width=2,
            )
    if seen != expected:
        raise ValueError("Inference did not process every test image")
    preview_images = sorted(preview_dir.glob("*.png"))
    if len(preview_images) != PREVIEW_COUNT or {path.stem for path in preview_images} != selected:
        raise ValueError("Expected exactly the 10 preselected prediction previews")
    prediction_path = run_dir / "test_predictions.json"
    write_json(prediction_path, {
        "split": "test",
        "images": len(predictions),
        "confidence_threshold": PREVIEW_CONF,
        "nms_iou": 0.7,
        "coordinate_space": "image_pixels",
        "preview_seed": SEED,
        "preview_directory": str(preview_dir),
        "preview_images": [path.name for path in preview_images],
        "predictions": predictions,
    })
    return {"images": len(predictions), "predictions": str(prediction_path), "previews": str(preview_dir)}


def main():
    parser = argparse.ArgumentParser(description="Fresh YOLOv8n-OBB training on the frozen 800/100/100 STL split.")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "dataset")
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "yolov8n-obb.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "runs" / "obb" / "UR16e_Labware_Detection")
    parser.add_argument("--name", default="stl_synthetic_800_30ep_holdout")
    args = parser.parse_args()
    if min(args.epochs, args.imgsz, args.batch) <= 0:
        parser.error("epochs, imgsz and batch must be positive")
    dataset = args.dataset.resolve()
    manifest, _, _, split_summary = inspect_dataset(dataset, SEED, require_split=True)
    manifest_bytes = (dataset / "split_manifest.json").read_bytes()
    if args.device != "cpu" and (not args.device.isdecimal() or not torch.cuda.is_available()):
        raise RuntimeError("Use one available CUDA device index or --device cpu")
    gpu = "CPU" if args.device == "cpu" else torch.cuda.get_device_name(int(args.device))
    model = YOLO(str(args.weights.resolve()))
    source_weights = Path(model.ckpt_path).resolve()
    source_hash = file_hash(source_weights)
    if model.task != "obb" or len(model.names) != 15 or model.names.get(0) != "plane":
        raise ValueError("Expected the original DOTA-pretrained Nano OBB checkpoint, not labware weights")
    training_args = {
        "data": str(dataset / "data.yaml"),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": 0,
        "seed": SEED,
        "deterministic": True,
        "resume": False,
        "pretrained": True,
        "patience": 0,
        "project": str(args.project.resolve()),
        "name": args.name,
        "exist_ok": False,
    }
    print(f"Fresh training from {source_weights} on device {args.device} ({gpu})", flush=True)
    print(json.dumps(split_summary, indent=2), flush=True)
    model.train(**training_args)
    run_dir = Path(model.trainer.save_dir).resolve()
    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.is_file():
        raise FileNotFoundError(best_weights)
    if (dataset / "split_manifest.json").read_bytes() != manifest_bytes:
        raise ValueError("Split manifest changed during training")
    (run_dir / "split_manifest.json").write_bytes(manifest_bytes)
    write_json(run_dir / "training_provenance.json", {
        "initial_weights": str(source_weights),
        "initial_weights_sha256": source_hash,
        "resume": False,
        "gpu": gpu,
        "split_seed": SEED,
        "splits": split_summary,
        "training_args": training_args,
    })
    best = YOLO(str(best_weights))
    if best.names != CLASS_NAMES:
        raise ValueError("Trained checkpoint has an unexpected class mapping")
    metrics = best.val(
        data=str(dataset / "data.yaml"),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        conf=0.001,
        iou=0.7,
        plots=True,
        project=str(run_dir),
        name="heldout_val",
        exist_ok=False,
    )
    if [int(value) for value in metrics.nt_per_class] != split_summary["val"]["instances_by_class"]:
        raise ValueError("Validation instance counts differ from the held-out manifest")
    report = {
        "split": "val",
        "images": SPLIT_COUNTS["val"],
        "checkpoint": str(best_weights),
        "split_seed": SEED,
        **summarize_metrics(metrics),
    }
    write_json(run_dir / "heldout_metrics.json", report)
    print("\n=== Held-out Validation Metrics (best.pt, 100 images) ===", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    test_outputs = export_test_predictions(best, dataset, run_dir, manifest, args.device, args.batch, args.imgsz)
    print("\n=== Test Inference Complete ===", flush=True)
    print(json.dumps(test_outputs, indent=2), flush=True)
    print(f"Run directory: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
