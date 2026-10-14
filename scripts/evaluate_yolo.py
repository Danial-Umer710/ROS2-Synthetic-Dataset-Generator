import argparse
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO

from train_yolo import summarize_metrics, write_json
from yolo_dataset_splitter import CLASS_NAMES, PROJECT_ROOT, file_hash, inspect_dataset


def main():
    parser = argparse.ArgumentParser(description="Evaluate a frozen OBB checkpoint without training or changing the split.")
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "weights" / "best.pt")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "dataset")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset, weights, output = args.dataset.resolve(), args.weights.resolve(), args.output.resolve()
    manifest, _, _, splits = inspect_dataset(dataset, 42, require_split=True)
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if output.exists():
        raise FileExistsError(f"Choose a new --output directory to preserve existing evaluation artifacts: {output}")
    if args.device != "cpu" and (not args.device.isdecimal() or not torch.cuda.is_available()):
        raise RuntimeError("Use one available CUDA device index or --device cpu")
    device_name = "CPU" if args.device == "cpu" else torch.cuda.get_device_name(int(args.device))
    model = YOLO(str(weights))
    if model.task != "obb" or model.names != CLASS_NAMES:
        raise ValueError("Checkpoint must contain the four STL labware classes in their original order")
    checkpoint_hash = file_hash(weights)
    manifest_hash = file_hash(dataset / "split_manifest.json")
    output.mkdir(parents=True)
    metrics = model.val(
        data=str(dataset / "data.yaml"),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        conf=0.001,
        iou=0.7,
        half=False,
        plots=True,
        project=str(output.parent),
        name=output.name,
        exist_ok=True,
    )
    if [int(value) for value in metrics.nt_per_class] != splits[args.split]["instances_by_class"]:
        raise ValueError("Evaluated instance counts differ from the frozen split manifest")
    if checkpoint_hash != file_hash(weights) or manifest_hash != file_hash(dataset / "split_manifest.json"):
        raise ValueError("Checkpoint or dataset manifest changed during evaluation")
    report = {
        "split": args.split,
        "images": splits[args.split]["images"],
        "instances": sum(splits[args.split]["instances_by_class"]),
        "checkpoint": str(weights),
        "checkpoint_sha256": checkpoint_hash,
        "split_seed": manifest["seed"],
        "split_manifest_sha256": manifest_hash,
        "configuration": {
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "device_name": device_name,
            "workers": 0,
            "confidence_cutoff": 0.001,
            "nms_iou": 0.7,
            "half": False,
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
        },
        **summarize_metrics(metrics),
    }
    write_json(output / "metrics.json", report)
    print(f"Frozen-checkpoint {args.split} evaluation saved to {output / 'metrics.json'}")
    print(report["overall"])


if __name__ == "__main__":
    main()
