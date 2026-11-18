"""
train_mixed.py
==============
Train one Sim+Real mixture built by ``scripts/build_mixed_datasets.py``.

The recipe is the one used by ``scripts/train_yolo.py`` for every published
model: fresh DOTA-pretrained ``yolov8n-obb.pt``, 30 epochs, imgsz 640, batch 16,
workers 0, deterministic, patience 0, and ``--train-seed`` as the Ultralytics
seed.  The only differences are the data: the training list is the mixture's
``train.txt`` and checkpoint selection (``best.pt``) uses the real_v2
validation capture for every mixture, including 0 percent real.

    python -u scripts/train_mixed.py --mixture mixed_datasets/real20 --train-seed 0 --name mixed_real20_seed0

Evaluate the resulting ``best.pt`` on the held-out captures with
``scripts/eval_real_test.py --dataset dataset/real_v2 --skip-predict``.
"""

import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO

from train_yolo import summarize_metrics, write_json
from yolo_dataset_splitter import CLASS_NAMES, PROJECT_ROOT, file_hash

TOTAL = 800
HELDOUT_CAPTURES = ("0006", "0007", "0009")


def load_mixture(mixture):
    """Validate the mixture folder and return (manifest, resolved train images)."""
    manifest = json.loads((mixture / "mixture_manifest.json").read_text(encoding="utf-8"))
    lines = (mixture / "train.txt").read_text(encoding="utf-8").split()
    images = [(mixture / line).resolve() for line in lines]
    listed = [(PROJECT_ROOT / entry["image"]).resolve() for entry in manifest["images"]]
    distinct = manifest.get("distinct_images", TOTAL)
    if len(images) != TOTAL or len(set(images)) != distinct or images != listed:
        raise ValueError(f"train.txt must list exactly the {TOTAL} entries ({distinct} distinct images) of the manifest")
    real = sum(entry["source"] == "real" for entry in manifest["images"])
    if real != manifest["real_images"] or TOTAL - real != manifest["synthetic_images"]:
        raise ValueError("manifest source counts are inconsistent")
    for image, entry in zip(images, manifest["images"]):
        if not image.is_file() or not (PROJECT_ROOT / entry["label"]).is_file():
            raise FileNotFoundError(image)
        if entry["source"] == "real" and image.stem.split("_")[0] in HELDOUT_CAPTURES:
            raise ValueError(f"held-out capture in the training list: {image.name}")
        if image.parent.name != "train":
            raise ValueError(f"training image outside a train split: {image}")
    return manifest, images


def main():
    parser = argparse.ArgumentParser(description="Fresh YOLOv8n-OBB training on one Sim+Real mixture.")
    parser.add_argument("--mixture", type=Path, required=True, help="mixed_datasets/real<NN> folder")
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "yolov8n-obb.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "runs" / "obb" / "UR16e_Labware_Detection")
    parser.add_argument("--name", required=True)
    parser.add_argument("--train-seed", type=int, default=42, help="Ultralytics training seed")
    args = parser.parse_args()
    if min(args.epochs, args.imgsz, args.batch) <= 0:
        parser.error("epochs, imgsz and batch must be positive")
    mixture = args.mixture.resolve()
    manifest, images = load_mixture(mixture)
    val_manifest = json.loads((PROJECT_ROOT / "dataset" / "real_v2" / "real_v2_manifest.json").read_text(encoding="utf-8"))
    val_expected = list(val_manifest["splits"]["val"]["instances_by_class"].values())
    run_dir = args.project.resolve() / args.name
    if run_dir.exists():
        raise FileExistsError(f"Choose a new --name: {run_dir}")
    if args.device != "cpu" and (not args.device.isdecimal() or not torch.cuda.is_available()):
        raise RuntimeError("Use one available CUDA device index or --device cpu")
    gpu = "CPU" if args.device == "cpu" else torch.cuda.get_device_name(int(args.device))
    model = YOLO(str(args.weights.resolve()))
    source_weights = Path(model.ckpt_path).resolve()
    source_hash = file_hash(source_weights)
    if model.task != "obb" or len(model.names) != 15 or model.names.get(0) != "plane":
        raise ValueError("Expected the original DOTA-pretrained Nano OBB checkpoint, not labware weights")
    training_args = {
        "data": str(mixture / "data.yaml"),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": 0,
        "seed": args.train_seed,
        "deterministic": True,
        "resume": False,
        "pretrained": True,
        "patience": 0,
        "project": str(args.project.resolve()),
        "name": args.name,
        "exist_ok": False,
    }
    # Ultralytics writes its label cache beside the first training label, i.e. inside a
    # source dataset.  Put every pre-existing train cache back afterwards and remove the
    # ones this run created, so the source datasets are left exactly as they were.
    caches = {(image.parent.parent.parent / "labels" / "train.cache") for image in images}
    saved = {path: path.read_bytes() if path.is_file() else None for path in caches}
    print(f"Fresh training from {source_weights} on device {args.device} ({gpu})", flush=True)
    print(f"mixture {mixture.name}: {manifest['real_images']} real + {manifest['synthetic_images']} synthetic, "
          f"training seed {args.train_seed}", flush=True)
    try:
        model.train(**training_args)
    finally:
        for path, data in saved.items():
            if data is not None:
                if path.read_bytes() != data:
                    path.write_bytes(data)
            elif path.is_file():
                path.unlink()
    if Path(model.trainer.save_dir).resolve() != run_dir:
        raise ValueError(f"unexpected run directory {model.trainer.save_dir}")
    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.is_file():
        raise FileNotFoundError(best_weights)
    (run_dir / "mixture_manifest.json").write_bytes((mixture / "mixture_manifest.json").read_bytes())
    write_json(run_dir / "training_provenance.json", {
        "initial_weights": str(source_weights),
        "initial_weights_sha256": source_hash,
        "resume": False,
        "gpu": gpu,
        "mixture": mixture.name,
        "real_share_percent": manifest["real_share_percent"],
        "real_images": manifest["real_images"],
        "synthetic_images": manifest["synthetic_images"],
        "mixture_order_seed": manifest["order_seed"],
        "mixture_manifest_sha256": file_hash(mixture / "mixture_manifest.json"),
        "train_list_sha256": file_hash(mixture / "train.txt"),
        "checkpoint_selection": "dataset/real_v2 val (capture 0006)",
        "training_seed": args.train_seed,
        "training_args": training_args,
    })
    best = YOLO(str(best_weights))
    if best.names != CLASS_NAMES:
        raise ValueError("Trained checkpoint has an unexpected class mapping")
    metrics = best.val(
        data=str(mixture / "data.yaml"),
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
    if [int(value) for value in metrics.nt_per_class] != val_expected:
        raise ValueError("Validation instance counts differ from the real_v2 manifest")
    report = {
        "split": "val",
        "dataset": "dataset/real_v2 val (capture 0006)",
        "images": val_manifest["splits"]["val"]["images"],
        "checkpoint": str(best_weights),
        "training_seed": args.train_seed,
        **summarize_metrics(metrics),
    }
    write_json(run_dir / "heldout_metrics.json", report)
    print("\n=== real_v2 validation metrics (best.pt) ===", flush=True)
    print(json.dumps(report["overall"], indent=2), flush=True)
    print(f"Run directory: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
