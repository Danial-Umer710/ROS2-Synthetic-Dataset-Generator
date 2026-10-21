import argparse
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from yolo_dataset_splitter import PROJECT_ROOT, file_hash, inspect_dataset

BASELINE_RUN = PROJECT_ROOT / "runs" / "obb" / "UR16e_Labware_Detection" / "stl_synthetic_800_30ep_holdout"
PUBLISHED = PROJECT_ROOT / "results" / "baseline"
STL_FILES = ("TubeHolder.stl", "Tube.stl", "Slide_holder.stl", "eppendorf_tarto.stl")
OUTPUT_SUFFIXES = {".png", ".jpg", ".jpeg", ".json", ".yaml", ".csv"}


def summarize_files(paths):
    files = sorted(paths)
    if not files or any(not path.is_file() for path in files):
        raise ValueError("A required publication asset is missing")
    largest = max(files, key=lambda path: path.stat().st_size)
    return {
        "files": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "largest_file": str(largest.relative_to(PROJECT_ROOT)),
        "largest_file_bytes": largest.stat().st_size,
    }


def audit():
    dataset = PROJECT_ROOT / "dataset"
    manifest, paths, _, split_summary = inspect_dataset(dataset, 42, require_split=True)
    stl_dir = PROJECT_ROOT / "stl_models"
    if not stl_dir.is_dir():
        stl_dir = PROJECT_ROOT / "Data Sets" / "STL dataset"
    weights_dir = PROJECT_ROOT / "weights" if (PROJECT_ROOT / "weights").is_dir() else BASELINE_RUN / "weights"
    results_dir = PUBLISHED if PUBLISHED.is_dir() else BASELINE_RUN
    preview_dir = PUBLISHED / "test" / "previews"
    if not preview_dir.is_dir():
        preview_dir = PROJECT_ROOT / f"test_predictions_preview_{BASELINE_RUN.name}"
    groups = {
        "dataset": [*paths["images"].values(), *paths["labels"].values(),
                    dataset / "data.yaml", dataset / "split_manifest.json"],
        "stl_models": [stl_dir / name for name in STL_FILES],
        "trained_weights": [weights_dir / name for name in ("best.pt", "last.pt")],
        "training_and_evaluation_outputs": [
            path for path in results_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in OUTPUT_SUFFIXES and preview_dir not in path.parents
        ],
        "test_previews": list(preview_dir.glob("*.png")),
    }
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=PROJECT_ROOT, check=True,
                             capture_output=True).stdout.decode("utf-8").split("\0")
    return {
        "split_seed": manifest["seed"],
        "splits": split_summary,
        "assets": {name: summarize_files(files) for name, files in groups.items()},
        "total_asset_bytes": sum(path.stat().st_size for files in groups.values() for path in files),
        "tracked_paths_by_top_level": dict(sorted(Counter(
            path.split("/")[0] if "/" in path else "<root>" for path in tracked if path
        ).items())),
    }


def copy_asset(source, destination):
    digest = file_hash(source)
    if destination.exists():
        if file_hash(destination) != digest:
            raise FileExistsError(f"Refusing to overwrite a different publication asset: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    if file_hash(destination) != digest:
        raise ValueError(f"Copy checksum mismatch: {destination}")
    return {
        "path": destination.relative_to(PROJECT_ROOT).as_posix(),
        "source": source.relative_to(PROJECT_ROOT).as_posix(),
        "bytes": destination.stat().st_size,
        "sha256": digest,
    }


def copy_assets():
    dataset = PROJECT_ROOT / "dataset"
    manifest, _, _, split_summary = inspect_dataset(dataset, 42, require_split=True)
    predictions = json.loads((BASELINE_RUN / "test_predictions.json").read_text(encoding="utf-8"))
    test_metrics = json.loads((BASELINE_RUN / "heldout_test" / "metrics.json").read_text(encoding="utf-8"))
    if test_metrics["checkpoint_sha256"] != file_hash(BASELINE_RUN / "weights" / "best.pt"):
        raise ValueError("Test metrics do not describe the checkpoint being published")
    if test_metrics["split_manifest_sha256"] != file_hash(dataset / "split_manifest.json"):
        raise ValueError("Test metrics do not describe the current frozen dataset")
    if predictions["images"] != 100 or len(predictions["predictions"]) != 100:
        raise ValueError("Missing test predictions")
    if {Path(row["image"]).stem for row in predictions["predictions"]} != set(manifest["splits"]["test"]):
        raise ValueError("Predictions do not cover the recorded test split")
    if len(predictions["preview_images"]) != 10 or len(set(predictions["preview_images"])) != 10:
        raise ValueError("Expected 10 distinct preview images")
    mapping = []
    for name in STL_FILES:
        mapping.append((PROJECT_ROOT / "Data Sets" / "STL dataset" / name, PROJECT_ROOT / "stl_models" / name))
    for name in ("best.pt", "last.pt"):
        mapping.append((BASELINE_RUN / "weights" / name, PROJECT_ROOT / "weights" / name))
    for source in sorted(BASELINE_RUN.iterdir()):
        if source.is_file() and source.suffix.lower() in OUTPUT_SUFFIXES and source.name not in {
            "heldout_metrics.json", "test_predictions.json"
        }:
            mapping.append((source, PUBLISHED / "training" / source.name))
    mapping.append((BASELINE_RUN / "heldout_metrics.json", PUBLISHED / "validation" / "metrics.json"))
    for source_dir, target_dir in (("heldout_val", "validation"), ("heldout_test", "test")):
        for source in sorted((BASELINE_RUN / source_dir).iterdir()):
            if source.is_file() and source.suffix.lower() in OUTPUT_SUFFIXES:
                mapping.append((source, PUBLISHED / target_dir / source.name))
    mapping.append((BASELINE_RUN / "test_predictions.json", PUBLISHED / "test" / "predictions.json"))
    for name in predictions["preview_images"]:
        source = PROJECT_ROOT / f"test_predictions_preview_{BASELINE_RUN.name}" / name
        mapping.append((source, PUBLISHED / "test" / "previews" / name))
    if len({destination for _, destination in mapping}) != len(mapping):
        raise ValueError("Publication destinations overlap")
    records = [copy_asset(source, destination) for source, destination in mapping]
    report = {
        "baseline": BASELINE_RUN.name,
        "split_seed": 42,
        "splits": split_summary,
        "dataset_config_sha256": file_hash(dataset / "data.yaml"),
        "dataset_manifest_sha256": file_hash(dataset / "split_manifest.json"),
        "dataset_integrity": "Image, decoded-pixel and label SHA256 values are stored in dataset/split_manifest.json.",
        "artifacts": records,
    }
    path = PUBLISHED / "publication_manifest.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != report:
            raise FileExistsError("Publication manifest changed; preserve the previous snapshot")
    else:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
    return {"copied_or_verified_assets": len(records), "manifest": str(path)}


def verify_index():
    manifest = json.loads((PROJECT_ROOT / "dataset" / "split_manifest.json").read_text(encoding="utf-8"))
    publication = json.loads((PUBLISHED / "publication_manifest.json").read_text(encoding="utf-8"))
    hashes = {row["path"]: row["sha256"] for row in publication["artifacts"]}
    for split, stems in manifest["splits"].items():
        for stem in stems:
            record = manifest["files"][stem]
            hashes[f"dataset/images/{split}/{record['image']}"] = record["image_sha256"]
            hashes[f"dataset/labels/{split}/{stem}.txt"] = record["label_sha256"]
    hashes["dataset/data.yaml"] = publication["dataset_config_sha256"]
    hashes["dataset/split_manifest.json"] = publication["dataset_manifest_sha256"]
    hashes["results/baseline/publication_manifest.json"] = file_hash(PUBLISHED / "publication_manifest.json")
    provenance = json.loads((PUBLISHED / "training" / "training_provenance.json").read_text(encoding="utf-8"))
    hashes["yolov8n-obb.pt"] = provenance["initial_weights_sha256"]
    core = {
        ".gitignore", ".gitattributes", "README.md", "AGENTS.md", "requirements.txt",
        "blender/generate_synthetic_dataset.py", "scripts/train_yolo.py", "scripts/yolo_dataset_splitter.py",
        "scripts/evaluate_yolo.py", "scripts/prepare_publication.py", "scripts/visualize_obb_labels.py",
    }
    def git_paths(*args):
        result = subprocess.run(["git", *args, "-z"], cwd=PROJECT_ROOT, check=True, capture_output=True)
        return set(filter(None, result.stdout.decode("utf-8").split("\0")))
    index = git_paths("ls-files")
    staged = git_paths("diff", "--cached", "--name-only")
    allowed = set(hashes) | core
    if not allowed <= index:
        raise ValueError(f"Publication files missing from Git index: {sorted(allowed - index)}")
    if index - allowed:
        raise ValueError(f"Non-baseline files still tracked in Git index: {sorted(index - allowed)[:20]}")
    total_bytes = 0
    process = subprocess.Popen(["git", "cat-file", "--batch"], cwd=PROJECT_ROOT,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for path, expected_hash in sorted(hashes.items()):
            process.stdin.write(f":{path}\n".encode("utf-8"))
            process.stdin.flush()
            header = process.stdout.readline().decode("utf-8").strip().split()
            if len(header) != 3 or header[1] != "blob":
                raise ValueError(f"Missing Git blob: {path}: {header}")
            remaining = size = int(header[2])
            if size >= 100 * 1024 * 1024:
                raise ValueError(f"Asset exceeds GitHub's normal file limit: {path}")
            digest = hashlib.sha256()
            while remaining:
                chunk = process.stdout.read(min(remaining, 1024 * 1024))
                if not chunk:
                    raise ValueError(f"Truncated Git blob: {path}")
                digest.update(chunk)
                remaining -= len(chunk)
            if process.stdout.read(1) != b"\n" or digest.hexdigest() != expected_hash:
                raise ValueError(f"Git index bytes do not match the frozen asset: {path}")
            total_bytes += size
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError(process.stderr.read().decode("utf-8", errors="replace"))
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()
    return {"verified_git_blobs": len(hashes), "verified_bytes": total_bytes,
            "tracked_files": len(index), "staged_files": len(staged), "extra_tracked_files": len(index - allowed)}


def main():
    parser = argparse.ArgumentParser(description="Audit or copy the frozen STL baseline into publishable asset folders.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--copy-assets", action="store_true", help="Copy local baseline outputs without changing originals.")
    mode.add_argument("--verify-index", action="store_true", help="Verify Git-index asset hashes and staged scope before publishing.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify_index() if args.verify_index else copy_assets() if args.copy_assets else audit()
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(text)
    print(text)


if __name__ == "__main__":
    main()
