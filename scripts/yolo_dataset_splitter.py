import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import cv2
import yaml

# --- CONFIGURATION ---
# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLIT_COUNTS = {"train": 800, "val": 100, "test": 100}
CLASS_NAMES = {0: "Tube_Holder", 1: "Glass_Tube", 2: "Slide_Holder", 3: "Eppendorf_Rack"}


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def inventory(root):
    paths = {}
    for kind, suffix in (("images", ".png"), ("labels", ".txt")):
        found = {}
        for split in SPLIT_COUNTS:
            folder = root / kind / split
            for path in sorted(folder.iterdir()) if folder.exists() else []:
                if not path.is_file() or path.suffix.lower() != suffix:
                    raise ValueError(f"Unexpected entry in {kind}/{split}: {path}")
                if path.stem in found:
                    raise ValueError(f"Duplicate {kind} stem across splits: {path.stem}")
                found[path.stem] = path
        paths[kind] = found
    if set(paths["images"]) != set(paths["labels"]):
        raise ValueError("Image and label stems do not match exactly")
    if len(paths["images"]) != sum(SPLIT_COUNTS.values()):
        raise ValueError(f"Expected 1000 image/label pairs, found {len(paths['images'])}")
    return paths


def inspect_pairs(paths):
    records, pixel_hashes = {}, set()
    for stem, image_path in sorted(paths["images"].items()):
        image = cv2.imread(str(image_path))
        if image is None or image.shape != (640, 640, 3):
            raise ValueError(f"Unreadable image or unexpected dimensions: {image_path}")
        pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
        if pixel_hash in pixel_hashes:
            raise ValueError(f"Duplicate decoded image content: {image_path}")
        pixel_hashes.add(pixel_hash)
        label_path = paths["labels"][stem]
        instances = [0] * len(CLASS_NAMES)
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) != 9:
                raise ValueError(f"Expected class ID and eight OBB coordinates: {label_path}")
            class_id = int(fields[0])
            coordinates = [float(value) for value in fields[1:]]
            if class_id not in CLASS_NAMES or any(
                not math.isfinite(value) or not 0 <= value <= 1 for value in coordinates
            ):
                raise ValueError(f"Invalid class ID or normalized coordinates: {label_path}")
            points = list(zip(coordinates[::2], coordinates[1::2]))
            area = abs(sum(
                points[i][0] * points[(i + 1) % 4][1]
                - points[(i + 1) % 4][0] * points[i][1]
                for i in range(4)
            )) / 2
            if area <= 0:
                raise ValueError(f"Degenerate OBB: {label_path}")
            instances[class_id] += 1
        if not sum(instances):
            raise ValueError(f"Empty label file: {label_path}")
        records[stem] = {
            "image": image_path.name,
            "image_sha256": file_hash(image_path),
            "pixels_sha256": pixel_hash,
            "label_sha256": file_hash(label_path),
            "instances": instances,
        }
    return records


def inspect_dataset(root, seed=42, require_split=False):
    root = Path(root).resolve()
    yaml_text = (root / "data.yaml").read_text(encoding="utf-8")
    config = yaml.safe_load(yaml_text)
    if config.get("names") != CLASS_NAMES:
        raise ValueError("Unexpected class mapping in data.yaml")
    paths = inventory(root)
    records = inspect_pairs(paths)
    manifest_path = root / "split_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["seed"] != seed or manifest["counts"] != SPLIT_COUNTS:
            raise ValueError("Existing split has a different seed or size; refusing to reshuffle")
        if manifest["files"] != records:
            raise ValueError("Images or labels differ from the saved split manifest")
    else:
        if require_split:
            raise ValueError("No split manifest; run the splitter with --apply first")
        if any(path.parent.name != "train" for found in paths.values() for path in found.values()):
            raise ValueError("Found held-out files without a split manifest; refusing to overwrite")
        stems = sorted(records)
        random.Random(seed).shuffle(stems)
        manifest = {
            "schema_version": 1,
            "seed": seed,
            "counts": SPLIT_COUNTS,
            "data_yaml_before_split": yaml_text,
            "splits": {"train": stems[:800], "val": stems[800:900], "test": stems[900:]},
            "files": records,
        }
    splits = manifest["splits"]
    if set(splits) != set(SPLIT_COUNTS):
        raise ValueError("Unexpected splits in manifest")
    assigned = [stem for stems in splits.values() for stem in stems]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(records):
        raise ValueError("Split assignments overlap or do not cover the complete dataset")
    summary = {}
    for split, expected in SPLIT_COUNTS.items():
        if len(splits[split]) != expected:
            raise ValueError(f"Incorrect {split} size")
        counts = [sum(records[stem]["instances"][i] for stem in splits[split]) for i in CLASS_NAMES]
        if not all(counts):
            raise ValueError(f"A class is missing from the {split} split")
        for stem in splits[split]:
            for found in paths.values():
                if found[stem].parent.name not in {"train", split}:
                    raise ValueError(f"File in an unexpected split: {found[stem]}")
                if require_split and found[stem].parent.name != split:
                    raise ValueError(f"Split is incomplete: {found[stem]}")
        summary[split] = {"images": expected, "labels": expected, "instances_by_class": counts}
    if require_split:
        configured_root = Path(config.get("path") or root).expanduser()
        if not configured_root.is_absolute() or configured_root.resolve() != root:
            raise ValueError("data.yaml must omit path or reference this dataset with an absolute path")
        if any(config.get(split) != f"images/{split}" for split in SPLIT_COUNTS):
            raise ValueError("data.yaml does not reference three separate splits")
    return manifest, paths, config, summary


def apply_split(root, manifest, paths, config):
    manifest_path = root / "split_manifest.json"
    if not manifest_path.exists():
        with manifest_path.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
    for split, stems in manifest["splits"].items():
        for kind in ("images", "labels"):
            destination = root / kind / split
            destination.mkdir(parents=True, exist_ok=True)
            for stem in stems:
                source = paths[kind][stem]
                target = destination / source.name
                if source != target:
                    if target.exists():
                        raise FileExistsError(f"Refusing to overwrite {target}")
                    source.rename(target)
    config.pop("path", None)
    config.update({split: f"images/{split}" for split in SPLIT_COUNTS})
    yaml_path = root / "data.yaml"
    text = yaml.safe_dump(config, sort_keys=False)
    if yaml_path.read_text(encoding="utf-8") != text:
        temporary = root / "data.split.yaml.tmp"
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(text)
        temporary.replace(yaml_path)


def main():
    parser = argparse.ArgumentParser(description="Reproducible 800/100/100 split of the generated STL dataset.")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "dataset")
    parser.add_argument("--seed", type=int, default=42)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="Move paired files and update data.yaml; no files are deleted.")
    mode.add_argument("--check", action="store_true", help="Require a complete, unchanged split and correct data.yaml.")
    args = parser.parse_args()
    root = args.dataset.resolve()
    manifest, paths, config, summary = inspect_dataset(root, args.seed, require_split=args.check)
    if args.apply:
        apply_split(root, manifest, paths, config)
        manifest, _, _, summary = inspect_dataset(root, args.seed, require_split=True)
    print(json.dumps({
        "mode": "applied" if args.apply else "verified" if args.check else "dry_run",
        "seed": manifest["seed"],
        "splits": summary,
        "overlapping_stems": 0,
        "duplicate_pixel_hashes": 0,
        "manifest": str(root / "split_manifest.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
