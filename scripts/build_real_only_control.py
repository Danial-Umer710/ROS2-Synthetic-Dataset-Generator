"""
build_real_only_control.py
==========================
Build the real-only control for the 20 percent mixture: the SAME 160 real images
as ``mixed_datasets/real20`` with no synthetic images.

Each image is listed ``800 / 160 = 5`` times, so a 30-epoch run takes the same
number of training steps as every 800-image mixture; only the absence of the 640
synthetic images differs from ``real20``.  Mosaic and the other train-time
augmentations make the repeated entries different samples in practice.

    python scripts/build_real_only_control.py   # writes mixed_datasets/real20_only/
"""

import json
from pathlib import Path

from yolo_dataset_splitter import PROJECT_ROOT

TOTAL = 800
SOURCE = PROJECT_ROOT / "mixed_datasets" / "real20"
OUT = PROJECT_ROOT / "mixed_datasets" / "real20_only"


def main():
    source = json.loads((SOURCE / "mixture_manifest.json").read_text(encoding="utf-8"))
    real = [e for e in source["images"] if e["source"] == "real"]
    if len(real) != source["real_images"] or TOTAL % len(real):
        raise ValueError(f"unexpected real image count {len(real)}")
    repeat = TOTAL // len(real)
    entries = real * repeat
    OUT.mkdir(exist_ok=False)
    (OUT / "train.txt").write_text("".join(f"./../../{e['image']}\n" for e in entries),
                                   encoding="utf-8", newline="\n")
    yaml = (SOURCE / "data.yaml").read_text(encoding="utf-8").splitlines()
    yaml[0] = f"# Real-only control: the {len(real)} real images of real20, each listed {repeat} times, no synthetic."
    (OUT / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8", newline="\n")
    manifest = {
        "mixture": OUT.name,
        "control_for": SOURCE.name,
        "real_share_percent": 100,
        "total_images": TOTAL,
        "distinct_images": len(real),
        "repeat": repeat,
        "real_images": TOTAL,
        "synthetic_images": 0,
        "order_seed": source["order_seed"],
        "real_images_per_capture": source["real_images_per_capture"],
        "sources": {"real": source["sources"]["real"]},
        "val": source["val"],
        "test": source["test"],
        "class_names": source["class_names"],
        "images": entries,
    }
    with (OUT / "mixture_manifest.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(f"{OUT.name}: {len(real)} distinct real images x {repeat} = {len(entries)} entries")


if __name__ == "__main__":
    main()
