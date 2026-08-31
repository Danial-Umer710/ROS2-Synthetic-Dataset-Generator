# 14. Local YOLO Training & Environment Optimization

## 1. Dataset Structuring & Standard Formatting
The raw synthetic output (1,500 rendered `.png` images and matching `.txt` YOLO bounding box files) was automatically structured using a custom Python script (`format_dataset.py`) to adhere to Ultralytics YOLO dataset specifications:
* **Train/Val Split:** 80% (1,200 images) allocated for training, 20% (300 images) allocated for validation.
* **Directory Topology:** Separated into `images/train`, `images/val`, `labels/train`, and `labels/val`.
* **Config Generation:** Automated creation of `data.yaml` mapping relative paths and class definitions (`0: Glass_Tube`).

## 2. Infrastructure & WSL2 I/O Optimization
Executing machine learning workloads across the WSL2 Windows-to-Linux bridge (`/mnt/d/`) created significant filesystem translation bottlenecks (9P protocol overhead), resulting in kernel memory exhaustion (`OSError: [Errno 12] Cannot allocate memory`).

To establish a zero-bottleneck MLOps pipeline, the infrastructure was restructured:
* **Native Storage Migration:** The dataset and execution scripts were transferred directly to native Linux ext4 storage (`~/YOLO_Dataset`), bypassing cross-drive I/O operations entirely.
* **Virtual Environment Isolation:** A dedicated Python virtual environment (`yolo_env`) was established on the native Linux drive to maintain clean dependency trees for `ultralytics` and `torch`.

## 3. Model Training Execution
Model training was initiated using `yolov8n.pt` (YOLOv8 Nano) as a baseline pretrained architecture, utilizing hardware acceleration via an NVIDIA GeForce RTX 4050 Laptop GPU.

* **Stability Overrides:**
  * `workers=0`: Set dataloader subprocess workers to zero to eliminate multi-process memory leaks across virtualized hardware bridges.
  * `amp=False`: Disabled Automatic Mixed Precision checks to avoid WSL2 GPU synchronization locks during non-maximum suppression (NMS) evaluations.
* **Training Parameters:** 50 epochs, batch size of 16, image resolution scaled to 640x640 pixels.