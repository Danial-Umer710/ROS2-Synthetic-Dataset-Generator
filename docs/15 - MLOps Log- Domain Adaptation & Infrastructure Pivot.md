### MLOps Log: Domain Adaptation & Infrastructure Pivot

**The Problem:**

Attempting to fine-tune the synthetic (Blender-trained) weights on 11 real-world images resulted in catastrophic failure. The model suffered from a severe domain gap, causing NMS time limit panics and a 1% mAP50 score. Simultaneously, the local WSL2/Windows bridge caused severe VRAM gridlock, extending 1-second epochs to 35 minutes.

**The Solution (DevOps Pivot):**

1. **Infrastructure Migration:** Abandoned local WSL2 training. Packaged the dataset and migrated the workload to a Google Colab T4 GPU container to bypass Windows filesystem bottlenecks.
    
2. **Model Reset (Pure Overfit Test):** Discarded the synthetic Blender weights entirely. Initialized a factory-reset `yolov8n.pt` base model and trained it exclusively on the 11 real-world images with zero hyperparameter constraints.
    

**The Results:**

- **Training Metrics:** The cloud infrastructure processed 50 epochs in ~40 seconds. The fresh model achieved **99.5% mAP50** and 100% precision.
    
- **Edge Deployment:** Weights (`best.pt`) were pulled back to the local WSL2 environment for local CPU inference.
    
- **Visual Validation:** Ran inference on `test.jpeg`. The model successfully ignored background clutter and transparent glare, isolating the target tube with **94% confidence**.
    

**Calculated Center Coordinates (Pixels):**

- `Center X`: 612.79
    
- `Center Y`: 828.76
    

**Next Steps:**

Develop the mathematical transformation matrix to map the camera's 2D pixel coordinates $(u, v)$ to the Universal Robot's 3D physical workspace coordinates $(X, Y)$ in millimeters.