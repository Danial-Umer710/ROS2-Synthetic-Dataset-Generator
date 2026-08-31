from ultralytics import YOLO
import os
import math

# 1. Load the rebalanced model (production-7)
model_path = r"D:\Projects\UR_Robotics_Project\models\glass_slide_production-7\weights\best.pt"
model = YOLO(model_path)

# 2. Point to your physical test photograph
img_path = r"D:\Projects\UR_Robotics_Project\google_test.jpeg" # Update this to your exact test image name

if not os.path.exists(img_path):
    raise FileNotFoundError(f"FATAL: Image does not exist at {img_path}")

print("--- RUNNING PURE AI OBB INFERENCE ---")

# 3. Run prediction with standard 25% confidence threshold
results = model(img_path, save=True, show=True)

for r in results:
    obb = r.obb
    if obb is not None and len(obb) > 0:
        for i in range(len(obb.xyxyxyxy)):
            angle_rad = obb.xywhr[i][4].item() 
            angle_deg = math.degrees(angle_rad)
            print(f"Target Acquired! Gripper Rotation Angle: {angle_deg:.2f} degrees")
    else:
        print("Model failed to detect any slides.")