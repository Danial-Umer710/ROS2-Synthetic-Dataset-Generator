import cv2
import os
import glob

# WSL2 paths mapping to your Windows D: drive
RENDER_DIR = "/mnt/d/Projects/UR_Robotics_Project/Test Renders"
OUTPUT_DIR = "/mnt/d/Projects/UR_Robotics_Project/QA_Check"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Grab all PNG files in the render directory
images = glob.glob(os.path.join(RENDER_DIR, "*.png"))

if not images:
    print(f"No images found in {RENDER_DIR}")
    exit()

print(f"Checking {len(images)} images...")

for img_path in images:
    txt_path = img_path.replace(".png", ".txt")
    if not os.path.exists(txt_path):
        print(f"Missing label for {img_path}")
        continue
        
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    
    with open(txt_path, "r") as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            # YOLO format: class x_center y_center width height
            x_c = float(parts[1])
            y_c = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])
            
            # Convert normalized math back to exact pixel coordinates
            x1 = int((x_c - bw / 2) * w)
            y1 = int((y_c - bh / 2) * h)
            x2 = int((x_c + bw / 2) * w)
            y2 = int((y_c + bh / 2) * h)
            
            # Draw a thick red box around the object
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
    out_path = os.path.join(OUTPUT_DIR, os.path.basename(img_path))
    cv2.imwrite(out_path, img)

print(f"DONE. Check the '{OUTPUT_DIR}' folder in Windows.")
