import os
import shutil
import glob

# Paths
REAL_IMG_DIR = r"D:\Projects\UR_Robotics_Project\Label"
SYNTH_DIR = r"D:\Projects\UR_Robotics_Project\Blender Dataset\Glass Slide"
YOLO_IMG_TRAIN = r"D:\Projects\UR_Robotics_Project\YOLO_Dataset\images\train"
YOLO_LBL_TRAIN = r"D:\Projects\UR_Robotics_Project\YOLO_Dataset\labels\train"

# Clean destination folders before copying
os.makedirs(YOLO_IMG_TRAIN, exist_ok=True)
os.makedirs(YOLO_LBL_TRAIN, exist_ok=True)
for f in glob.glob(os.path.join(YOLO_IMG_TRAIN, "*")):
    os.remove(f)
for f in glob.glob(os.path.join(YOLO_LBL_TRAIN, "*")):
    os.remove(f)

# Helper function to grab all image formats
def get_images(folder):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return list(set(files))

# 1. Oversample Real Images (8 images * 30 copies = 240 samples)
real_images = get_images(REAL_IMG_DIR)
print(f"Found {len(real_images)} real images. Replicating 30x...")

for img_path in real_images:
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    txt_path = os.path.join(REAL_IMG_DIR, base_name + ".txt")
    ext = os.path.splitext(img_path)[1]
    
    if not os.path.exists(txt_path):
        print(f"Warning: Missing label for {img_path}, skipping.")
        continue
        
    for copy_idx in range(30):
        new_img_name = f"real_{base_name}_copy{copy_idx}{ext}"
        new_txt_name = f"real_{base_name}_copy{copy_idx}.txt"
        
        shutil.copy(img_path, os.path.join(YOLO_IMG_TRAIN, new_img_name))
        shutil.copy(txt_path, os.path.join(YOLO_LBL_TRAIN, new_txt_name))

# 2. Add 200 Synthetic Images
synth_images = get_images(SYNTH_DIR)[:200]
print(f"Found and copying {len(synth_images)} synthetic images...")

for img_path in synth_images:
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    txt_path = os.path.join(SYNTH_DIR, base_name + ".txt")
    
    if os.path.exists(txt_path):
        shutil.copy(img_path, os.path.join(YOLO_IMG_TRAIN, os.path.basename(img_path)))
        shutil.copy(txt_path, os.path.join(YOLO_LBL_TRAIN, os.path.basename(txt_path)))

print("--- REBALANCE COMPLETE: Dataset populated in YOLO_Dataset ---")