import os
import cv2
import glob

# Search both train and val image folders
dataset_path = "/mnt/d/Projects/UR_Robotics_Project/YOLO_Dataset/images/**/*.png"
image_files = glob.glob(dataset_path, recursive=True)

print(f"Scanning {len(image_files)} images for corruption...")

bad_count = 0
for img_path in image_files:
    # cv2.imread will silently fail and return None if the PNG is corrupted
    img = cv2.imread(img_path)
    if img is None:
        print(f"Corrupt image found: {img_path}")
        os.remove(img_path)
        
        # Delete the matching label file to prevent YOLO mismatch errors
        label_path = img_path.replace('images', 'labels').replace('.png', '.txt')
        if os.path.exists(label_path):
            os.remove(label_path)
            
        bad_count += 1

print(f"Cleanup complete. Deleted {bad_count} corrupted files.")
