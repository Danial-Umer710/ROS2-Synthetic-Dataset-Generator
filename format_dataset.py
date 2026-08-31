import os
import shutil
import random

# Source and Destination Paths
SOURCE_DIR = "/mnt/d/Projects/UR_Robotics_Project/Test Renders"
DATASET_DIR = "/mnt/d/Projects/UR_Robotics_Project/YOLO_Dataset"

# 1. Build YOLO Directory Structure
splits = ['train', 'val']
for split in splits:
    os.makedirs(os.path.join(DATASET_DIR, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, 'labels', split), exist_ok=True)

# 2. Gather and Shuffle Data
images = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.png')]
random.shuffle(images) # Randomize to ensure a healthy mix in both sets

# 3. Calculate 80/20 Split
split_index = int(len(images) * 0.8)
train_images = images[:split_index]
val_images = images[split_index:]

def process_files(file_list, split_name):
    print(f"Moving {len(file_list)} files to {split_name}...")
    for img_file in file_list:
        txt_file = img_file.replace('.png', '.txt')
        
        src_img = os.path.join(SOURCE_DIR, img_file)
        src_txt = os.path.join(SOURCE_DIR, txt_file)
        
        # Move image
        shutil.copy(src_img, os.path.join(DATASET_DIR, 'images', split_name, img_file))
        
        # Move label if it exists
        if os.path.exists(src_txt):
            shutil.copy(src_txt, os.path.join(DATASET_DIR, 'labels', split_name, txt_file))

# Execute the split
process_files(train_images, 'train')
process_files(val_images, 'val')

# 4. Generate data.yaml Configuration
yaml_content = f"""path: {DATASET_DIR}
train: images/train
val: images/val

# Classes
names:
  0: Glass_Tube
"""

yaml_path = os.path.join(DATASET_DIR, 'data.yaml')
with open(yaml_path, 'w') as f:
    f.write(yaml_content)

print(f"\nDataset successfully built at: {DATASET_DIR}")
print(f"data.yaml created successfully.")
