import os
import shutil
import random

# --- CONFIGURATION ---
# --- CONFIGURATION ---
SOURCE_DIR = r"D:\Projects\UR_Robotics_Project\Blender Dataset\Main_Batch_4000"
TARGET_DIR = r"D:\Projects\UR_Robotics_Project\YOLO_Main_Dataset"

print("Creating YOLO directory structure...")
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(TARGET_DIR, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(TARGET_DIR, 'labels', split), exist_ok=True)

print("Scanning source directory...")
all_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.jpg')]
valid_files = []
for f in all_files:
    base_name = os.path.splitext(f)[0]
    if os.path.exists(os.path.join(SOURCE_DIR, base_name + '.txt')):
        valid_files.append(base_name)

print(f"Found {len(valid_files)} valid image/label pairs.")
random.seed(42) 
random.shuffle(valid_files)

total = len(valid_files)
train_end = int(total * 0.8)
val_end = int(total * 0.9)

train_files = valid_files[:train_end]
val_files = valid_files[train_end:val_end]
test_files = valid_files[val_end:]

def copy_to_split(file_list, split_name):
    for base_name in file_list:
        shutil.copy2(os.path.join(SOURCE_DIR, base_name + '.jpg'), os.path.join(TARGET_DIR, 'images', split_name, base_name + '.jpg'))
        shutil.copy2(os.path.join(SOURCE_DIR, base_name + '.txt'), os.path.join(TARGET_DIR, 'labels', split_name, base_name + '.txt'))

print(f"Copying {len(train_files)} files to Train...")
copy_to_split(train_files, 'train')

print(f"Copying {len(val_files)} files to Validation...")
copy_to_split(val_files, 'val')

print(f"Copying {len(test_files)} files to Test...")
copy_to_split(test_files, 'test')

print("Generating data.yaml configuration...")
yaml_content = f"""path: {TARGET_DIR.replace('\\', '/')}
train: images/train
val: images/val
test: images/test

names:
  0: Glass_Slide
  1: Glass_Tube
  2: Slide_Holder
  3: Tube_Holder
"""
with open(os.path.join(TARGET_DIR, 'data.yaml'), 'w') as f:
    f.write(yaml_content)

print("\nSuccess! Dataset is ready for YOLOv8 training.")