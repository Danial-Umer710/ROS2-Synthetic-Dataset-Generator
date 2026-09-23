from ultralytics import YOLO

# Load your newly trained custom weights
model = YOLO(r"D:\Projects\UR_Robotics_Project\runs\obb\train\weights\best.pt")

# Run prediction on the isolated test dataset
results = model.predict(
    source=r"D:\Projects\UR_Robotics_Project\YOLO_Dataset\images\test",
    save=True,   # Tells YOLO to save images with the predicted boxes drawn on them
    imgsz=640,
    conf=0.7     # Only accept predictions the AI is 70% or more confident about
)

print("Testing complete. Check the new runs folder for the visual outputs!")