from ultralytics import YOLO

def main():
    print("--- INITIALIZING YOLO PIPELINE ---")
    
    # Load a blank, lightweight YOLOv8 model for rapid prototyping
    model = YOLO("yolov8n-obb.pt")
    
    # Execute the training loop
    results = model.train(
        data="D:/Projects/UR_Robotics_Project/YOLO_Dataset/data.yaml",
        epochs=100,               # Scaled up for deep learning
        imgsz=640,
        device=0,
        project="D:/Projects/UR_Robotics_Project/models",
        name="glass_slide_production",  # Renamed for the final run
        workers=2
    )
    
    print("--- TRAINING COMPLETE ---")

if __name__ == '__main__':
    # Windows requires this block to safely utilize multithreading
    main()