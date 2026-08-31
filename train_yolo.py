from ultralytics import YOLO

def main():
    # Load a lightweight pre-trained YOLOv8 baseline model
    model = YOLO("yolov8n.pt")

    # Execute training using the custom dataset configuration
    results = model.train(
        data="/home/dani/YOLO_Dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,  # Automatically detects your GPU (CUDA) or falls back to CPU
        workers=0
    )

    print("Training complete! Model weights saved.")

if __name__ == '__main__':
    main()
