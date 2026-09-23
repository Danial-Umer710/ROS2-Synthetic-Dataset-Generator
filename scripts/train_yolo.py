from ultralytics import YOLO

def main():
    # Load the YOLOv8 Nano OBB model
    model = YOLO("yolov8n-obb.pt")

    # Train the model on the full 4,000-image dataset
    results = model.train(
        data=r'D:\Projects\UR_Robotics_Project\YOLO_Main_Dataset\data.yaml',
        epochs=50,       # Increased to 50 for the final training run
        imgsz=640,
        batch=16,
        device=0,        # Uses your RTX GPU
        workers=0,       # Kept at 0 to prevent Windows multiprocessing freezes
        project='UR16e_Labware_Detection',
        name='ur16e_main_run_4000' # New folder name for the final results
    )

if __name__ == '__main__':
    main()