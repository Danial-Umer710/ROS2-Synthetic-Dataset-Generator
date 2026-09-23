import cv2
import math
from ultralytics import YOLO

# 1. Load your production model
model_path = r"D:\Projects\UR_Robotics_Project\models\glass_slide_production-7\weights\best.pt"
model = YOLO(model_path)

# 2. Open camera (0 is default webcam; change to 1 if using an external USB camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not access camera.")

print("--- LIVE VISION ACTIVE (Press 'q' to quit) ---")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run OBB prediction with 0.4 confidence to filter out noise
    results = model(frame, conf=0.8, verbose=False)

    for r in results:
        obb = r.obb
        if obb is not None and len(obb) > 0:
            for i in range(len(obb.xyxyxyxy)):
                # Extract orientation angle in degrees
                angle_rad = obb.xywhr[i][4].item()
                angle_deg = math.degrees(angle_rad)

                # Get the 4 corner points
                pts = obb.xyxyxyxy[i].cpu().numpy().astype(int)
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

                # Display angle on screen
                center_x, center_y = int(obb.xywhr[i][0].item()), int(obb.xywhr[i][1].item())
                cv2.putText(
                    frame,
                    f"Angle: {angle_deg:.1f} deg",
                    (center_x - 40, center_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

    cv2.imshow("Robot Vision - Live Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()