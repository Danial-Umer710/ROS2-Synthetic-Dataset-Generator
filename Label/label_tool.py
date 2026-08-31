import cv2
import glob
import os

image_files = glob.glob("*.jpeg") + glob.glob("*.png")
class_id = 0

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            # Made the dots and lines thicker so you can see them on huge photos
            cv2.circle(img_display, (x, y), 8, (0, 255, 0), -1) 
            
            if len(points) > 1:
                cv2.line(img_display, points[-2], points[-1], (0, 255, 0), 4)
            if len(points) == 4:
                cv2.line(img_display, points[3], points[0], (0, 255, 0), 4)
                
            cv2.imshow("Labeler", img_display)

print("--- OBB LABELING TOOL STARTED ---")
print("Click the 4 corners sequentially. Press ANY KEY to save and next.")

# 1. FIX: Unlock the OpenCV window so it can be resized
cv2.namedWindow("Labeler", cv2.WINDOW_NORMAL)
# 2. FIX: Force the initial window size to fit comfortably on a laptop screen
cv2.resizeWindow("Labeler", 1280, 720) 

for img_path in image_files:
    points = []
    img = cv2.imread(img_path)
    img_display = img.copy()
    h, w = img.shape[:2]
    
    cv2.imshow("Labeler", img_display)
    cv2.setMouseCallback("Labeler", click_event)
    
    cv2.waitKey(0) 
    
    if len(points) == 4:
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        with open(txt_path, "w") as f:
            line = f"{class_id}"
            for p in points:
                norm_x = max(0.0, min(1.0, p[0] / w))
                norm_y = max(0.0, min(1.0, p[1] / h))
                line += f" {norm_x:.6f} {norm_y:.6f}"
            f.write(line + "\n")
        print(f"Success: Saved {txt_path}")
    else:
        print(f"Skipped {img_path}: You must click exactly 4 points.")

cv2.destroyAllWindows()
print("--- LABELING COMPLETE ---")