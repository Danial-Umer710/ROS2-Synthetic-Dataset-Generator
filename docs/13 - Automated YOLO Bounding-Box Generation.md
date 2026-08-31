## Overview

To eliminate manual data labeling and ensure mathematical accuracy for machine learning training, the synthetic pipeline was upgraded to automatically generate ground-truth annotations. For every frame rendered, the server calculates the exact 2D screen coordinates of the target object and exports a `.txt` file alongside the `.png` image using the standard YOLO annotation format.

## Architectural Mechanisms

### 1. 3D-to-2D Mathematical Projection

Blender operates in a 3D coordinate space, while YOLO object detection models require normalized 2D screen coordinates.

- The server uses `bpy_extras.object_utils.world_to_camera_view` to cast rays from the 8 corners of the object's 3D bounding box (`obj.bound_box`) directly onto the 2D camera plane.
    
- **Scene Graph Refresh:** Before calculating coordinates, the script forces a strict scene update via `bpy.context.view_layer.update()`. Without this, Blender caches stale camera transforms, resulting in severely inaccurate bounding boxes when the camera orbits.
    

### 2. Coordinate System Translation & Inversion

Blender and YOLO utilize incompatible 2D coordinate systems, requiring mathematical translation prior to export:

- **Blender Origin:** `(0,0)` is the **bottom-left** of the screen.
    
- **YOLO Origin:** `(0,0)` is the **top-left** of the screen.
    
- **The Fix:** The script manually inverts the Y-axis for the center coordinate (`y_center = 1.0 - raw_y`) to ensure the bounding box correctly maps to the YOLO specification.
    

### 3. Output Formatting (Normalized YOLO Standard)

Coordinates are clamped between `0.0` and `1.0` to ensure bounding boxes do not exceed screen dimensions if the object clips the camera edge. The server synchronously writes a `.txt` file containing the standardized YOLO structure: `[class_id] [x_center] [y_center] [width] [height]`

**Example Output (`0000.txt`):** `0 0.512345 0.487654 0.254321 0.123456`

## 4. Strict Scene Isolation (The Teleportation Method)

Standard Blender API commands (`obj.hide_render = True`) are frequently overridden by master visibility toggles on parent Collections within the `.blend` file. To guarantee a pristine render environment without background artifacts or phantom occlusions, the server script employs a brute-force physical isolation strategy:

*   **Whitelist Filtering:** The script explicitly identifies the target training object (e.g., `Glass_Tube`) and the necessary background plane (`Charuco`).
*   **Z-Axis Translation:** All other non-essential meshes (holders, environmental pillars, base ground planes) are completely removed from the camera's frustum by overriding their Z-coordinate to `-100.0`. This absolutely prevents unwanted objects from interfering with the synthetic data generation.

## 5. Dataset QA & Validation Pipeline

Generating a massive dataset without verifying the coordinate math results in useless training data. Before executing a full production render batch (e.g., 1,500 frames), the pipeline requires a Quality Assurance (QA) micro-batch.

*   **Coordinate Reversion:** A dedicated QA script (`verify_yolo.py`) parses the normalized YOLO coordinates from the generated `.txt` files and translates them back into absolute pixel coordinates using the following logic:
    *   `x_min = int((x_center - width / 2) * image_width)`
    *   `y_min = int((y_center - height / 2) * image_height)`
*   **OpenCV Visual Confirmation:** The script utilizes `cv2.rectangle` to draw the calculated bounding boxes directly over the rendered images. This provides immediate visual confirmation that the 3D-to-2D projection math is flawlessly wrapping the target object across randomized lighting, rotations, and positions before scaling up to thousands of frames.