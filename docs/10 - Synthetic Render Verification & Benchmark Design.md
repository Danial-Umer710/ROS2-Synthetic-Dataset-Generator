
# 10 - Synthetic Render Verification & Benchmarking

## 1. Overview & Verification Scope
This stage validates that the composed Blender 4.3 digital twin (`Models_Calibration_Base.blend`) correctly generates both **synthetic instance segmentation masks** for AI model training and **extrinsic optical calibration frames** for Universal Robots (UR) camera alignment.

Our verification confirms that:
* Blender Annotation Tool (BAT) outputs multi-layer `.exr` files with clean class boundaries and zero pixel bleed.
* Calibration scenery (`Charuco.png`) is completely isolated from the AI segmentation mask (`Pass Index = 0`).
* High-contrast overhead laboratory lighting satisfies OpenCV (`cv2.aruco`) corner-detection thresholds without casting destructive shadows.

---

## 2. Object Registry & Class Tagging (BAT Panel)
All 5 target medical assets are grouped into distinct Collections and linked to unique class IDs in the **BAT Panel -> Objects** registry. Non-target environment meshes are explicitly demoted to `Pass Index 0` so BAT treats them as Background (`ClassID = 0`).

| Class Name | Blender Collection | Pass Index | Target Pipeline | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **Glass Tube** | `Glass_Tube_Col` | `1–5` | AI Segmentation (YOLO / Mask R-CNN) | **Verified** |
| **Glass Slide** | `Glass_Slide_Col` | `1–5` | AI Segmentation (YOLO / Mask R-CNN) | **Verified** |
| **Slide Holder** | `Slide_Holder_Col` | `1–5` | AI Segmentation (YOLO / Mask R-CNN) | **Verified** |
| **Tube Holder** | `Tube_Holder_Col` | `1–5` | AI Segmentation (YOLO / Mask R-CNN) | **Verified** |
| **Cap** | `Cap_Col` | `1–5` | AI Segmentation (YOLO / Mask R-CNN) | **Verified** |
| **ChArUco Table** | `Calibration_Board` | `0` (Manual) | Background (`0`) / OpenCV Calibration Anchor | **Verified** |

---

## 3. Lighting & Optical Calibration Configuration
To prevent OpenCV corner-detection failures (`cv2.aruco`) during downstream extrinsic calibration, scene lighting and object placement adhere to strict visibility rules:
* **ChArUco Board Placement:** Imported via built-in `Mesh Plane` (`R -> X -> 90`) and elevated slightly above the default ground grid to eliminate Z-fighting/texture flickering.
* **Marker Visibility Rule:** >70% of the ArUco markers and chessboard corners remain unobstructed by medical objects.
* **Overhead Laboratory Lighting:** A `Sun` light is positioned overhead (`Strength: 3.0–5.0`) and oriented vertically downward (`Alt + R`).
  * *Shadow Mitigation:* Vertical orientation prevents long, hard shadows across chessboard squares that cause edge-detection dropouts in OpenCV.
  * *Soft Occlusion:* Sun `Angle` set to `15°–30°` to simulate diffused laboratory lighting.

---

## 4. Dual-Export Verification Workflow
Because BAT overrides standard renderers to output unlit mathematical passes, two separate rendering methods are required depending on which downstream script consumes the data:

| Output Artifact | Trigger Method | Active View / Pass | Downstream Consumer |
| :--- | :--- | :--- | :--- |
| **`0000.exr`** | BAT Panel -> **Render annotation** | `ClassID` (Multi-layer OpenEXR) | Python bbox extraction scripts (`Label_Factory`) for AI vision model training |
| **`calibration_frame.png`** | Keyboard -> **`F12`** -> Save Image | Combined RGB Photograph | ROS2 / OpenCV (`cv2.aruco`) extrinsic camera pose calibration |

---

## 5. Technical Issues Solved & Debug Log

### Issue 1: ChArUco Board Labeled as Target Class in `ClassID` Mask
* **Root Cause:** Importing `Charuco.png` while an active medical collection was highlighted in the Outliner placed the plane inside that collection, stamping it with an active class ID.
* **Resolution:** Moved `Charuco` into a dedicated top-level `Calibration_Board` collection and manually forced **Object Properties -> Relations -> Pass Index = 0**.

### Issue 2: BAT `.exr` Export Rendering Silhouettes Instead of Color Photo
* **Root Cause:** BAT's `Render annotation` operator is strictly designed for synthetic ground-truth layers (`ClassID`, `InstanceID`, `Depth`), intentionally overriding RGB lighting and background textures.
* **Resolution:** Used standard Blender rendering (`F12`) to capture full RGB textured frames for ChArUco optical calibration.

### Issue 3: Camera & Object Controls Unresponsive (Outliner Selection Lock)
* **Root Cause:** Clicking a specific object strip in the Outliner locked viewport transform inputs to that sub-mode selection.
* **Resolution:** Cleared selection by clicking empty Outliner space and re-selecting `Camera` as the sole active object.

---

## 6. Verification Artifacts

### Ground-Truth Segmentation Mask (`ClassID` Pass)
*5 distinct white silhouettes verified against a pitch-black `Pass Index = 0` background:*
![[Rendered Image.png]]

### Extrinsic Calibration Reference Frame (`F12` RGB Pass)
*High-contrast chessboard squares with vertical overhead lighting and zero marker occlusion:*
![[camera_image.png]]