
# Synthetic Data Pipeline: Blender + BAT + Label_Factory

## 1. Architecture & Workflow
Our goal is to generate synthetic ground-truth segmentation masks for transparent medical objects so an AI vision model can guide a Universal Robots (UR) arm.
* **Blender (Windows) + BAT (Blender Annotation Tool):** Generates 3D scenes, assigns class IDs, and renders multi-layer ground-truth segmentation masks (`.exr`).
* **ROS2 (`Label_Factory` in WSL2):** Orchestrates calibration, camera intrinsics/extrinsics, and triggers Blender over network port `12345`.
* **ChArUco Board (`Charuco.png`):** Bridges the virtual Blender camera coordinate system to the physical UR robot camera setup.

---

## 2. Object Registry & Class Assignments
All meshes must be segregated into specific Collections and explicitly linked inside the BAT Panel registry. If an object is not assigned in the BAT Panel, its `ClassID` defaults to `0` (Background).

| Class ID Name | Blender Collection | Status |
| :--- | :--- | :--- |
| **Glass Tube** | `Glass_Tube_Col` | Verified |
| **Glass Slide** | `Glass_Slide_Col` | Verified |
| **Slide Holder** | `Slide_Holder_Col` | Verified |
| **Tube Holder** | `Tube_Holder_Col` | Verified |
| **Cap** | `Cap_Col` | Verified |

---

## 3. Debugging & Error Log (Blender / BAT)

### Error 1: `TypeError: 'BAT_ViewLayer' not found in ('BAT_ViewLayer_001')`
* **Root Cause:** BAT's Python script is hardcoded to look for `BAT_ViewLayer`. If a render crashes or a layer is duplicated, Blender renames subsequent layers to `_001`, breaking BAT's lookup.
* **Fix:** 
  1. Delete any duplicate/broken `BAT_ViewLayer_001` layers in the top-right Outliner dropdown.
  2. Keep the primary layer named default `ViewLayer` so BAT can cleanly generate temporary layers without naming collisions.
  3. Purge orphaned data: `File -> Clean Up -> Recursive Unused Data-Blocks`.

### Error 2: `RuntimeError: Cannot render, no camera`
* **Root Cause:** Having a `Camera` object in the Outliner does not automatically assign it as the scene's active rendering lens.
* **Fix:** 
  1. Go to **Scene Properties** (cone + sphere icon in bottom-right).
  2. Under **Scene -> Camera**, explicitly assign the `Camera` object.

### Error 3: Meshes Rendering as Pitch Black (ClassID = 0)
* **Root Cause:** BAT ignores meshes that are either unassigned in the `BAT Panel -> Objects` field or were moved into collections after BAT initialized its internal `pass_index`.
* **Fix:** 
  1. Select the missing class in `BAT Panel -> Current class`.
  2. Remove and re-add the target collection (e.g., `Slide_Holder_Col`) to force Blender to re-stamp the mesh with a valid Class ID.

---

## 4. Verification: The Multi-Layer `.EXR` Mask
* **Output File:** `0000.exr` (OpenEXR format).
* **Why EXR?:** Instead of exporting separate PNGs for RGB, Segmentation, and Depth, OpenEXR stores 32-bit floating-point data for all layers in one file.
* **ClassID Pass:**
  * `0` = Background (Black)
  * `1–5` = Solid white/labeled pixel silhouettes for each medical object.
* **Why No `.JSON` Yet?:** BAT exports raw pixel masks in `.exr`. Automated Python/ROS scripts in `Label_Factory` parse the `.exr` ClassID boundaries downstream to compute YOLO/COCO `.json` bounding boxes.

  
# Synthetic Data Pipeline: Blender + BAT + Label_Factory

## 1. Architecture & Workflow
Our goal is to generate synthetic ground-truth segmentation masks for transparent medical objects so an AI vision model can guide a Universal Robots (UR) arm.
* **Blender (Windows) + BAT (Blender Annotation Tool):** Generates 3D scenes, assigns class IDs, and renders multi-layer ground-truth segmentation masks (`.exr`).
* **ROS2 (`Label_Factory` in WSL2):** Orchestrates calibration, camera intrinsics/extrinsics, and triggers Blender over network port `12345`.
* **ChArUco Board (`Charuco.png`):** Bridges the virtual Blender camera coordinate system to the physical UR robot camera setup.

---

## 2. Object Registry & Class Assignments
All meshes must be segregated into specific Collections and explicitly linked inside the BAT Panel registry. If an object is not assigned in the BAT Panel, its `ClassID` defaults to `0` (Background).

| Class ID Name | Blender Collection | Status |
| :--- | :--- | :--- |
| **Glass Tube** | `Glass_Tube_Col` | Verified |
| **Glass Slide** | `Glass_Slide_Col` | Verified |
| **Slide Holder** | `Slide_Holder_Col` | Verified |
| **Tube Holder** | `Tube_Holder_Col` | Verified |
| **Cap** | `Cap_Col` | Verified |

---

## 3. Debugging & Error Log (Blender / BAT)

### Error 1: `TypeError: 'BAT_ViewLayer' not found in ('BAT_ViewLayer_001')`
* **Root Cause:** BAT's Python script is hardcoded to look for `BAT_ViewLayer`. If a render crashes or a layer is duplicated, Blender renames subsequent layers to `_001`, breaking BAT's lookup.
* **Fix:** 
  1. Delete any duplicate/broken `BAT_ViewLayer_001` layers in the top-right Outliner dropdown.
  2. Keep the primary layer named default `ViewLayer` so BAT can cleanly generate temporary layers without naming collisions.
  3. Purge orphaned data: `File -> Clean Up -> Recursive Unused Data-Blocks`.

### Error 2: `RuntimeError: Cannot render, no camera`
* **Root Cause:** Having a `Camera` object in the Outliner does not automatically assign it as the scene's active rendering lens.
* **Fix:** 
  1. Go to **Scene Properties** (cone + sphere icon in bottom-right).
  2. Under **Scene -> Camera**, explicitly assign the `Camera` object.

### Error 3: Meshes Rendering as Pitch Black (ClassID = 0)
* **Root Cause:** BAT ignores meshes that are either unassigned in the `BAT Panel -> Objects` field or were moved into collections after BAT initialized its internal `pass_index`.
* **Fix:** 
  1. Select the missing class in `BAT Panel -> Current class`.
  2. Remove and re-add the target collection (e.g., `Slide_Holder_Col`) to force Blender to re-stamp the mesh with a valid Class ID.

---

## 4. Verification: The Multi-Layer `.EXR` Mask
* **Output File:** `0000.exr` (OpenEXR format).
* **Why EXR?:** Instead of exporting separate PNGs for RGB, Segmentation, and Depth, OpenEXR stores 32-bit floating-point data for all layers in one file.
* **ClassID Pass:**
  * `0` = Background (Black)
  * `1–5` = Solid white/labeled pixel silhouettes for each medical object.
* **Why No `.JSON` Yet?:** BAT exports raw pixel masks in `.exr`. Automated Python/ROS scripts in `Label_Factory` parse the `.exr` ClassID boundaries downstream to compute YOLO/COCO `.json` bounding boxes.

![[Rendered Image.png]]

---

## 5. Next Phase: Extrinsic Camera Calibration
* **Objective:** Align the virtual camera pose with physical reality.
* **Action:** Import `Charuco.png` (from `Label_Factory`) into the Blender table scene using `Images as Planes` to serve as the optical calibration anchor.
---

## 5. Next Phase: Extrinsic Camera Calibration
* **Objective:** Align the virtual camera pose with physical reality.
* **Action:** Import `Charuco.png` (from `Label_Factory`) into the Blender table scene using `Images as Planes` to serve as the optical calibration anchor.