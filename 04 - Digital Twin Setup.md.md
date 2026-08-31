
# 04 - Digital Twin Setup

## Phase 4: Synthetic Data Environment (Windows Host)

### 1. The Environment Split
While the ROS2 backend processes kinematics and data pipelines natively in Linux (WSL), 3D rendering engines require direct hardware access to the GPU. To prevent graphical bottlenecks and display driver errors, the Digital Twin components are installed strictly on the Windows host.

### 2. Installing Blender 4.3.2
The `Label_Factory` scripts are version-locked to ensure the Python API matches the toolsets. 
* Download the Windows `.msi` installer for **Blender 4.3.2** from the official Blender archives.
* Execute the installer and proceed with the standard system installation.

### 3. The Blender Annotation Tool (BAT)
BAT is the plugin responsible for calculating where the 3D objects are on the screen and generating the actual machine learning labels (segmentation masks, bounding boxes, depth maps).
* Download the BAT release `.zip` file as referenced in the repository documentation. 
* **Crucial:** Do not extract the `.zip` file. Blender requires the archive to remain intact for installation.

### 4. Initializing BAT
Inject the BAT plugin into the Blender Python environment:
1. Launch Blender on Windows.
2. Navigate to `Edit` > `Preferences` > `Add-ons`.
3. Click `Install...` (or the dropdown arrow at the top right) and locate the BAT `.zip` file.
4. Enable the add-on by checking the box next to its name in the list.