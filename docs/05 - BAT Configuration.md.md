# 05 - BAT Configuration

## Phase 5: Synthetic Data Engine Setup

### 1. Installing the Add-on
The BAT plugin must be installed directly from its compressed archive. 
1. Download the BAT repository as a `.zip` file. Do not extract it. 
2. In Blender, navigate to `Edit > Preferences > Add-ons`. 
3. Click the dropdown arrow in the upper right corner and select `Install from Disk`. 
4. Select the downloaded `.zip` file. 
5. Locate the add-on in the **Render** category and click the checkbox to activate it.

### 2. Critical Initialization (Viewport Trigger)
Immediately after activating the add-on, you must click anywhere inside the main Blender 3D viewport. This manually triggers a background script to set the default 'Background' class. Failing to do this leaves the background class missing and corrupts the dataset generation logic.

### 3. The HTTP Communication Bridge
BAT automatically starts a local HTTP server on port `12345` when Blender launches. This serves as the cross-OS bridge. The Linux WSL (ROS2) pipeline will send automated HTTP POST requests to this Windows port to dynamically update camera parameters, move the 3D medical objects, and trigger the OpenEXR Multilayer render outputs without manual intervention.