
# 03 - Offline Validation & Calibration

## Phase 3: Application Verification

### 1. Validating the Build Pipeline
To prove the ROS2 workspace, Python dependencies, and `Label_Factory` scripts are communicating correctly without requiring physical hardware, execute the ChArUco generation node. 

The application uses hidden Python libraries in `.LF_venv`, so the `PYTHONPATH` must be exported before execution:

```bash
cd ~/Projects/UR_Robotics_Project/Label_Factory
export PYTHONPATH=".LF_venv/lib/python3.12/site-packages:$PYTHONPATH"
ros2 run label_factory ChArUco
````

### 2. Extracting the Calibration Artifact

The previous command generates `Charuco.png` in the project root. This acts as the physical anchor for the camera during the real-world dataset collection. Extract this file to the Windows host for printing:

Bash

```
explorer.exe .
```
