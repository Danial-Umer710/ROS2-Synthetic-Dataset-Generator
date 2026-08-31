
# 07 - API Verification & Asset Scaling

## Phase 7: Service Testing & Geometric Accuracy

### 1. HTTP Endpoint Verification
Tested the BAT local server bridge to ensure it is actively listening for cross-OS remote control commands.
* **Execution:** Polled the API via `curl http://localhost:12345/frame` on the host machine.
* **Expected Output:** A valid JSON payload confirming the current Blender frame (e.g., `{"status": "success", "frame": 1}`). 

### 2. Strict Dimensional Accuracy
Standard primitive meshes must be scaled to perfectly match the physical objects. The automated label projection relies entirely on matched mathematical bounds between the physical desk and the virtual environment.
* Deployed bounding primitives for the target objects.
* Modified absolute dimensions (X, Y, Z) to match physical caliper measurements.
* **Critical Operation:** Executed `Apply Scale` (`Ctrl + A` -> `Scale`) on all meshes to reset the transform matrix to 1.0. Failure to apply scale results in distorted bounding boxes during dataset generation.
