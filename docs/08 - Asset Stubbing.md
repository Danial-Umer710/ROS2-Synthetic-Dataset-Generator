
# 08 - Asset Stubbing

## Phase 8: Digital Twin Placeholders & Measurement Strategy

### 1. Asset Stubbing
Due to the unavailability of physical measurements during the offline preparation phase, 3D assets are currently acting as structural stubs.
* **Action Taken:** Generated generic primitive meshes (`Medical_Box`, `Glass_Tube`) to establish the Outliner hierarchy and allow saving the baseline `digital_twin_scene.blend` file. 

### 2. Hardware Day Prerequisite (Blocker)
The automated projection pipeline strictly requires 1:1 scale mapping. The following protocol must be executed before initiating any robot kinematics or ROS2 data generation sequences:
1. **Physical Measurement:** Acquire precise X, Y, Z bounding dimensions of all target medical objects using calipers.
2. **Dimension Updates:** Input physical measurements into the Blender transform panel (`N`).
3. **Matrix Reset:** Execute `Ctrl + A` -> `Scale` on all updated meshes to bake the transforms. Proceeding without applying scale will corrupt the generated annotation coordinates.
