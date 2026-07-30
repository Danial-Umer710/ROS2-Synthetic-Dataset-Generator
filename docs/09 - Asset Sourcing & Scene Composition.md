# 09 - Asset Sourcing & Scene Composition

## Phase 9: Topology & Data Collection Strategy

### 1. Asset Sourcing Protocol
Avoid utilizing third-party 3D models for basic geometric shapes. 
* **Dependency Risk:** Downloaded assets frequently contain bloated polygon counts, offset origin points, and unapplied matrix transforms. This introduces unnecessary variables that can silently corrupt the automated projection math.
* **Execution:** Manually generate clean, native mesh primitives (Cylinder, Cube) within Blender. This ensures strict control over object topology and pivot points.

### 2. Physical Scene Composition
Objects must be tracked as independent mathematical entities.
* **Physical Staging:** Extract the glass tube from the box during initial data collection. The vision model requires unobstructed views of the transparent edges to establish a baseline before introducing occlusions.
* **Digital Twin Constraints:** `Medical_Box` and `Glass_Tube` must remain completely isolated mesh objects within the Blender Outliner hierarchy. Merging them will irreversibly break the instance segmentation ID tracking during batch rendering.
