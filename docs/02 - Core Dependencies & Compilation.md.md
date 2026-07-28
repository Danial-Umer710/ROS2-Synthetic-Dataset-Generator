
# 02 - Core Dependencies & Compilation

## Phase 2: Robotics Libraries & Workspace Build

### 1. The Build Orchestrator (colcon)
ROS2 does not use standard compilers directly. It uses `colcon` as an orchestrator to map dependencies and trigger the correct C++ or Python builders in the exact right order. Install the colcon extensions:

```bash
sudo apt install python3-colcon-common-extensions -y
````

### 2. Core Robotics Binaries

The `Label_Factory` requires MoveIt2 (for motion planning) and the official Universal Robots driver. Building these from source can take hours and cause dependency hell. Instead, pull their pre-compiled binaries directly from the ROS2 servers:

Bash

```
sudo apt install ros-jazzy-moveit ros-jazzy-ur -y
```

### 3. The Dependency Auditor (rosdep)

Before building custom code, we must ensure Ármin's team didn't use any hidden third-party libraries. `rosdep` acts as an auditor that scans the local folders and automatically installs any missing dependencies. Initialize and run the scan:

Bash

```
sudo apt install python3-rosdep -y
sudo rosdep init
rosdep update
rosdep install --from-paths . --ignore-src -r -y
```

_(Note: If python throws a `DeprecationWarning` about `pkg_resources`, ignore it. As long as it prints "All required rosdeps installed successfully", the system is clean.)_

### 4. Compiling the Workspace

With the environment primed and dependencies resolved, trigger the orchestrator to build the actual `Label_Factory` repository. Ensure you are in the root `Label_Factory` directory when running this:

Bash

```
colcon build
```

### 5. Local Workspace Overlay

Even after compiling successfully, ROS2 does not automatically know this new code exists. You must explicitly tell Linux to look inside the newly generated `install` folder. Automate this in `.bashrc` so the terminal always loads your local project on startup:

Bash

```
source install/setup.bash
echo "source ~/Projects/UR_Robotics_Project/Label_Factory/install/setup.bash" >> ~/.bashrc
```