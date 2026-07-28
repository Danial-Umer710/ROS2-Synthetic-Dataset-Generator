




# 01 - Environment Setup

## Phase 1: ROS2 Infrastructure

### 1. Filesystem Isolation
WSL file systems can cause severe compilation lag and permission errors if cross-talk occurs between Windows and Linux. The workspace must be built directly in the native Ubuntu filesystem:

```bash
mkdir -p ~/Projects/UR_Robotics_Project
cd ~/Projects/UR_Robotics_Project
git clone [https://github.com/ABC-iRobotics/Label_Factory.git](https://github.com/ABC-iRobotics/Label_Factory.git)
cd Label_Factory
````

### 2. Strict Locale Configuration

ROS2 has a hard requirement for UTF-8 language settings. If the locale is not set precisely, the ROS2 installation will fail silently halfway through. Ensure Ubuntu is configured correctly:

Bash

```
sudo apt update && sudo apt install locales -y
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
```

### 3. Importing Security Keys & Repositories

Ubuntu only knows about standard software by default. Before downloading ROS2, you must enable the Universe repository and provide the cryptographic GPG key to prove the ROS2 downloads are legitimate:

Bash

```
sudo apt install software-properties-common curl -y
sudo add-apt-repository universe -y

sudo curl -sSL [https://raw.githubusercontent.com/ros/rosdistro/master/ros.key](https://raw.githubusercontent.com/ros/rosdistro/master/ros.key) -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] [http://packages.ros.org/ros2/ubuntu](http://packages.ros.org/ros2/ubuntu) $(. /etc/os-release && echo$UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

### 4. Installing ROS2 Jazzy

With the security keys in place, update the system's package list to see the new ROS2 servers, then pull down the heavy desktop installation:

Bash

```
sudo apt update
sudo apt install ros-jazzy-desktop -y
```

### 5. Automating Environment Activation

ROS2 installs into `/opt/ros/jazzy`, which the terminal ignores by default. The environment must be activated manually. To prevent doing this on every Windows reboot or WSL launch, automate it via the `.bashrc` file:

Bash

```
source /opt/ros/jazzy/setup.bash
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
```

_Verify the environment variable is active:_

Bash

```
echo $ROS_DISTRO
```````