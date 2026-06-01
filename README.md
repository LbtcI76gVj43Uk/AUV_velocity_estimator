# AUV Velocity Estimator

Lorem Ipsum

## Repository Structure

```text
.
├── Dockerfile
├── README.md
├── ros2_network_interfaces
│   ├── CMakeLists.txt
│   ├── include
│   │   └── ros2_network_interfaces
│   ├── package.xml
│   └── src
├── ros_entrypoint.sh
└── velocity_estimator
    ├── package.xml
    ├── resource
    │   └── velocity_estimator
    ├── setup.cfg
    ├── setup.py
    ├── test
    │   ├── test_copyright.py
    │   ├── test_flake8.py
    │   └── test_pep257.py
    └── velocity_estimator
        ├── __init__.py
        └── camera_publisher.py

```

---

## Setup

### 1. Prerequisites

* **Windows Host:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) with WSL2 backend enabled.
* **Camera Passthrough:** [usbipd-win](https://github.com/dorssel/usbipd-win) installed on Windows.

### 2. Connect Hardware (PowerShell)

To make your USB camera visible to the Linux kernel inside WSL/Docker, run **PowerShell as Administrator**:

```bash
# 1. Identify the BusID of your camera
usbipd list

# 2. Attach the device to WSL
usbipd attach --wsl --busid <BusID>

# 3. Verify it appears in WSL
wsl ls /dev/video*

```

### 3. Build and Run

From the root of this repository:

```bash
# Build the image
docker build -t velocity_estimator .

# Run the container with hardware access
docker run -it --rm --device=/dev/video0:/dev/video0 velocity_estimator ros2 run velocity_estimator camera_node --ros-args -p frequency:=10.0 -p video_index:=0

```

## Building Locally

If you prefer to build without Docker:

```bash
# Build workspace
colcon build --symlink-install

# Source and Run
source install/setup.bash
ros2 run velocity_estimator camera_node

```

---

## ROS 2 API

### Published Topics

| Topic | Type | Description |
| --- | --- | --- |
| `/camera/image_raw` | `sensor_msgs/msg/Image` | Raw BGR8 camera stream |

### Node Parameters

* **Frequency:** Defaults to 30Hz (configurable in `camera_publisher.py`).
* **Device Index:** Defaults to `0` (matches `/dev/video0`).

---
