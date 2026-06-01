# AUV Velocity Estimator

Lorem Ipsum

## Repository Structure

```text
.
├── ros2_network_interfaces/   # Custom ROS 2 message/service definitions
├── velocity_estimator/        # Python package for CV and estimation
│   ├── velocity_estimator/
│   │   └── camera_publisher.py # Main camera streaming node
│   └── package.xml            # Dependencies (cv_bridge, sensor_msgs)
├── Dockerfile                 # Multi-stage ROS 2 Humble build
├── ros_entrypoint.sh          # Environment sourcing and LF-fix script
└── README.md

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

# 2. Attach the device to WSL (replace 2-3 with your BusID)
usbipd attach --wsl --busid 2-3

# 3. Verify it appears in WSL
wsl ls /dev/video*

```

### 3. Build and Run

From the root of this repository:

```bash
# Build the image
docker build -t velocity_estimator .

# Run the container with hardware access
docker run -it --rm --device=/dev/video0:/dev/video0 velocity_estimator

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
