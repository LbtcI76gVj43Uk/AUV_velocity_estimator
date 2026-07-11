# AUV Velocity Estimator

Lorem Ipsum

## Repository Structure

```text
.
├── Dockerfile
├── README.md
├── requirements.txt
├── ros_entrypoint.sh
└── velocity_estimator
    ├── launch
    │   └── velocity_estimator_launch.py
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
        ├── RAFT
        ├── __init__.py
        ├── __pycache__
        ├── camera_publisher.py
        ├── estimator_pipeline.py
        ├── logger.py
        ├── prepare_pipeline_input.py
        ├── segmentation_mask.py
        └── speed_estimator.py

7 directories, 19 files

```

---

## Setup

This setup assumes you are using Linux via WSL on Windows and Podman. Docker should work just as well.

### Prerequisites

* **Podman:** [podman](https://podman.io/docs/installation) installed in WSL.
* **Camera Passthrough:** [usbipd-win](https://github.com/dorssel/usbipd-win) installed on Windows.

### Connect Hardware (PowerShell)

To make a USB camera visible to the Linux kernel inside WSL/Podman, run **PowerShell as Administrator**:

```powershell
# 1. Identify the BusID of your camera
usbipd list

# 2. Attach the device to WSL
usbipd attach --wsl --busid <BusID>
```

Go to WSL and verify camera availability and accessibility

```bash
# Verify it appears in WSL
ls /dev/video*

# Set rights for device (adjust index)
sudo chmod 666 /dev/video0

```

### Build containerized

Select CPU or GPU build

```bash
cd <path/to/repo>

# Build the image for cpu (default)
podman build --cgroup-manager=cgroupfs --build-arg PYTORCH_WHL=cpu -t velocity_estimator .

# Build the image for gpu
podman build --cgroup-manager=cgroupfs --build-arg PYTORCH_WHL=cu121 -t velocity_estimator .
```

#### Building Locally

The project can also be build locally assuming ROS2 Humble is installed

```bash
# Build workspace
colcon build --symlink-install

# Source and Run
source install/setup.bash
ros2 run velocity_estimator camera_node
```

### Run

```bash

# Run the container with cpu build
podman run -it --rm --privileged -e OPENCV_VIDEOIO_PRIORITY_BACKEND=V4L2 --device=/dev/video0:/dev/video0 -v <path/to/models>:/auv_ws/install/velocity_estimator/share/velocity_estimator/models:Z --group-add=keep-groups --security-opt label=disable velocity_estimator ros2 launch velocity_estimator velocity_estimator_launch.py freq:=5.0 video_index:=0 run_logger:=true

# Run the container with gpu build
podman run -it --rm --device nvidia.com/gpu=all -v ~/dev/images/frames/:/auv_ws/src/velocity_estimator/test_images:Z -v ~/dev/models/:/auv_ws/install/velocity_estimator/share/velocity_estimator/models:Z --group-add=keep-groups --security-opt label=disable velocity_estimator:latest ros2 launch velocity_estimator velocity_estimator_launch.py freq:=10.0 image_folder:=/auv_ws/src/velocity_estimator/test_images

```

The results are published to the topic `/velocity_estimation/result` and can be either echoed directly from terminal or stored as .csv via logger node.

```bash

# Echo result topic
ros2 topic echo /velocity_estimation/result
```

---

## ROS 2 API

### Published Topics

| Topic | Type | Description |
| --- | --- | --- |
| `/camera_feed/image_raw` | `sensor_msgs/msg/Image` | Raw BGR8 camera stream |
| `/velocity_estimation/result` | `geometry_msgs/msg/Vector3Stamped` | Results of velocity estimation |

### Node Parameters

* **Frequency:** Defaults to 30Hz (configurable in `camera_publisher.py`).
* **Device Index:** Defaults to `0` (matches `/dev/video0`).
* **Logging Node:** Defaults to `false`.
