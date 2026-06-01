#!/bin/bash
set -e

# Source ROS 2 base and our local workspace
source "/opt/ros/$ROS_DISTRO/setup.bash"
source "/auv_ws/install/setup.bash"

exec "$@"