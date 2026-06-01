# Use the official ROS 2 Humble base image
FROM osrf/ros:humble-desktop

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    ros-humble-cv-bridge \
    ros-humble-sensor-msgs \
    libopencv-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Create and set the workspace
WORKDIR /auv_ws
COPY ./ros2_network_interfaces ./src/ros2_network_interfaces
COPY ./velocity_estimator ./src/velocity_estimator

# Install ROS dependencies using rosdep
RUN . /opt/ros/$ROS_DISTRO/setup.sh && \
    apt-get update && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -y && \
    rm -rf /var/lib/apt/lists/*

# Build the workspace
RUN . /opt/ros/$ROS_DISTRO/setup.sh && \
    colcon build --symlink-install

# Setup the entrypoint
COPY ./ros_entrypoint.sh /
RUN sed -i 's/\r$//' /ros_entrypoint.sh && chmod +x /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]

CMD ["ros2", "launch", "velocity_estimator", "velocity_estimator_launch.py"]
