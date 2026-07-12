import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3Stamped
from cv_bridge import CvBridge
import velocity_estimator
from ament_index_python.packages import get_package_share_directory

import os
import sys
import argparse
import cv2
import numpy as np
import torch

try:
    package_share_dir = get_package_share_directory('velocity_estimator')
    raft_core_path = os.path.join(package_share_dir, '../../../../src/AUV_velocity_estimator/velocity_estimator/velocity_estimator/RAFT/core')
    if os.path.exists(raft_core_path):
        sys.path.insert(0, os.path.abspath(raft_core_path))
except Exception:
    pass

from velocity_estimator.speed_estimator import EfficientNetFlowSpeed, load_efficientnet_flow_speed

class VelocityEstimator(Node):
    def __init__(self):
        super().__init__('velocity_estimator')

        package_share_dir = get_package_share_directory('velocity_estimator')

        raft_core_absolute_path = os.path.join(package_share_dir, 'RAFT', 'core')
        sys.path.insert(0, raft_core_absolute_path)

        default_model_path = os.path.join(package_share_dir, 'models', 'efficientnet.pth')
        default_raft_path = os.path.join(package_share_dir, 'models', 'raft-kitti.pth')

        self.declare_parameter('model_path', default_model_path)
        self.declare_parameter('raft_model_path', default_raft_path)
        self.declare_parameter('target_fps', 10.0)
        # Muss zum Training passen (siehe efficientnet_baseline/generate_raw_flow_2ch.py)
        self.declare_parameter('flow_width', 480)
        self.declare_parameter('flow_height', 640)

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        raft_model_path = self.get_parameter('raft_model_path').get_parameter_value().string_value
        self.target_interval = 1.0 / self.get_parameter('target_fps').get_parameter_value().double_value
        self.flow_width = self.get_parameter('flow_width').get_parameter_value().integer_value
        self.flow_height = self.get_parameter('flow_height').get_parameter_value().integer_value

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f'Using device: {self.device}')

        self.get_logger().info('Loading EfficientNetFlowSpeed...')
        self.net = load_efficientnet_flow_speed(model_path, self.device)

        print("Loading RAFT...")
        raft_core_path = os.path.join(os.path.dirname(velocity_estimator.__file__), 'RAFT', 'core')
        sys.path.append(raft_core_path)

        from velocity_estimator.RAFT.core.raft import RAFT

        raft_args = argparse.Namespace(small=False, mixed_precision=False, alternate_corr=False)
        self.raft = RAFT(raft_args)
        weights = torch.load(raft_model_path, map_location=self.device)
        self.raft.load_state_dict({k.replace("module.", ""): v for k, v in weights.items()})
        self.raft = self.raft.to(self.device).eval()

        self.prev_img = None
        self.last_processed_timestamp = 0.0
        self.bridge = CvBridge()

        # Subscribe to the camera topic
        self.subscription = self.create_subscription(
            Image,
            'camera_feed/image_raw',
            self.listener_callback,
            10)

        # Publisher for the velocity results
        self.publisher = self.create_publisher(
            Vector3Stamped,
            'velocity_estimation/result',
            10)

        self.get_logger().info('Velocity Estimator node started. Subscribed to /camera_feed/image_raw, publishing to /velocity_estimation/result')

    def compute_flow(self, img1, img2):
        """Roher (dx, dy)-Flow - keine HSV/RGB-Kodierung, kein Masking.
        img1, img2 muessen bereits auf (flow_width, flow_height) skaliert sein."""
        from velocity_estimator.RAFT.core.utils.utils import InputPadder

        image1 = torch.from_numpy(img1).permute(2,0,1).float()[None].to(self.device)
        image2 = torch.from_numpy(img2).permute(2,0,1).float()[None].to(self.device)

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        with torch.no_grad():
            _, flow_up = self.raft(image1, image2, iters=6, test_mode=True)

        flow_up = padder.unpad(flow_up)
        return flow_up[0].permute(1,2,0).cpu().numpy()  # (H, W, 2) float32, roh

    def listener_callback(self, msg):
        try:
            current_timestamp = msg.header.stamp.sec + (msg.header.stamp.nanosec * 1e-9)

            # Convert ROS image message stream to OpenCV BGR image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Resize VOR der Flow-Berechnung (nicht danach!) - muss zum
            # Training passen, siehe efficientnet_baseline/generate_raw_flow_2ch.py
            curr_img = cv2.resize(cv_image[:, :, ::-1], (self.flow_width, self.flow_height),
                                  interpolation=cv2.INTER_LINEAR).astype(np.uint8)

            if self.prev_img is not None:

                # 1. optical flow - roh, kein Masking/Segmentierung mehr noetig
                flow = self.compute_flow(self.prev_img, curr_img)

                # 2. prep CNN - keine /255-Normalisierung, roher Flow ist kein 0-255-Bild
                x = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0).to(self.device)

                # 3. Inference
                with torch.no_grad():
                    estimation_result = max(self.net(x).item(), 0.0)

                # Nachricht bauen und senden
                out_msg = Vector3Stamped()
                out_msg.header = msg.header 
                out_msg.vector.x = estimation_result

                self.publisher.publish(out_msg)

                self.get_logger().info(
                    f'Timestamp: {current_timestamp:.4f} | Velocity: {estimation_result:.2f} m/s, {estimation_result*3.6:.2f} km/h'
                )

            else:
                self.get_logger().info('Initializing pipeline... Waiting for second frame.')

            # WICHTIG: prev_img wird JETZT IMMER lückenlos aktualisiert
            self.prev_img = curr_img
            self.last_processed_timestamp = current_timestamp

        except Exception as e:
            self.get_logger().error(f'Failed to process image: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    estimator = VelocityEstimator()
    try:
        rclpy.spin(estimator)
    except KeyboardInterrupt:
        pass
    finally:
        estimator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()