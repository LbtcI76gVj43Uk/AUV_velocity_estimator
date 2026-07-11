import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3Stamped
from cv_bridge import CvBridge
import velocity_estimator
from ament_index_python.packages import get_package_share_directory

import os
import sys
import cv2
import numpy as np
import torch

from velocity_estimator.raft_flow import RaftFlowEstimator
from velocity_estimator.speed_estimator import load_bojarski_cnn

class VelocityEstimator(Node):
    def __init__(self):
        super().__init__('velocity_estimator')

        package_share_dir = get_package_share_directory('velocity_estimator')

        # Pfade auflösen
        default_model_path = os.path.join(package_share_dir, 'models', 'model.pth')
        default_raft_path = os.path.join(package_share_dir, 'models', 'raft-kitti.pth')

        self.declare_parameter('model_path', default_model_path)
        self.declare_parameter('raft_model_path', default_raft_path)

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        raft_model_path = self.get_parameter('raft_model_path').get_parameter_value().string_value

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f'Using device: {self.device}')

        # 1. Korrektes Modell laden (BojarskiCNN)
        self.get_logger().info('Loading BojarskiCNN...')
        self.net = load_bojarski_cnn(model_path, self.device)

        # 2. RAFT über den sauberen Wrapper laden
        self.get_logger().info('Loading RAFT Estimator...')
        self.raft_estimator = RaftFlowEstimator(weights_path=raft_model_path, device=self.device)

        self.prev_img = None
        self.bridge = CvBridge()

        # Feste Zielgröße für das BojarskiCNN (aus Tabelle I/II des Papers)
        self.cnn_width = 220
        self.cnn_height = 110

        # Subscriptions & Publishers
        self.subscription = self.create_subscription(
            Image,
            'camera_feed/image_raw',
            self.listener_callback,
            10)

        self.publisher = self.create_publisher(
            Vector3Stamped,
            'velocity_estimation/result',
            10)

        self.get_logger().info('Velocity Estimator node started and ready.')

    def listener_callback(self, msg):
        try:
            current_timestamp = msg.header.stamp.sec + (msg.header.stamp.nanosec * 1e-9)

            # ROS Image -> OpenCV BGR
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # RAFT arbeitet intern am besten mit RGB
            curr_img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            if self.prev_img is not None:
                # 1. Berechne RGB-codierten Flow (HSV -> RGB Visualisierung wie im Paper!)
                flow_rgb = self.raft_estimator.compute_flow_rgb(self.prev_img, curr_img_rgb)

                # 2. Resize auf die exakte Eingangsgröße des BojarskiCNN (110, 220)
                flow_resized = cv2.resize(flow_rgb, (self.cnn_width, self.cnn_height), interpolation=cv2.INTER_LINEAR)

                # 3. Vorbereitung für PyTorch (HWC -> CHW, Normalisierung auf 0.0 - 1.0)
                x = torch.from_numpy(flow_resized).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
                x = x / 255.0  # Da es jetzt ein echtes RGB-Bild ist, wird durch 255 geteilt

                # 4. Inferenz
                with torch.no_grad():
                    estimation_result = max(self.net(x).item(), 0.0)

                # Senden
                out_msg = Vector3Stamped()
                out_msg.header = msg.header
                out_msg.vector.x = estimation_result
                self.publisher.publish(out_msg)
                
                self.get_logger().info(
                    f'Timestamp: {current_timestamp:.4f} | Velocity: {estimation_result:.2f} m/s, {estimation_result*3.6:.2f} km/h'
                )

            else:
                self.get_logger().info('Initializing pipeline... Waiting for second frame.')

            # Frame-Historie lückenlos aktualisieren
            self.prev_img = curr_img_rgb

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