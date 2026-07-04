import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3Stamped
from cv_bridge import CvBridge
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

from velocity_estimator.speed_estimator import BojarskiCNN
from velocity_estimator.segmentation_mask import predict_class_map, get_dynamic_mask
from velocity_estimator.prepare_pipeline_input import dilate_mask, inpaint_flow

class VelocityEstimator(Node):
    def __init__(self):
        super().__init__('velocity_estimator')
        
        package_share_dir = get_package_share_directory('velocity_estimator')
    
        raft_core_absolute_path = os.path.join(package_share_dir, 'RAFT', 'core')
        sys.path.insert(0, raft_core_absolute_path)

        default_model_path = os.path.join(package_share_dir, 'models', 'model.pth')
        default_raft_path = os.path.join(package_share_dir, 'models', 'raft-kitti.pth')

        self.declare_parameter('model_path', default_model_path)
        self.declare_parameter('raft_model_path', default_raft_path)
        self.declare_parameter('target_fps', 10.0)
        
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        raft_model_path = self.get_parameter('raft_model_path').get_parameter_value().string_value
        self.target_interval = 1.0 / self.get_parameter('target_fps').get_parameter_value().double_value
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f'Using device: {self.device}')

        self.get_logger().info('Loading BojarskiCNN...')
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            dropout_p = checkpoint.get("dropout_p", 0.0)
            state_dict = checkpoint["state_dict"]
        else:
            dropout_p = 0.0
            state_dict = checkpoint
        self.cnn = BojarskiCNN(dropout_p=dropout_p)
        self.cnn.load_state_dict(state_dict)
        self.cnn = self.cnn.to(self.device).eval()

        self.get_logger().info('Loading RAFT...')
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

    def compute_flow_rgb(self, img1, img2):
        from velocity_estimator.RAFT.core.utils.utils import InputPadder
        
        image1 = torch.from_numpy(img1).permute(2,0,1).float()[None].to(self.device)
        image2 = torch.from_numpy(img2).permute(2,0,1).float()[None].to(self.device)

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        with torch.no_grad():
            _, flow_up = self.raft(image1, image2, iters=6, test_mode=True)

        flow_up = padder.unpad(flow_up)
        flow = flow_up[0].permute(1,2,0).cpu().numpy()

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        hsv[...,0] = ang * 180 / np.pi / 2
        hsv[...,1] = 255
        hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def preprocess(self, frame_rgb, flow_rgb):
        small_frame = cv2.resize(frame_rgb, (220, 110), interpolation=cv2.INTER_LINEAR)
        class_map = predict_class_map(small_frame)
        mask = get_dynamic_mask(class_map)
        fh, fw = flow_rgb.shape[:2]
        if mask.shape != (fh, fw):
            mask = cv2.resize(mask.astype(np.uint8), (fw, fh), interpolation=cv2.INTER_NEAREST)
        dilated = dilate_mask(mask, kernel_size=15)
        inpainted = inpaint_flow(flow_rgb, dilated, inpaint_radius=3)
        return cv2.resize(inpainted, (220, 110), interpolation=cv2.INTER_LINEAR)

    def listener_callback(self, msg):
        try:
            # convert ros timestamp into seconds
            current_timestamp = msg.header.stamp.sec + (msg.header.stamp.nanosec * 1e-9)
            
            # fps matching
            if self.prev_img is not None:
                time_since_last_process = current_timestamp - self.last_processed_timestamp
                if time_since_last_process < (self.target_interval - 0.005): # 5ms tolerance
                    return
            
            # Convert ROS image message stream to OpenCV BGR image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            curr_img = cv_image[:, :, ::-1].copy().astype(np.uint8)
            
            if self.prev_img is not None:
                
                # 1. optical flow
                flow_rgb = self.compute_flow_rgb(self.prev_img, curr_img)
                
                # 2. segmentation
                flow_input = self.preprocess(curr_img, flow_rgb)
                
                # 3. prep CNN
                x = torch.from_numpy(flow_input.astype(np.float32) / 255.0)
                x = x.permute(2, 0, 1).unsqueeze(0).to(self.device)
                
                # 4. Inference
                with torch.no_grad():
                    estimation_result = max(self.cnn(x).item(), 0.0)
                
                # Extract the timestamp from the ROS Header
                self.last_processed_timestamp = current_timestamp
                
                # Construct and publish telemetry message
                out_msg = Vector3Stamped()
                out_msg.header = msg.header # hardware timestamp from the camera frame
                out_msg.vector.x = estimation_result # velocity value into the x channel
                
                self.publisher.publish(out_msg)
                
            else:
                self.get_logger().info('Initializing pipeline... Waiting for second frame.')
                self.last_processed_timestamp = current_timestamp
            
            self.prev_img = curr_img

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
