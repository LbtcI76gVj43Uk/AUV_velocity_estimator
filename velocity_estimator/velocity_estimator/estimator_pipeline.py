import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class VelocityEstimator(Node):
    def __init__(self):
        super().__init__('velocity_estimator')
        
        # Subscribe to the topic published by your camera_node
        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        
        self.bridge = CvBridge()
        self.get_logger().info('Velocity Estimator node started. Subscribed to /camera/image_raw')

    def listener_callback(self, msg):
        try:
            # Convert ROS image message stream to OpenCV BGR image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # placeholder
            gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray_image)
            
            # Extract the timestamp from the ROS Header
            timestamp = msg.header.stamp.sec + (msg.header.stamp.nanosec * 1e-9)
            
            self.get_logger().info(
                f'Timestamp: {timestamp:.4f} | Velovity: {avg_brightness:.2f} m/s'
            )
            
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
