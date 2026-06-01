import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        
        # default parameter values
        self.declare_parameter('video_index', 0)
        self.declare_parameter('frequency', 30.0)

        video_index = self.get_parameter('video_index').get_parameter_value().integer_value
        frequency = self.get_parameter('frequency').get_parameter_value().double_value

        self.get_logger().info(f'Starting camera node with index: {video_index} at {frequency}Hz')

        self.publisher_ = self.create_publisher(Image, 'camera/image_raw', 10)
        
        # Capture frequency in Hz
        timer_period = 1.0 / frequency 
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.cap = cv2.VideoCapture(video_index)
        self.bridge = CvBridge()

    def timer_callback(self):
        self.get_logger().info('Publishing video frame')
        ret, frame = self.cap.read()
        if ret:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            self.publisher_.publish(msg)
        else:
            self.get_logger().warn('Failed to capture frame', once=True)

    def __del__(self):
        self.cap.release()

def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
