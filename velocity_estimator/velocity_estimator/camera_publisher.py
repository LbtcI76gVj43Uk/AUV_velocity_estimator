import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import os
from glob import glob

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        
        # default parameter values
        # Wir deklarieren einen Bild-Pfad anstelle des Video-Index
        self.declare_parameter('image_folder', '/auv_ws/src/velocity_estimator/test_images')
        self.declare_parameter('frequency', 30.0)

        self.image_folder = self.get_parameter('image_folder').get_parameter_value().string_value
        frequency = self.get_parameter('frequency').get_parameter_value().double_value

        self.get_logger().info(f'Starting image stream from folder: {self.image_folder} at {frequency}Hz')

        self.publisher_ = self.create_publisher(Image, 'camera_feed/image_raw', 10)
        
        # Finde alle Bilder im Ordner (unterstützt .png, .jpg, .jpeg)
        extensions = ('*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG')
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(glob(os.path.join(self.image_folder, ext)))
        
        # Wichtig: Alphabetisch sortieren, damit die Bildreihenfolge stimmt
        self.image_paths.sort()
        self.current_image_index = 0

        if not self.image_paths:
            self.get_logger().error(f'No images found in folder: {self.image_folder}')
            return
        
        self.get_logger().info(f'Found {len(self.image_paths)} images to stream.')

        self.bridge = CvBridge()
        
        # Timer für die Frequenz starten
        timer_period = 1.0 / frequency 
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        # 1. Aktuellen Pfad holen
        img_path = self.image_paths[self.current_image_index]
        
        # 2. Bild einlesen
        frame = cv2.imread(img_path)
        
        if frame is not None:
            # 3. Konvertieren und publizieren
            msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "camera_frame"
            self.publisher_.publish(msg)
        else:
            self.get_logger().warn(f'Failed to load image: {img_path}')

        # 4. Index hochzählen (und am Ende wieder von vorne anfangen -> Loop)
        self.current_image_index = (self.current_image_index + 1) % len(self.image_paths)


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
