import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3Stamped
import csv
import os

class Logger(Node):
    def __init__(self):
        super().__init__('logger')
        
        self.csv_filename = 'velocity_log.csv'
        self.init_csv()

        # Subscribe to the estimation topic
        self.subscription = self.create_subscription(
            Vector3Stamped,
            'velocity_estimation/result',
            self.listener_callback,
            10)
        
        self.get_logger().info(f'Velocity Logger node started. Saving to {os.path.abspath(self.csv_filename)}')

    def init_csv(self):
        # Create file and write headers if it doesn't exist yet
        if not os.path.exists(self.csv_filename):
            with open(self.csv_filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'velocity_m_s'])

    def listener_callback(self, msg):
        # Compute exact float timestamp
        timestamp = msg.header.stamp.sec + (msg.header.stamp.nanosec * 1e-9)
        velocity = msg.vector.x
        
        try:
            with open(self.csv_filename, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([f"{timestamp:.6f}", f"{velocity:.4f}"])
            
            self.get_logger().debug(f'Logged: {timestamp:.4f}, {velocity:.2f}')
        except Exception as e:
            self.get_logger().error(f'Failed to write to CSV: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    logger_node = Logger()
    try:
        rclpy.spin(logger_node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            logger_node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
