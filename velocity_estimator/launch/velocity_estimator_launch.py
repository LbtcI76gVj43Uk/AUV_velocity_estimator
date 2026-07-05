from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition

def generate_launch_description():
    
    video_index_arg = DeclareLaunchArgument('video_index', default_value='0')
    freq_arg = DeclareLaunchArgument('freq', default_value='10.0')
    run_logger_arg = DeclareLaunchArgument(
        'run_logger', 
        default_value='false',
        description='Set to "true" to run the logger node alongside the estimator.'
    )
    
    return LaunchDescription([
        video_index_arg,
        freq_arg,
        run_logger_arg,
        
        Node(
            package='velocity_estimator',
            executable='camera_node',
            name='camera_publisher',
            parameters=[{
                'video_index': LaunchConfiguration('video_index'),
                'frequency': LaunchConfiguration('freq')
            }]
        ),
        
        Node(
            package='velocity_estimator',
            executable='estimator_node',
            name='velocity_estimator',
            output='screen'
        ),
        
        Node(
            package='velocity_estimator',
            executable='logger_node',
            name='velocity_logger',
            output='screen',
            condition=IfCondition(LaunchConfiguration('run_logger'))
        )
    ])
