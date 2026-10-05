from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ur3_llm_control',
            executable='skill_executor',
            name='skill_executor_node',
            output='screen',
            emulate_tty=True
        )
    ])