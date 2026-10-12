import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    directory = get_package_share_directory('mujoco_ros2')
    xml_scene_path = os.path.join(directory, 'model', 'low_cost_robot_arm', 'scene.xml')

    if not os.path.exists(xml_scene_path):
        raise FileNotFoundError(f"Scene file does not exist: {xml_scene_path}.")

    node = Node(
        package='mujoco_ros2',
        executable='low_cost_robot_arm_node',
        name='low_cost_robot_arm_node',
        output='screen',
        arguments=[xml_scene_path],
        parameters=[
            {'joint_state_topic_name': 'low_cost_robot_arm/joint_state'},
            {'joint_command_topic_name': 'low_cost_robot_arm/joint_commands'},
            {'control_mode': 'POSITION'},
            {'simulation_frequency': 1000},
            {'visualisation_frequency': 20},
            {'camera_focal_point': [0.0, 0.0, 0.1]},
            {'camera_distance': 1.0},
            {'camera_azimuth': 210.0},
            {'camera_elevation': -25.0},
            {'camera_orthographic': True},
        ],
    )

    return LaunchDescription([node])
