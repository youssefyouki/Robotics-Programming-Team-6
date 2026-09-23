import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    directory = get_package_share_directory('mujoco_ros2')
    xml_scene_path = os.path.join(directory, 'model', 'universal_robots_ur10e', 'scene.xml')

    if not os.path.exists(xml_scene_path):
        raise FileNotFoundError(f"Scene file does not exist: {xml_scene_path}.")

    ur10e = Node(
        package='mujoco_ros2',
        executable='fr3_node',
        name='ur10e_node',
        output='screen',
        arguments=[xml_scene_path],
        parameters=[
            {'joint_state_topic_name': 'ur10e/joint_state'},
            {'joint_command_topic_name': 'ur10e/joint_commands'},
            {'control_mode': 'VELOCITY'},
            {'simulation_frequency': 1000},
            {'visualisation_frequency': 20},
            {'camera_focal_point': [0.0, 0.0, 0.5]},
            {'camera_distance': 3.0},
            {'camera_azimuth': 135.0},
            {'camera_elevation': -20.0},
            {'camera_orthographic': True},
        ],
    )

    motion = Node(
        package='mujoco_ros2',
        executable='ur10e_motion',
        name='ur10e_motion',
        output='screen',
    )

    return LaunchDescription([ur10e, motion])
