"""
Launch the MuJoCo bridge for the Franka Panda in TORQUE mode (Challenge 3).

  ros2 launch mujoco_ros2 panda_torque.py

Differences from example.py (position control):
  * the unmodified Panda model (scene.xml): its actuators are position servos, so the bridge converts
    each commanded torque into the servo target that produces exactly that torque
  * control_mode TORQUE with torque_passthrough = true: the numbers on /joint_commands are applied
    as raw joint torques and HELD until the next message. (Without it the bridge adds gravity /
    Coriolis compensation and clears the torque after one 1 ms step.)
  * the command is 9 numbers: 7 joint torques [Nm], gripper (0..255, 255 = open), 1 spare.
"""
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    directory = get_package_share_directory('mujoco_ros2')
    xml_scene_path = os.path.join(directory, 'model', 'franka_emika_panda', 'scene.xml')

    if not os.path.exists(xml_scene_path):
        raise FileNotFoundError(f"Scene file does not exist: {xml_scene_path}.")

    mujoco = Node(
        package="mujoco_ros2",
        executable="mujoco_node",
        output="screen",
        arguments=[xml_scene_path],
        parameters=[
            {"joint_state_topic_name": "joint_state"},
            {"joint_command_topic_name": "joint_commands"},
            {"control_mode": "TORQUE"},
            {"torque_passthrough": True},
            {"simulation_frequency": 1000},
            {"visualisation_frequency": 20},
            {"camera_focal_point": [0.3, 0.0, 0.5]},
            {"camera_distance": 2.2},
            {"camera_azimuth": 135.0},
            {"camera_elevation": -15.0},
            {"camera_orthographic": True},
        ],
    )

    return LaunchDescription([mujoco])
