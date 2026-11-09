"""
Challenge 2 launch file: MuJoCo bridge + trajectory node + tracking logger, one control mode.

  ros2 launch panda_kinematics challenge2.launch.py mode:=POSITION
  ros2 launch panda_kinematics challenge2.launch.py mode:=VELOCITY
  ros2 launch panda_kinematics challenge2.launch.py mode:=POSITION logger:=false   # just watch
  ros2 launch panda_kinematics challenge2.launch.py mode:=VELOCITY vis_hz:=5        # slower rendering

Uses trajectory_experiment (extras around the same maths). To test the instructor's trajectory_node.py
itself: launch with experiment:=false template:=true, then run `ros2 run panda_kinematics trajectory_node`.

`mode` is given to BOTH the bridge (control_mode parameter, what the challenge text asks you to
edit in example.py) and the trajectory node, so they can never disagree. Same effect as editing
example.py + `colcon build --packages-select mujoco_ros2`, without touching any files.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    scene = os.path.join(get_package_share_directory("mujoco_ros2"),
                         "model", "franka_emika_panda", "scene.xml")
    mode = LaunchConfiguration("mode")

    bridge = Node(
        package="mujoco_ros2", executable="mujoco_node", output="screen", arguments=[scene],
        parameters=[{
            "joint_state_topic_name": "joint_state",
            "joint_command_topic_name": "joint_commands",
            "control_mode": mode,                       # POSITION or VELOCITY
            "simulation_frequency": 1000,               # must equal 1/DT in trajectory_node.py
            "visualisation_frequency": LaunchConfiguration("vis_hz"),
            "camera_focal_point": [0.3, 0.0, 0.5],
            "camera_distance": 2.2,
            "camera_azimuth": 135.0,
            "camera_elevation": -15.0,
            "camera_orthographic": True,
        }])

    trajectory = Node(package="panda_kinematics", executable="trajectory_experiment", output="screen",
                      condition=IfCondition(LaunchConfiguration("experiment")),
                      parameters=[{"control_mode": mode}])

    logger = Node(package="panda_kinematics", executable="tracking_logger", output="screen",
                  condition=IfCondition(LaunchConfiguration("logger")),
                  parameters=[{"control_mode": mode,
                               "template": ParameterValue(LaunchConfiguration("template"), value_type=bool),
                               "output_dir": LaunchConfiguration("output_dir")}])

    return LaunchDescription([
        DeclareLaunchArgument("mode", default_value="POSITION",
                              description="POSITION or VELOCITY (bridge and node)"),
        DeclareLaunchArgument("vis_hz", default_value="20",
                              description="bridge render rate in Hz (see report: it affects VELOCITY mode)"),
        DeclareLaunchArgument("logger", default_value="true",
                              description="record tracking errors and exit when the run is done"),
        DeclareLaunchArgument("output_dir", default_value="challenge2_results"),
        DeclareLaunchArgument("experiment", default_value="true",
                              description="true: run trajectory_experiment (prep move, 1 kHz, mode parameter); "
                                          "false: start the instructor's trajectory_node yourself"),
        DeclareLaunchArgument("template", default_value="false",
                              description="logger: the instructor's trajectory_node is the one running"),
        bridge, trajectory, logger,
        # when the logger has finished its run, stop everything
        RegisterEventHandler(OnProcessExit(target_action=logger, on_exit=[EmitEvent(event=Shutdown())])),
    ])
