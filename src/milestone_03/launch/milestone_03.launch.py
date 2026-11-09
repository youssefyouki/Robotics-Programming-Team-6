"""
Milestone 03 launch file: simulator + kinematics nodes + validation.

  ros2 launch milestone_03 milestone_03.launch.py                       # kinematic validation, circle
  ros2 launch milestone_03 milestone_03.launch.py mode:=dynamic          # physics + servo tracking
  ros2 launch milestone_03 milestone_03.launch.py trajectory:=pick_place mode:=dynamic
  ros2 launch milestone_03 milestone_03.launch.py viewer:=false          # headless

The nodes import `mujoco`; if it is installed in the project venv, run first:
  export PYTHONPATH=$PWD/.venv/lib/python3.12/site-packages:$PYTHONPATH
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.actions import RegisterEventHandler, EmitEvent
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    mode = LaunchConfiguration("mode")
    kinematic = ParameterValue(PythonExpression(["'", mode, "' == 'kinematic'"]), value_type=bool)
    viewer = ParameterValue(LaunchConfiguration("viewer"), value_type=bool)
    environment = ParameterValue(LaunchConfiguration("environment"), value_type=bool)
    loop = ParameterValue(LaunchConfiguration("loop"), value_type=bool)

    sim = Node(package="milestone_03", executable="sim_node", output="screen",
               parameters=[{"kinematic_mode": kinematic, "viewer": viewer,
                            "environment": environment}])
    forward = Node(package="milestone_03", executable="velocity_kinematics_node", output="screen")
    inverse = Node(package="milestone_03", executable="inverse_kinematics_node", output="screen")
    trajectory = Node(package="milestone_03", executable="trajectory_node", output="screen",
                      parameters=[{"trajectory": LaunchConfiguration("trajectory"), "loop": loop}])
    validation = Node(package="milestone_03", executable="validation_node", output="screen",
                      condition=IfCondition(LaunchConfiguration("validate")),
                      parameters=[{"check_inverse": kinematic,
                                   "output_dir": LaunchConfiguration("output_dir")}])

    return LaunchDescription([
        DeclareLaunchArgument("mode", default_value="kinematic",
                              description="kinematic (impose q,qd,qdd) or dynamic (physics + servos)"),
        DeclareLaunchArgument("trajectory", default_value="circle",
                              description="circle or pick_place"),
        DeclareLaunchArgument("viewer", default_value="true"),
        DeclareLaunchArgument("environment", default_value="true"),
        DeclareLaunchArgument("loop", default_value="false",
                              description="repeat pick_place forever"),
        DeclareLaunchArgument("validate", default_value="true",
                              description="run the validation node and exit when it finishes"),
        DeclareLaunchArgument("output_dir", default_value="milestone_03_results"),
        sim, forward, inverse, trajectory, validation,
        RegisterEventHandler(OnProcessExit(target_action=validation,
                                           on_exit=[EmitEvent(event=Shutdown())])),
    ])
