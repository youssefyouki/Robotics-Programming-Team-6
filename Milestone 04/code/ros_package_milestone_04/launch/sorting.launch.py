"""
Milestone 04 launch file: colour-sorting cell, planner, motor-level controller and control panel.

  ros2 launch milestone_04 sorting.launch.py                          # one window: live view + controls
  ros2 launch milestone_04 sorting.launch.py controller:=pd_gravity   # or servo | computed_torque
  ros2 launch milestone_04 sorting.launch.py panel:=false             # automatic run + evaluation report
  ros2 launch milestone_04 sorting.launch.py panel:=false viewer:=false realtime:=false   # fast headless

The simulation view is rendered by a separate view node and shown inside the panel window
(viewer:=true additionally opens the native MuJoCo window). With the panel the arm waits at home: use the panel to run/pause/reset the simulation, move the joints
or the tool point by hand, or start the sorting cycle. With panel:=false the sorting cycle starts by
itself and the evaluation node writes the results to output_dir when it finishes.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    controller = LaunchConfiguration("controller")
    panel = LaunchConfiguration("panel")
    actuation = PythonExpression(["'position' if '", controller, "' == 'servo' else 'torque'"])
    viewer = ParameterValue(LaunchConfiguration("viewer"), value_type=bool)
    realtime = ParameterValue(LaunchConfiguration("realtime"), value_type=bool)
    auto_start = ParameterValue(PythonExpression(["'", panel, "' != 'true'"]), value_type=bool)
    stream_image = ParameterValue(PythonExpression(["'", panel, "' == 'true'"]), value_type=bool)

    sim = Node(package="milestone_04", executable="cell_sim_node", output="screen",
               parameters=[{"actuation": actuation, "viewer": viewer, "realtime": realtime,
                            "stream_image": stream_image}])
    planner = Node(package="milestone_04", executable="sorting_node", output="screen",
                   parameters=[{"auto_start": auto_start}])
    ctrl = Node(package="milestone_04", executable="controller_node", output="screen",
                parameters=[{"controller": controller}])
    gui = Node(package="milestone_04", executable="control_panel", output="screen",
               condition=IfCondition(panel))
    view = Node(package="milestone_04", executable="view_node", output="screen",
                condition=IfCondition(panel))
    evaluation = Node(package="milestone_04", executable="evaluation_node", output="screen",
                      condition=UnlessCondition(panel),
                      parameters=[{"output_dir": LaunchConfiguration("output_dir"),
                                   "controller": controller}])
    return LaunchDescription([
        DeclareLaunchArgument("controller", default_value="computed_torque",
                              description="servo | pd_gravity | computed_torque"),
        DeclareLaunchArgument("panel", default_value="true",
                              description="true: manual control panel; false: automatic run + evaluation"),
        DeclareLaunchArgument("viewer", default_value="false",
                              description="also open the native MuJoCo window"),
        DeclareLaunchArgument("realtime", default_value="true"),
        DeclareLaunchArgument("output_dir", default_value="milestone_04_results"),
        sim, planner, ctrl, view, gui, evaluation,
        RegisterEventHandler(OnProcessExit(target_action=evaluation, on_exit=[EmitEvent(event=Shutdown())]),
                             condition=UnlessCondition(panel)),
        RegisterEventHandler(OnProcessExit(target_action=gui, on_exit=[EmitEvent(event=Shutdown())]),
                             condition=IfCondition(panel)),
    ])
