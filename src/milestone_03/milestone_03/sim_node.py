#!/usr/bin/env python3
"""
Python MuJoCo simulator node for Koch v1.1 (Milestone 03).

Keeps the same topics as the C++ mujoco_ros2 node (joint_commands in, joint_state out) and
additionally publishes what MuJoCo itself senses at the end effector.

  Subscribes : low_cost_robot_arm/joint_commands       std_msgs/Float64MultiArray  (6 positions)
               low_cost_robot_arm/joint_kinematics_cmd Float64MultiArray [t, q, qd, qdd]
                                                       (only used in kinematic_mode)
  Publishes  : low_cost_robot_arm/joint_state          sensor_msgs/JointState
               low_cost_robot_arm/joint_kinematics     Float64MultiArray [t, q, qd, qdd]
               low_cost_robot_arm/ee/sim               Float64MultiArray [t, p, v, a] (sensed)

Modes
  dynamic   (default) : position actuators + physics, stepped in real time.
  kinematic           : joint position/velocity/acceleration are imposed directly from
                        joint_kinematics_cmd (no dynamics) - used to validate the equations.

Parameters: kinematic_mode (bool), environment (bool), viewer (bool), rate (Hz),
            scene_xml (str, default: installed mujoco_ros2 model)
"""
import numpy as np
import mujoco
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from milestone_03 import sim_model as S
from milestone_03 import topics as T


class SimNode(Node):
    def __init__(self):
        super().__init__("sim_node")
        self.declare_parameter("kinematic_mode", False)
        self.declare_parameter("environment", True)
        self.declare_parameter("viewer", True)
        self.declare_parameter("rate", 100.0)
        self.declare_parameter("scene_xml", "")

        self.kinematic = self.get_parameter("kinematic_mode").value
        rate = self.get_parameter("rate").value
        scene = self.get_parameter("scene_xml").value or None
        self.environment = self.get_parameter("environment").value

        self.model, self.data, self.site = S.load_model(self.environment, scene)
        self.dt = self.model.opt.timestep
        self.steps_per_tick = max(1, round(1.0 / rate / self.dt))
        self.ctrl = np.zeros(self.model.nu)
        self.joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                            for i in range(self.model.nq)]
        mujoco.mj_forward(self.model, self.data)

        self.pub_state = self.create_publisher(JointState, T.JOINT_STATE, T.QOS_DEPTH)
        self.pub_kin = self.create_publisher(Float64MultiArray, T.JOINT_KINEMATICS, T.QOS_DEPTH)
        self.pub_ee = self.create_publisher(Float64MultiArray, T.EE_SIM, T.QOS_DEPTH)
        self.create_subscription(Float64MultiArray, T.JOINT_COMMANDS, self._on_command, T.QOS_DEPTH)
        self.create_subscription(Float64MultiArray, T.JOINT_KINEMATICS_CMD,
                                 self._on_kinematic_command, T.QOS_DEPTH)

        self.viewer = None
        if self.get_parameter("viewer").value:
            from mujoco import viewer as mj_viewer
            self.viewer = mj_viewer.launch_passive(self.model, self.data)
            self.viewer.cam.lookat[:] = [-0.05, 0.05, 0.1]
            self.viewer.cam.distance, self.viewer.cam.azimuth, self.viewer.cam.elevation = 1.0, 215, -25

        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            f"MuJoCo simulator ready ({'kinematic' if self.kinematic else 'dynamic'} mode, "
            f"environment={self.environment}).")

    # ------------------------------------------------------------------ input
    def _on_command(self, msg):
        if len(msg.data) != self.model.nu:
            self.get_logger().warn(f"joint_commands needs {self.model.nu} values, got {len(msg.data)}",
                                   throttle_duration_sec=5.0)
            return
        self.ctrl[:] = msg.data
        if not self.kinematic:
            self.data.ctrl[:] = self.ctrl

    def _on_kinematic_command(self, msg):
        if not self.kinematic or len(msg.data) != 16:
            return
        t, q, qd, qdd = T.unpack_joint_kinematics(msg.data)
        d = self.data
        d.qpos[:5], d.qvel[:5] = q, qd
        d.qpos[5], d.qvel[5] = self.ctrl[5], 0.0          # gripper follows its last command
        mujoco.mj_forward(self.model, d)
        d.qacc[:5], d.qacc[5] = qdd, 0.0
        self._publish(t)

    # ------------------------------------------------------------------ simulation step
    def _tick(self):
        if self.viewer is not None and not self.viewer.is_running():
            self.get_logger().info("Viewer closed, shutting down.")
            rclpy.try_shutdown()
            return
        if self.environment:
            S.move_belt(self.model, self.data, self.steps_per_tick * self.dt)
        if not self.kinematic:
            for _ in range(self.steps_per_tick):
                mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)      # qacc consistent with the current state
            self._publish(self.data.time)
        if self.viewer is not None:
            self.viewer.sync()

    # ------------------------------------------------------------------ output
    def _publish(self, t):
        d, m = self.data, self.model

        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = self.joint_names
        js.position = d.qpos.tolist()
        js.velocity = d.qvel.tolist()
        js.effort = d.actuator_force.tolist()
        self.pub_state.publish(js)

        self.pub_kin.publish(T.pack([t, *d.qpos[:5], *d.qvel[:5], *d.qacc[:5]]))

        v, _ = S.site_velocity(m, d, self.site)
        a, _ = S.site_acceleration(m, d, self.site)
        self.pub_ee.publish(T.pack_ee(t, d.site_xpos[self.site], v, a))

    def close(self):
        if self.viewer is not None:
            self.viewer.close()


def main(args=None):
    rclpy.init(args=args)
    node = SimNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
