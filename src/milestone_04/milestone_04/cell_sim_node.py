#!/usr/bin/env python3
"""
MuJoCo simulator node for the sorting cell (Milestone 04, lock-step co-simulation).

  Publishes  : m4/state, m4/boxes (10 Hz), m4/boxes_initial (once), sensor_msgs/JointState
             m4/view_state [t, qpos...] (~40 Hz, rendered by view_node for the panel)
  Subscribes : m4/actuator_cmd [t, u(5), gripper], m4/sim_control
  Parameters : actuation ('position' | 'torque'), viewer (bool, native MuJoCo window), realtime (bool),
               speed (float), viewer_every (int, control ticks between viewer refreshes),
               min_subscribers (int, nodes that must listen to m4/state before starting),
               stream_image (bool, publish m4/view_state)

The first state is sent when the planner and the controller are connected. After every state the
node waits for the command with the same time stamp, applies it for one control period (2 physics
steps of 2 ms) and publishes the next state.
"""
import json
import time
import numpy as np
import mujoco
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String

from milestone_04 import cell as C, topics as T
from milestone_04.offline import CTRL_STEPS


class CellSimNode(Node):
    def __init__(self):
        super().__init__("cell_sim_node")
        self.declare_parameter("actuation", "torque")
        self.declare_parameter("viewer", True)
        self.declare_parameter("realtime", True)
        self.declare_parameter("speed", 1.0)
        self.declare_parameter("viewer_every", 4)
        self.declare_parameter("min_subscribers", 3)     # planner, controller, evaluation
        self.declare_parameter("stream_image", False)     # publish m4/view_state for the view node

        self.stream = self.get_parameter("stream_image").value
        self.model, self.data = C.load_cell(actuation=self.get_parameter("actuation").value)
        C.reset_to_home(self.model, self.data)
        self.joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                            for i in range(C.ARM_JOINTS + 1)]
        self.tick = 0
        self.paused = False
        self.held_cmd = None
        self.epoch = 0
        self.expected_t = None
        self.last_state_wall = None
        self.started = False
        self.wall0 = None
        self.realtime = self.get_parameter("realtime").value
        self.speed = self.get_parameter("speed").value
        self.viewer_every = self.get_parameter("viewer_every").value

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.pub_state = self.create_publisher(Float64MultiArray, T.STATE, T.QOS_DEPTH)
        self.pub_boxes = self.create_publisher(Float64MultiArray, T.BOXES, 10)
        self.pub_boxes0 = self.create_publisher(Float64MultiArray, T.BOXES_INITIAL, latched)
        self.pub_js = self.create_publisher(JointState, T.JOINT_STATE, 10)
        self.create_subscription(Float64MultiArray, T.ACTUATOR_CMD, self._on_cmd, T.QOS_DEPTH)
        self.create_subscription(String, T.SIM_CONTROL, self._on_sim_control, 10)

        self.viewer = None
        if self.get_parameter("viewer").value:
            from mujoco import viewer as mj_viewer
            self.viewer = mj_viewer.launch_passive(self.model, self.data)
            self.viewer.cam.lookat[:] = [-0.01, 0.0, 0.04]
            self.viewer.cam.distance, self.viewer.cam.azimuth, self.viewer.cam.elevation = 0.6, 215, -35

        if self.stream:
            self.pub_view = self.create_publisher(Float64MultiArray, T.VIEW_STATE, qos_profile_sensor_data)
            self._publish_view()

        self.pub_boxes0.publish(self._boxes_msg())
        self.start_timer = self.create_timer(0.2, self._try_start)
        self.watchdog = self.create_timer(3.0, self._watchdog)
        self.get_logger().info("Sorting cell ready; waiting for planner and controller.")

    # ------------------------------------------------------------------ lock-step handshake
    def _try_start(self):
        if self.count_subscribers(T.STATE) >= self.get_parameter("min_subscribers").value and self.count_publishers(T.ACTUATOR_CMD) >= 1 \
                and self.count_publishers(T.REFERENCE) >= 1:
            self.start_timer.cancel()
            self.started = True
            self.wall0 = time.time()
            self.get_logger().info("All nodes connected, starting the simulation.")
            self._publish_state()

    def _watchdog(self):
        if self.started and not self.paused and self.last_state_wall \
                and time.time() - self.last_state_wall > 3.0:
            self.get_logger().warn("No command received, re-sending the state.")
            self._publish_state()

    def _on_cmd(self, msg):
        if not self.started or self.expected_t is None or abs(msg.data[0] - self.expected_t) > 1e-9:
            return
        if self.paused:                       # keep the command; it is applied on resume
            self.held_cmd = msg
            return
        u = np.array(msg.data[1:6])
        self.data.ctrl[:C.ARM_JOINTS] = u
        self.data.ctrl[C.ARM_JOINTS] = msg.data[6]
        for _ in range(CTRL_STEPS):
            mujoco.mj_step(self.model, self.data)
        self.tick += 1
        if self.viewer is not None and self.tick % self.viewer_every == 0:
            self.viewer.sync()
            if not self.viewer.is_running():
                rclpy.try_shutdown()
                return
        if self.realtime:
            ahead = self.data.time - (time.time() - self.wall0) * self.speed
            if ahead > 0.002:
                time.sleep(ahead)
        self._publish_state()

    def _publish_view(self):
        """Joint positions of everything (arm, boxes) for the view node, which renders them."""
        self.pub_view.publish(T.pack([self.data.time, *self.data.qpos]))

    # ------------------------------------------------------------------ run / pause / reset / speed
    def _on_sim_control(self, msg):
        try:
            cmd = json.loads(msg.data)
            name = cmd["cmd"]
        except (ValueError, KeyError):
            return
        if name == "pause":
            self.paused = True
            self.get_logger().info("Simulation paused.")
        elif name == "resume":
            self.paused = False
            self._rebase_clock()
            self.get_logger().info("Simulation running.")
            if self.held_cmd is not None:
                held, self.held_cmd = self.held_cmd, None
                self._on_cmd(held)
        elif name == "reset":
            self._reset()
        elif name == "speed":
            self.speed = max(0.05, float(cmd["value"]))
            self._rebase_clock()
            self.get_logger().info(f"Simulation speed x{self.speed:g}.")

    def _rebase_clock(self):
        self.wall0 = time.time() - self.data.time / self.speed

    def _reset(self):
        t_old = self.data.time
        mujoco.mj_resetData(self.model, self.data)
        self.data.time = t_old + 1e-3          # new time stamp: commands of the old run are ignored
        mujoco.mj_forward(self.model, self.data)
        self.epoch += 1
        self.held_cmd = None
        self.tick = 0
        self._rebase_clock()
        if self.viewer is not None:
            self.viewer.sync()
        self.get_logger().info("Simulation reset.")
        self._publish_state()

    # ------------------------------------------------------------------ output
    def _boxes_msg(self):
        vals = [self.data.time]
        for color, (pos, yaw) in C.box_poses(self.model, self.data).items():
            vals += [*pos, yaw, float(C.box_in_bin(pos, color))]
        return T.pack(vals)

    def _publish_state(self):
        d = self.data
        if self.stream and self.tick % 6 == 0:                  # ~40 Hz at the 250 Hz control rate
            self._publish_view()
        self.expected_t = d.time
        self.last_state_wall = time.time()
        self.pub_state.publish(T.pack([d.time, *d.qpos[:5], *d.qvel[:5], d.qpos[5], self.epoch]))
        if self.tick % 25 == 0:
            self.pub_boxes.publish(self._boxes_msg())
            js = JointState()
            js.header.stamp = self.get_clock().now().to_msg()
            js.name = self.joint_names
            js.position = d.qpos[:C.ARM_JOINTS + 1].tolist()
            js.velocity = d.qvel[:C.ARM_JOINTS + 1].tolist()
            self.pub_js.publish(js)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()


def main(args=None):
    rclpy.init(args=args)
    node = CellSimNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    except RuntimeError:
        if rclpy.ok():                      # a real error; during shutdown it is just a teardown race
            raise
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
