#!/usr/bin/env python3
"""
Motor-level joint controller node (Milestone 04).

Pairs the simulator state and the planner reference with the same time stamp, evaluates the
selected control law (see control.py) and answers with the actuator command for that stamp. The
controller can be changed while the system runs; the command carries the actuator mode (position
target or torque), so the simulator switches at exactly the same step.

It also monitors the tracking: live errors at 10 Hz and a validation result for every finished
trajectory segment (see monitor.py).

  Subscribes : m4/state, m4/reference, m4/controller (String: servo | pd_gravity | computed_torque)
  Publishes  : m4/actuator_cmd [t, u(5), gripper_ref, kind], m4/controller_info (JSON, latched),
               m4/monitor [t, phase, max joint error, TCP error], m4/validation (JSON per segment)
  Parameters : controller (initial), tol_task_mm, tol_joint_rad
"""
import json
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray, String

from milestone_04 import control, topics as T
from milestone_04.monitor import LiveMonitor


class ControllerNode(Node):
    def __init__(self):
        super().__init__("controller_node")
        self.declare_parameter("controller", "computed_torque")
        self.declare_parameter("tol_task_mm", 3.0)
        self.declare_parameter("tol_joint_rad", 0.03)
        self.ctrls = {name: control.make_controller(name) for name in control.CONTROLLERS}   # all ready: switching is instant
        self.name = self.get_parameter("controller").value
        self.state, self.ref = {}, {}
        self.tick = 0
        self.monitor = LiveMonitor(self.get_parameter("tol_task_mm").value, self.get_parameter("tol_joint_rad").value)

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.pub = self.create_publisher(Float64MultiArray, T.ACTUATOR_CMD, T.QOS_DEPTH)
        self.pub_info = self.create_publisher(String, T.CONTROLLER_INFO, latched)
        self.pub_monitor = self.create_publisher(Float64MultiArray, T.MONITOR, 10)
        self.pub_valid = self.create_publisher(String, T.VALIDATION, 50)
        self.create_subscription(Float64MultiArray, T.STATE, self._on_state, T.QOS_DEPTH)
        self.create_subscription(Float64MultiArray, T.REFERENCE, self._on_ref, T.QOS_DEPTH)
        self.create_subscription(String, T.CONTROLLER, self._on_select, 10)
        self._publish_info()
        self.get_logger().info(f"Controller '{self.name}' ({self.ctrls[self.name].kind} commands) ready.")

    # ------------------------------------------------------------------ controller selection
    def _publish_info(self):
        self.pub_info.publish(String(data=json.dumps(self.ctrls[self.name].describe())))

    def _on_select(self, msg):
        name = msg.data.strip()
        if name not in self.ctrls:
            self.get_logger().warn(f"unknown controller '{name}'")
            return
        if name != self.name:
            self.name = name
            self._publish_info()
            self.get_logger().info(f"Controller switched to '{name}'.")

    # ------------------------------------------------------------------ control loop
    def _on_state(self, msg):
        self.state[msg.data[0]] = msg.data
        self._try(msg.data[0])

    def _on_ref(self, msg):
        self.ref[msg.data[0]] = msg.data
        self._try(msg.data[0])

    def _try(self, t):
        if t not in self.state or t not in self.ref:
            return
        s, r = self.state.pop(t), self.ref.pop(t)
        for old in [k for k in self.state if k < t - 1.0]:
            del self.state[old]
        for old in [k for k in self.ref if k < t - 1.0]:
            del self.ref[old]
        q, qd, grip_q = np.array(s[1:6]), np.array(s[6:11]), s[11]
        q_ref, qd_ref, qdd_ref, grip_ref, phase = (np.array(r[1:6]), np.array(r[6:11]),
                                                   np.array(r[11:16]), r[16], r[17])
        ctrl = self.ctrls[self.name]
        u = ctrl(q, qd, q_ref, qd_ref, qdd_ref, grip_q=grip_q)
        self.pub.publish(T.pack([t, *u, grip_ref, 1.0 if ctrl.kind == "torque" else 0.0]))

        live, event = self.monitor.update(t, q, q_ref, phase)
        self.tick += 1
        if self.tick % 25 == 0:
            self.pub_monitor.publish(T.pack([t, phase, live[0], live[1]]))
        if event is not None:
            event["controller"] = self.name
            self.pub_valid.publish(String(data=json.dumps(event)))


def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    except RuntimeError:
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
