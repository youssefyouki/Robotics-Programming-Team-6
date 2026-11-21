#!/usr/bin/env python3
"""
Motion manager: sorting task planner + manual motion commands (Milestone 04).

Streams the reference for every simulator state. What it plays depends on the last command:

  start_sorting  plan the colour-sorting cycle for the boxes that are not yet in their bin
                 (box poses are read from the simulator) and run it
  goto_joint     smooth quintic move to joint angles          {"q": [5 rad], "grip": rad, "T": s}
  goto_tcp       move the tool point to a Cartesian position  {"p": [x,y,z], "pitch": rad,
                 (IK, joint-space quintic or straight line)    "roll": rad|null, "grip": rad, "linear": bool}
  jog            relative TCP step in the world frame         {"d": [dx,dy,dz]}
  gripper        move only the gripper                        {"value": rad}
  home           go to the home pose
  hold           decelerate smoothly and keep the pose

Manual commands start from the current reference state, so they can interrupt a running plan
without jumps. With auto_start=true the sorting cycle starts by itself (headless evaluation).

  Subscribes : m4/state, m4/boxes, m4/boxes_initial, m4/command
  Publishes  : m4/reference [t, q_ref, qd_ref, qdd_ref, gripper_ref, phase], m4/status (String),
               m4/done (Bool, when the sorting cycle has finished)
  Parameters : auto_start (bool), tail (s after the cycle before `done`)
"""
import json
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import Bool, Float64MultiArray, String

from milestone_03 import kinematics as K
from milestone_04 import cell as C, planning as P, tcp_kinematics as TK, topics as T


class SortingNode(Node):
    def __init__(self):
        super().__init__("sorting_node")
        self.declare_parameter("auto_start", True)
        self.declare_parameter("tail", 1.5)
        self.exec = None
        self.boxes = None                    # colour -> (position, yaw, in_correct_bin)
        self.t_now = 0.0
        self.epoch = None
        self.auto_end = None                 # sim time at which the sorting plan ends
        self.done_sent = True
        self.last_phase = None
        self.goal_tcp = None
        self.pending_commands = []
        self.token = 0                       # bumped by every command / reset; stale background plans are dropped
        self.new_plan = None                 # (token, plan or error text) from the planning thread
        self.auto_pending = self.get_parameter("auto_start").value

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.pub_ref = self.create_publisher(Float64MultiArray, T.REFERENCE, T.QOS_DEPTH)
        self.pub_status = self.create_publisher(String, T.STATUS, 10)
        self.pub_done = self.create_publisher(Bool, T.DONE, latched)
        self.create_subscription(Float64MultiArray, T.BOXES_INITIAL, self._on_boxes, latched)
        self.create_subscription(Float64MultiArray, T.BOXES, self._on_boxes, 10)
        self.create_subscription(Float64MultiArray, T.STATE, self._on_state, T.QOS_DEPTH)
        self.create_subscription(String, T.COMMAND, self._on_command, 20)
        self.get_logger().info("Motion manager waiting for the first simulator state.")

    # ------------------------------------------------------------------ inputs
    def _on_boxes(self, msg):
        d = msg.data
        self.boxes = {c: (np.array(d[1 + 5 * i:4 + 5 * i]), d[4 + 5 * i], bool(d[5 + 5 * i]))
                      for i, c in enumerate(C.COLORS)}

    def _status(self, text, level="info"):
        self.pub_status.publish(String(data=text))
        if level == "warn":
            self.get_logger().warn(text)
        else:
            self.get_logger().info(text)

    def _on_state(self, msg):
        t, q, grip_q = msg.data[0], np.array(msg.data[1:6]), msg.data[11]
        epoch = msg.data[12] if len(msg.data) > 12 else 0
        self.t_now = t
        if self.exec is None or epoch != self.epoch:        # first state, or the simulation was reset
            first = self.exec is None
            self.exec = P.Executor(q, grip_q)
            self.epoch, self.auto_end, self.goal_tcp, self.last_phase = epoch, None, None, None
            self.token += 1
            if not first:
                self._status("simulation reset - holding the home pose")
        if self.auto_pending and self.boxes is not None:
            self.auto_pending = False
            self._start_sorting()
        while self.pending_commands:
            self._handle(self.pending_commands.pop(0))
        if self.new_plan is not None:
            token, result = self.new_plan
            self.new_plan = None
            if token == self.token:
                if isinstance(result, str):
                    self._status(f"planning failed: {result}", "warn")
                else:
                    self.exec.set_plan(result, self.t_now)
                    self.auto_end = self.t_now + result.duration
                    self.done_sent = False
                    self._status(f"sorting plan ready: {result.duration:.1f} s")

        q_ref, qd_ref, qdd_ref, grip, name, color = self.exec.sample(t)
        if self.auto_end is None and name == "done":
            name = "manual"
        self.pub_ref.publish(T.pack([t, *q_ref, *qd_ref, *qdd_ref, grip, T.PHASES.index(name)]))

        if (name, color) != self.last_phase:
            self.last_phase = (name, color)
            if self.auto_end is not None or name != "manual":
                self.pub_status.publish(String(data=f"{color or '-'}: {name}"))
                self.get_logger().info(f"t={t:6.2f}s  {color or '-'}: {name}")
        if self.auto_end is not None and t > self.auto_end + self.get_parameter("tail").value \
                and not self.done_sent:
            self.done_sent = True
            self.auto_end = None
            self.pub_done.publish(Bool(data=True))
            self._status("sorting cycle finished")

    def _on_command(self, msg):
        self.pending_commands.append(msg.data)       # handled on the next state (same time base)

    # ------------------------------------------------------------------ commands
    def _handle(self, raw):
        try:
            cmd = json.loads(raw)
            name = cmd["cmd"]
        except (ValueError, KeyError):
            self._status(f"bad command: {raw}", "warn")
            return
        q0, v0, a0, g0 = self.exec.last
        try:
            if name == "start_sorting":
                self._start_sorting()
                return
            self.auto_end = None
            self.token += 1
            if name == "hold":
                plan = P.plan_stop(q0, v0, a0, g0)
            elif name == "goto_joint":
                q1 = np.array(cmd["q"], float) if cmd.get("q") is not None else q0
                plan = P.plan_joint_goto(q0, v0, a0, q1, g0, cmd.get("grip", g0), cmd.get("T"))
            elif name == "gripper":
                plan = P.plan_joint_goto(q0, v0, a0, q0, g0, cmd["value"], 0.8)
            elif name == "goto_tcp":
                roll = cmd.get("roll")
                plan = P.plan_tcp_goto(q0, v0, a0, np.array(cmd["p"], float),
                                       cmd.get("pitch", -np.pi / 2), q0[4] if roll is None else roll,
                                       g0, cmd.get("grip", g0), cmd.get("T"), cmd.get("linear", False))
                self.goal_tcp = np.array(cmd["p"], float)
            elif name == "jog":
                base = self.goal_tcp if (self.exec.active(self.t_now) and self.goal_tcp is not None) \
                    else TK.tcp_position(q0)
                target = base + np.array(cmd["d"], float)
                plan = P.plan_tcp_goto(q0, v0, a0, target, TK.tool_pitch(q0), q0[4], g0, g0,
                                       cmd.get("T", 0.6))
                self.goal_tcp = target
            elif name == "home":
                q_home = P.home_pose(q0)
                plan = P.plan_joint_goto(q0, v0, a0, q_home, g0, g0)
            else:
                self._status(f"unknown command '{name}'", "warn")
                return
        except P.PlanError as exc:
            self._status(f"cannot do that: {exc}", "warn")
            return
        self.exec.set_plan(plan, self.t_now)
        self._status(f"manual: {name}")

    def _start_sorting(self):
        if self.boxes is None:
            self._status("no box information from the simulator yet", "warn")
            return
        todo = [c for c in C.COLORS if not self.boxes[c][2]]
        if not todo:
            self._status("all boxes are already in their bins - reset the simulation to sort again")
            return
        q0, v0, a0, g0 = self.exec.last
        self.token += 1
        token = self.token
        boxes = {c: (p, y) for c, (p, y, _) in self.boxes.items()}
        self._status(f"planning the sorting cycle for {', '.join(todo)} - the robot stays live meanwhile...")

        def worker():
            try:
                self.new_plan = (token, P.build_sorting_plan(q0, boxes, grip_start=g0, order=todo))
            except P.PlanError as exc:
                self.new_plan = (token, str(exc))
        threading.Thread(target=worker, daemon=True).start()


def main(args=None):
    rclpy.init(args=args)
    node = SortingNode()
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
