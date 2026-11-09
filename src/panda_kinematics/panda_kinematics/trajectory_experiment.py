#!/usr/bin/env python3
"""
Challenge 2 EXPERIMENT RUNNER (extra, not part of the instructor's template).

The official, submit-ready file is trajectory_node.py (instructor template, EDIT section filled in).
This module re-uses its maths (Jacobian, Jacobian derivative, planner, guard) and adds what a clean
POSITION-vs-VELOCITY experiment needs:
  * a slow "prep" move from the simulator's start pose (all zeros) to Q_START, plus rests, so the
    real motion starts from the right pose in BOTH modes (in VELOCITY mode the bridge's target starts at 0),
  * waiting for the bridge before publishing,
  * the mode as a ROS parameter (control_mode) instead of commenting/uncommenting lines,
  * a 1 kHz timer (TRAJ_DT = 0.001).

    ros2 launch panda_kinematics challenge2.launch.py mode:=POSITION      # or VELOCITY
"""
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from panda_kinematics.trajectory_node import *            # noqa: F401,F403  (the maths and the variant)
from panda_kinematics.trajectory_node import _frames, _quintic   # noqa: F401

T_TRAJ = TRAJ_DURATION      # s, main motion (the planner may stretch it)
DT = TRAJ_DT                # s, 1 ms = 1 / simulation_frequency of the bridge
T_PREP = 6.0                # s, slow set-up move from all-zeros to Q_START
HOLD_START = 1.0            # s, rest at Q_START before the main motion
HOLD_END = 1.5              # s, rest at Q_END so the settled error can be measured

# ----------------------------------------------------------
# 7. FULL TIMELINE PLAYED BY THE NODE
# ----------------------------------------------------------
PHASES = ["prep", "hold_start", "main", "hold_end"]


def build_timeline():
    """
    Concatenate the four phases into one long trajectory sampled every DT seconds:

      prep       slow quintic from the simulator's start pose (all zeros) to Q_START
      hold_start stand still at Q_START
      main       the Team 6 move Q_START -> Q_END       <- the part we evaluate
      hold_end   stand still at Q_END (lets the tracking error settle)

    Why 'prep' starts from zeros: the simulator starts with every joint at 0, and in VELOCITY
    mode the bridge starts its internal position target at 0 as well and only ADDS velocity
    to it, so planning from zeros keeps the planned positions and the bridge's integrated
    target identical.
    Returns times, q, dq, ddq, phase_id (each of length N).
    """
    parts = []
    q_init = np.zeros(7)
    parts.append((0, _quintic(q_init, Q_START, max(T_PREP, minimum_time(q_init, Q_START)), DT)))
    n = int(round(HOLD_START / DT))
    parts.append((1, (np.arange(n) * DT, np.tile(Q_START, (n, 1)), np.zeros((n, 7)), np.zeros((n, 7)))))
    parts.append((2, plan_trajectory(Q_START, Q_END, T_TRAJ, DT)))
    q_goal = parts[-1][1][1][-1]                     # the guard may have shortened the move
    n = int(round(HOLD_END / DT))
    parts.append((3, (np.arange(n) * DT, np.tile(q_goal, (n, 1)), np.zeros((n, 7)), np.zeros((n, 7)))))

    q = np.vstack([p[1][1] for p in parts])
    dq = np.vstack([p[1][2] for p in parts])
    ddq = np.vstack([p[1][3] for p in parts])
    phase = np.concatenate([np.full(len(p[1][0]), p[0]) for p in parts])
    times = np.arange(len(q)) * DT
    return times, q, dq, ddq, phase


# =================================================================
# DO NOT MODIFY BELOW (INFRASTRUCTURE: topics, timer, publishing)
# =================================================================

GRIPPER_CTRL = 0.0      # gripper command, identical in both modes so the runs are comparable
CMD_SIZE = 9            # the bridge expects nq = 9 values: 7 arm joints + 2 gripper entries
START_DELAY = 1.0       # s to wait after the bridge is connected before the trajectory starts


class TrajectoryNode(Node):
    def __init__(self):
        super().__init__("trajectory_experiment")

        # POSITION or VELOCITY, must match the bridge's control_mode parameter
        self.declare_parameter("control_mode", "POSITION")
        self.mode = self.get_parameter("control_mode").value.upper()
        if self.mode not in ("POSITION", "VELOCITY"):
            raise ValueError("control_mode must be POSITION or VELOCITY")

        # Plan the whole timeline once, up front (so the 1 kHz loop only indexes arrays)
        (self.t_traj, self.q_traj, self.dq_traj,
         self.ddq_traj, self.phase) = build_timeline()
        self.step_idx = 0                            # which sample is published next
        self.n_steps = len(self.q_traj)

        # Report what was planned (limits used, singularity margin, ...)
        main = self.phase == 2
        info = check_trajectory(self.q_traj[main], self.dq_traj[main], self.ddq_traj[main])
        self.get_logger().info(
            f"mode={self.mode}  main move {main.sum() * DT:.2f} s  "
            f"hand {np.round(info['hand_start'], 3)} -> {np.round(info['hand_end'], 3)} m")
        self.get_logger().info(
            f"peak speed {np.round(100 * info['peak_speed_ratio'].max(), 0):.0f} % / "
            f"accel {np.round(100 * info['peak_accel_ratio'].max(), 0):.0f} % of joint limits, "
            f"min joint-limit margin {np.degrees(info['limit_margin_rad'].min()):.1f} deg, "
            f"min sigma_min(J) {info['sigma_min']:.3f}")

        # ROS interfaces (relative names, same as Challenge 1)
        self.got_state = False
        self.ready_since = None
        self.last_phase = -1
        self.create_subscription(JointState, "joint_state", self.on_joint_state, 10)
        self.cmd_pub = self.create_publisher(Float64MultiArray, "joint_commands", 10)
        self.create_timer(DT, self.tick)             # 1 kHz publishing loop

    def on_joint_state(self, msg):
        # We only need to know that the bridge is alive and publishing.
        self.got_state = True

    def tick(self):
        # Wait until the bridge publishes and has subscribed to our commands. Early VELOCITY
        # messages would otherwise be lost and the arm would end short of the goal.
        if self.ready_since is None:
            if self.got_state and self.cmd_pub.get_subscription_count() > 0:
                self.ready_since = self.get_clock().now()
            return
        waited = (self.get_clock().now() - self.ready_since).nanoseconds * 1e-9
        if waited < START_DELAY:
            return

        if self.step_idx < self.n_steps:
            if self.mode == "POSITION":
                q_target = self.q_traj[self.step_idx]       # planned joint positions
            else:
                q_target = self.dq_traj[self.step_idx]      # planned joint velocities
            ph = int(self.phase[self.step_idx])
            if ph != self.last_phase:
                self.get_logger().info(f"phase: {PHASES[ph]}")
                self.last_phase = ph
            self.step_idx += 1
        else:
            # Trajectory finished: keep holding the last pose (POSITION) / zero velocity (VELOCITY)
            q_target = self.q_traj[-1] if self.mode == "POSITION" else np.zeros(7)
            if self.last_phase != 4:
                self.get_logger().info("trajectory finished, holding.")
                self.last_phase = 4

        cmd = Float64MultiArray()
        cmd.data = q_target.tolist() + [GRIPPER_CTRL, 0.0]   # 7 joints + gripper + spare entry
        assert len(cmd.data) == CMD_SIZE
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
