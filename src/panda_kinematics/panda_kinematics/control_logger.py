#!/usr/bin/env python3
"""
Control logger for Challenge 3 - records how well the torque controller tracks the trajectory.

For every torque command published by controller_node (message k) it stores the desired joint state of step k
(rebuilt with the same plan_trajectory the node uses), the ACTUAL joint state last reported on /joint_state and the
commanded torques. After the trajectory plus a short settling time it writes metrics.txt, tracking.csv and
tracking.png into <output_dir>/ and exits.

  ros2 run panda_kinematics control_logger --ros-args -p output_dir:=challenge3_results/run1
"""
import os

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from panda_kinematics.controller_node import (Q_END, Q_START, TRAJ_DT, TRAJ_DURATION, V_MAX, TAU_MAX,
                                              compute_dynamics, forward_kinematics, plan_trajectory)

SETTLE_TIME = 1.5        # s of extra recording after the last planned sample
SKIP = 0.3               # s: "after the start-up transient" statistics ignore the first SKIP seconds


class ControlLogger(Node):
    def __init__(self):
        super().__init__("control_logger")
        self.declare_parameter("output_dir", "challenge3_results/run")
        self.out = self.get_parameter("output_dir").value
        self.t_plan, self.q_plan, self.dq_plan, self.ddq_plan = plan_trajectory(Q_START, Q_END, TRAJ_DURATION, TRAJ_DT)
        self.n = len(self.q_plan)                       # motion + the controlled stop appended by the planner
        moving = np.where(np.abs(self.dq_plan).max(axis=1) > 0)[0]
        self.n_motion = int(moving[-1]) + 2             # samples of the motion itself (the last one has zero speed)
        self.n_total = self.n + int(SETTLE_TIME / TRAJ_DT)
        self.q_act, self.dq_act = np.zeros(7), np.zeros(7)
        self.rows = []                                   # (wall, k, q_act[7], dq_act[7], tau[7])
        self.t0, self.done = None, False
        self.create_subscription(JointState, "joint_state", self.on_state, 10)
        self.create_subscription(Float64MultiArray, "joint_commands", self.on_command, 1000)
        self.get_logger().info(f"logging {self.n_total} torque commands ({self.n} planned samples)")

    def on_state(self, msg):
        self.q_act, self.dq_act = np.array(msg.position[:7]), np.array(msg.velocity[:7])

    def on_command(self, msg):
        if self.done:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        self.t0 = now if self.t0 is None else self.t0
        self.rows.append((now - self.t0, len(self.rows), *self.q_act, *self.dq_act, *msg.data[:7]))
        if len(self.rows) >= self.n_total:
            self.done = True
            self.finalise()

    def finalise(self):
        os.makedirs(self.out, exist_ok=True)
        r = np.array(self.rows)
        wall, k = r[:, 0], r[:, 1].astype(int)
        q, dq, tau = r[:, 2:9], r[:, 9:16], r[:, 16:23]
        kk = np.minimum(k, self.n - 1)
        q_d, dq_d = self.q_plan[kk], self.dq_plan[kk]
        t = k * TRAJ_DT
        np.savetxt(os.path.join(self.out, "tracking.csv"), np.column_stack([wall, t, q_d, q, dq_d, dq, tau]), delimiter=",",
                   header="wall_s,plan_s," + ",".join(f"q_des{i+1}" for i in range(7)) + "," + ",".join(f"q{i+1}" for i in range(7))
                          + "," + ",".join(f"dq_des{i+1}" for i in range(7)) + "," + ",".join(f"dq{i+1}" for i in range(7))
                          + "," + ",".join(f"tau{i+1}" for i in range(7)))
        err = np.degrees(q - q_d)
        mv = np.where(k < self.n_motion)[0]                # the planned motion (without the stop)
        late = mv[mv >= int(SKIP / TRAJ_DT)]
        ee_err = np.array([np.linalg.norm(forward_kinematics(a)[:3, 3] - forward_kinematics(b)[:3, 3]) for a, b in zip(q, q_d)]) * 1000
        final = np.degrees(q[-10:].mean(axis=0) - self.q_plan[-1])
        # inertial + Coriolis torque against gravity torque along the plan
        ratios = []
        for a, b, c in zip(self.q_plan[:self.n_motion], self.dq_plan[:self.n_motion], self.ddq_plan[:self.n_motion]):
            _, cg = compute_dynamics(a, b, c)
            g = compute_dynamics(a, np.zeros(7), c)[1]
            M, _ = compute_dynamics(a, b, c)
            ratios.append(np.linalg.norm(M @ c + cg - g) / np.linalg.norm(g))
        m = dict(rms=np.sqrt((err[mv] ** 2).mean()), mx=np.abs(err[mv]).max(), rms_late=np.sqrt((err[late] ** 2).mean()),
                 mx_late=np.abs(err[late]).max(), pj=np.sqrt((err[mv] ** 2).mean(axis=0)), pj_late=np.sqrt((err[late] ** 2).mean(axis=0)),
                 ee_rms=np.sqrt((ee_err[mv] ** 2).mean()), ee_max=ee_err[mv].max(), final=final, rate=(self.n - 1) / wall[self.n - 1], rest=np.abs(dq[-10:]).max(),
                 dtau=np.abs(np.diff(tau[:self.n], axis=0)).max(), tau_ratio=(np.abs(tau[:self.n]).max(axis=0) / TAU_MAX).max(),
                 speed=(np.abs(dq_d).max(axis=0) / V_MAX).max(), ratio_mean=float(np.mean(ratios)), ratio_max=float(np.max(ratios)))
        f = lambda a: np.array2string(np.asarray(a), precision=2)
        lines = [f"publish rate achieved: {m['rate']:.0f} Hz (target {1 / TRAJ_DT:.0f})",
                 f"motion duration: {self.t_plan[self.n_motion - 1]:.2f} s; fastest joint at {100 * m['speed']:.0f} % of its speed limit",
                 f"inertial+Coriolis / gravity torque (norm over joints): mean {m['ratio_mean']:.2f}, max {m['ratio_max']:.2f}",
                 f"tracking error (all joints): rms {m['rms']:.2f} deg, max {m['mx']:.2f} deg",
                 f"tracking error after the first {SKIP} s: rms {m['rms_late']:.2f} deg, max {m['mx_late']:.2f} deg",
                 "per-joint rms error [deg]: " + f(m["pj"]),
                 "per-joint rms error after start-up [deg]: " + f(m["pj_late"]),
                 f"hand position error: rms {m['ee_rms']:.1f} mm, max {m['ee_max']:.1f} mm",
                 "final joint error vs goal [deg]: " + f(m["final"]),
                 f"residual joint speed at the end of the recording: {m['rest']:.3f} rad/s",
                 f"peak commanded torque: {100 * m['tau_ratio']:.0f} % of the actuator limit; largest torque step between commands {m['dtau']:.1f} Nm"]
        open(os.path.join(self.out, "metrics.txt"), "w").write("\n".join(lines) + "\n")
        for line in lines:
            self.get_logger().info(line)
        fig, ax = plt.subplots(2, 2, figsize=(14, 8))
        for j in range(7):
            ax[0, 0].plot(t, np.degrees(q_d[:, j]), color=f"C{j}", lw=1.4)
            ax[0, 0].plot(t, np.degrees(q[:, j]), color=f"C{j}", ls="--", lw=1.2)
            ax[0, 1].plot(t, err[:, j], label=f"q{j+1}")
            ax[1, 0].plot(t, tau[:, j], label=f"tau{j+1}")
        ax[0, 0].set(title="desired (solid) vs actual (dashed) joint angles", xlabel="time [s]", ylabel="deg")
        ax[0, 1].set(title="joint error (actual - desired)", xlabel="time [s]", ylabel="deg"); ax[0, 1].legend(fontsize=7, ncol=4)
        ax[1, 0].set(title="commanded torques", xlabel="time [s]", ylabel="Nm"); ax[1, 0].legend(fontsize=7, ncol=4)
        ax[1, 1].plot(t, ee_err); ax[1, 1].set(title="hand position error", xlabel="time [s]", ylabel="mm")
        for a in ax.flat:
            a.grid(alpha=0.3)
            a.axvline(self.t_plan[self.n_motion - 1], color="k", ls=":", lw=1)
        fig.tight_layout(); fig.savefig(os.path.join(self.out, "tracking.png"), dpi=130); plt.close(fig)
        self.get_logger().info(f"results saved to {self.out}")


def main(args=None):
    rclpy.init(args=args)
    node = ControlLogger()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
