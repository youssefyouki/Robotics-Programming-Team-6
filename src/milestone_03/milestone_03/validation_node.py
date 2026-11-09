#!/usr/bin/env python3
"""
Kinematics validation node (Milestone 03).

Collects the three EE streams and compares them within a tolerance:

  forward check : ee/sim (sensed by MuJoCo)  vs  ee/equations (FK, J qd, J qdd + Jdot qd)
  inverse check : ee/desired                 vs  ee/sim   (kinematic mode only - in dynamic
                  mode the position servos lag, so the tracking error is only reported)

Samples are paired by their shared time stamp t. When the trajectory is finished the node
prints a PASS/FAIL table, saves plots + summary.txt to `output_dir` and exits
(exit code 0 = all checks passed).

  Subscribes : low_cost_robot_arm/ee/sim, ee/equations, ee/desired, trajectory_done
  Parameters : tol_pos [m], tol_vel [m/s], tol_acc [m/s^2], check_inverse (bool),
               output_dir (str), finish_delay (s), max_wait (s, 0 = unlimited)
"""
import os
import sys
import time
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float64MultiArray

from milestone_03 import topics as T


class ValidationNode(Node):
    def __init__(self):
        super().__init__("validation_node")
        self.declare_parameter("tol_pos", 1e-5)
        self.declare_parameter("tol_vel", 1e-5)
        self.declare_parameter("tol_acc", 1e-4)
        self.declare_parameter("check_inverse", True)
        self.declare_parameter("output_dir", "milestone_03_results")
        self.declare_parameter("finish_delay", 1.0)
        self.declare_parameter("max_wait", 0.0)

        self.sim, self.eq, self.des = {}, {}, {}
        self.track = []                      # (t, |p_sim - latest desired p|) for dynamic mode
        self.latest_des = None
        self.done_at = None
        self.started = time.time()
        self.passed = None

        qos = T.QOS_DEPTH
        self.create_subscription(Float64MultiArray, T.EE_SIM, self._on_sim, qos)
        self.create_subscription(Float64MultiArray, T.EE_EQ, self._on_eq, qos)
        self.create_subscription(Float64MultiArray, T.EE_DESIRED, self._on_des, qos)
        self.create_subscription(Bool, T.TRAJECTORY_DONE, self._on_done, 10)
        self.get_logger().info("Validation node collecting samples...")

    def _on_sim(self, msg):
        t = msg.data[0]
        self.sim[t] = np.array(msg.data[1:10])
        if self.latest_des is not None:
            self.track.append((t, np.linalg.norm(self.sim[t][:3] - self.latest_des[:3])))

    def _on_eq(self, msg):
        self.eq[msg.data[0]] = np.array(msg.data[1:10])

    def _on_des(self, msg):
        arr = np.array(msg.data[1:10])
        self.des[msg.data[0]] = arr
        self.latest_des = arr

    def _on_done(self, msg):
        self.done_at = time.time()
        self.get_logger().info("Trajectory finished, finalising shortly.")

    def should_finish(self):
        delay = self.get_parameter("finish_delay").value
        if self.done_at is not None and time.time() - self.done_at > delay:
            return True
        max_wait = self.get_parameter("max_wait").value
        return max_wait > 0 and time.time() - self.started > max_wait

    # ------------------------------------------------------------------ results
    def finalize(self):
        tol_p, tol_v, tol_a = (self.get_parameter(n).value for n in ("tol_pos", "tol_vel", "tol_acc"))
        out = os.path.abspath(self.get_parameter("output_dir").value)
        os.makedirs(out, exist_ok=True)

        lines, checks = [], []

        def row(name, errs, tol, unit):
            if len(errs) == 0:
                lines.append(f"{name:<34s} no paired samples")
                checks.append(False)
                return
            errs = np.asarray(errs)
            ok = bool(errs.max() < tol)
            checks.append(ok)
            lines.append(f"{name:<34s} n={len(errs):5d}  max={errs.max():.3e} {unit}  "
                         f"mean={errs.mean():.3e} {unit}  tol={tol:.0e}  {'PASS' if ok else 'FAIL'}")

        ts = sorted(set(self.sim) & set(self.eq))
        S = np.array([self.sim[t] for t in ts]).reshape(-1, 9)
        E = np.array([self.eq[t] for t in ts]).reshape(-1, 9)
        lines.append("--- forward: simulator-sensed vs equations ---")
        row("position  |sim - FK(q)|", np.linalg.norm(S[:, :3] - E[:, :3], axis=1), tol_p, "m")
        row("velocity  |sim - Jv qd|", np.linalg.norm(S[:, 3:6] - E[:, 3:6], axis=1), tol_v, "m/s")
        row("accel.    |sim - eq.|", np.linalg.norm(S[:, 6:9] - E[:, 6:9], axis=1), tol_a, "m/s^2")

        td = sorted(set(self.sim) & set(self.des))
        D = SD = np.zeros((0, 9))
        if self.get_parameter("check_inverse").value:
            D = np.array([self.des[t] for t in td]).reshape(-1, 9)
            SD = np.array([self.sim[t] for t in td]).reshape(-1, 9)
            lines.append("--- inverse: desired vs simulator-sensed ---")
            row("position  |desired - sim|", np.linalg.norm(D[:, :3] - SD[:, :3], axis=1), tol_p, "m")
            row("velocity  |desired - sim|", np.linalg.norm(D[:, 3:6] - SD[:, 3:6], axis=1), tol_v, "m/s")
            row("accel.    |desired - sim|", np.linalg.norm(D[:, 6:9] - SD[:, 6:9], axis=1), tol_a, "m/s^2")
        elif self.track:
            tr = np.array([e for _, e in self.track])
            lines.append("--- dynamic mode: tracking performance (informational) ---")
            lines.append(f"EE position tracking error         max={tr.max()*1000:.1f} mm  "
                         f"mean={tr.mean()*1000:.1f} mm  (servo lag, not a kinematics error)")

        self.passed = all(checks) and len(checks) > 0
        lines.append("ALL CHECKS PASSED" if self.passed else "SOME CHECKS FAILED")
        text = "\n".join(lines)
        print("\n" + text)
        with open(os.path.join(out, "summary.txt"), "w") as f:
            f.write(text + "\n")
        self._plots(out, ts, S, E, td, D, SD)
        print(f"results saved to {out}")

    def _plots(self, out, ts, S, E, td, D, SD):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:                          # plotting is optional
            self.get_logger().warn(f"matplotlib unavailable, skipping plots ({exc})")
            return
        if len(ts) == 0:
            return
        t0 = np.array(ts)
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
        for j, (title, unit, sl) in enumerate([("EE position", "m", slice(0, 3)),
                                               ("EE velocity", "m/s", slice(3, 6)),
                                               ("EE acceleration", "m/s^2", slice(6, 9))]):
            for i, ax_name in enumerate("xyz"):
                ax[j].plot(t0, E[:, sl][:, i], label=f"equation {ax_name}")
                ax[j].plot(t0, S[:, sl][:, i], "--", label=f"simulator {ax_name}")
            ax[j].set_title(title); ax[j].set_xlabel("t [s]"); ax[j].set_ylabel(unit)
            ax[j].legend(fontsize=7); ax[j].grid(alpha=.3)
        fig.suptitle("Forward kinematics: equations vs simulator-sensed end effector")
        fig.tight_layout(); fig.savefig(os.path.join(out, "forward_validation.png"), dpi=130)
        plt.close(fig)

        if len(td) and len(D):
            t1 = np.array(td)
            fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
            for j, (title, unit, sl) in enumerate([("EE position", "m", slice(0, 3)),
                                                   ("EE velocity", "m/s", slice(3, 6)),
                                                   ("EE acceleration", "m/s^2", slice(6, 9))]):
                for i, ax_name in enumerate("xyz"):
                    ax[j].plot(t1, D[:, sl][:, i], label=f"desired {ax_name}")
                    ax[j].plot(t1, SD[:, sl][:, i], "--", label=f"simulator {ax_name}")
                ax[j].set_title(title); ax[j].set_xlabel("t [s]"); ax[j].set_ylabel(unit)
                ax[j].legend(fontsize=7); ax[j].grid(alpha=.3)
            fig.suptitle("Inverse kinematics: desired vs simulator-sensed end effector")
            fig.tight_layout(); fig.savefig(os.path.join(out, "inverse_validation.png"), dpi=130)
            plt.close(fig)
        elif self.track:
            tt = np.array(self.track)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(tt[:, 0], tt[:, 1] * 1000)
            ax.set_xlabel("sim t [s]"); ax.set_ylabel("EE position error [mm]")
            ax.set_title("Tracking error (dynamic mode)"); ax.grid(alpha=.3)
            fig.tight_layout(); fig.savefig(os.path.join(out, "tracking_error.png"), dpi=130)
            plt.close(fig)


def main(args=None):
    rclpy.init(args=args)
    node = ValidationNode()
    try:
        while rclpy.ok() and not node.should_finish():
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    node.finalize()
    node.destroy_node()
    rclpy.try_shutdown()
    sys.exit(0 if node.passed else 1)


if __name__ == "__main__":
    main()
