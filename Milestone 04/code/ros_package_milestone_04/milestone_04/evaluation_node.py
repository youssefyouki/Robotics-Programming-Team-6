#!/usr/bin/env python3
"""
Evaluation node for the sorting cycle (Milestone 04).

Records state, reference, actuator command and box poses. When the planner reports `done` it
writes to `output_dir`:
  summary.txt           joint / TCP tracking errors, torques, boxes sorted, cycle time, and the
                        validation of the task-space segments (straight lines) and the joint-space
                        segments (quintic moves) against the simulator
  segments.csv          one row per segment: phase, duration, line deviation / joint error, end error
  joint_tracking.png    reference vs measured joints and tracking error (phases shaded)
  tcp_tracking.png      TCP error over time and the TCP path
  results.npz           raw logs
Exit code 0 only if every box ended in the bin of its colour.

  Subscribes : m4/state, m4/reference, m4/actuator_cmd, m4/boxes, m4/status, m4/done
  Parameters : output_dir (str), controller (str, label only), max_wait (s, 0 = unlimited),
               tol_task_mm (max deviation of the measured TCP from the commanded straight line),
               tol_joint_rad (max joint tracking error on the joint-space moves)
"""
import os
import sys
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import Bool, Float64MultiArray, String

from milestone_03 import kinematics as K
from milestone_04 import cell as C, tcp_kinematics as TK, topics as T


class EvaluationNode(Node):
    def __init__(self):
        super().__init__("evaluation_node")
        self.declare_parameter("output_dir", "milestone_04_results")
        self.declare_parameter("controller", "")
        self.declare_parameter("max_wait", 0.0)
        self.declare_parameter("tol_task_mm", 3.0)
        self.declare_parameter("tol_joint_rad", 0.03)
        self.state, self.ref, self.cmd, self.boxes, self.phases = {}, {}, {}, [], []
        self.done_at = None
        self.started = time.time()
        self.passed = None

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        qos = T.QOS_DEPTH
        self.create_subscription(Float64MultiArray, T.STATE, lambda m: self.state.__setitem__(m.data[0], m.data), qos)
        self.create_subscription(Float64MultiArray, T.REFERENCE, lambda m: self.ref.__setitem__(m.data[0], m.data), qos)
        self.create_subscription(Float64MultiArray, T.ACTUATOR_CMD, lambda m: self.cmd.__setitem__(m.data[0], m.data), qos)
        self.create_subscription(Float64MultiArray, T.BOXES, lambda m: self.boxes.append(list(m.data)), 10)
        self.create_subscription(String, T.STATUS, self._on_status, 10)
        self.create_subscription(Bool, T.DONE, self._on_done, latched)
        self.last_t = 0.0
        self.create_subscription(Float64MultiArray, T.STATE, lambda m: setattr(self, "last_t", m.data[0]), qos)

    def _on_status(self, msg):
        parts = msg.data.split(": ", 1)
        if len(parts) == 2 and parts[1] in T.PHASES:          # keep only '<colour>: <phase>' messages
            self.phases.append((self.last_t, msg.data))

    def _on_done(self, msg):
        self.done_at = time.time()
        self.get_logger().info("Cycle finished, writing results.")

    def should_finish(self):
        if self.done_at is not None and time.time() - self.done_at > 0.5:
            return True
        mw = self.get_parameter("max_wait").value
        return mw > 0 and time.time() - self.started > mw

    # ------------------------------------------------------------------ results
    def finalize(self):
        out = os.path.abspath(self.get_parameter("output_dir").value)
        os.makedirs(out, exist_ok=True)
        ts = sorted(set(self.state) & set(self.ref) & set(self.cmd))
        if not ts:
            print("no samples collected")
            self.passed = False
            return
        S = np.array([self.state[t] for t in ts]); R = np.array([self.ref[t] for t in ts])
        U = np.array([self.cmd[t] for t in ts])
        t = np.array(ts)
        q, qd, gq = S[:, 1:6], S[:, 6:11], S[:, 11]
        q_ref, qd_ref, g_ref = R[:, 1:6], R[:, 6:11], R[:, 16]
        u = U[:, 1:6]
        e = q_ref - q
        tcp = np.array([TK.tcp_position(x) for x in q])
        tcp_ref = np.array([TK.tcp_position(x) for x in q_ref])
        tcp_err = np.linalg.norm(tcp - tcp_ref, axis=1) * 1000

        # ---- validation per segment type: task-space lines and joint-space quintic moves
        TASK = ("descend", "lift", "lower", "retreat")
        JOINT = ("approach", "transfer", "home")
        names = [T.PHASES[int(i)] for i in R[:, 17]]
        runs, start = [], 0
        for i in range(1, len(t) + 1):
            if i == len(t) or names[i] != names[start]:
                runs.append((names[start], start, i - 1))
                start = i
        rows, line_dev, task_end, joint_max, joint_end = [], [], [], [], []
        for name, i0, i1 in runs:
            dur = t[i1] - t[i0]
            if name in TASK:
                p0, p1 = tcp_ref[i0], tcp_ref[i1]
                L = np.linalg.norm(p1 - p0)
                if L < 1e-6:
                    continue
                u = (p1 - p0) / L
                rel = tcp[i0:i1 + 1] - p0
                dev = np.linalg.norm(rel - np.outer(rel @ u, u), axis=1).max() * 1000
                end = np.linalg.norm(tcp[i1] - tcp_ref[i1]) * 1000
                line_dev.append(dev); task_end.append(end)
                rows.append((name, "task-space", round(dur, 2), round(dev, 3), round(end, 3)))
            elif name in JOINT:
                err = np.abs(e[i0:i1 + 1]).max()
                end = np.abs(e[i1]).max()
                joint_max.append(err); joint_end.append(end)
                rows.append((name, "joint-space", round(dur, 2), round(err * 1000, 3), round(end * 1000, 3)))
        tol_t, tol_j = self.get_parameter("tol_task_mm").value, self.get_parameter("tol_joint_rad").value
        task_ok = bool(line_dev) and max(line_dev) < tol_t
        joint_ok = bool(joint_max) and max(joint_max) < tol_j

        last = np.array(self.boxes[-1][1:]).reshape(3, 5)
        sorted_flags = {c: bool(last[i, 4]) for i, c in enumerate(C.COLORS)}
        ok = all(sorted_flags.values())
        self.passed = ok
        label = self.get_parameter("controller").value or "controller"
        lines = [f"Sorting cycle evaluation - {label}",
                 f"samples: {len(ts)}  duration: {t[-1]:.1f} s (control period {np.median(np.diff(t))*1000:.1f} ms)",
                 f"joint RMS error   [rad]: {np.round(np.sqrt((e**2).mean(0)), 4).tolist()}",
                 f"joint max error   [rad]: {np.round(np.abs(e).max(0), 4).tolist()}",
                 f"TCP error  RMS {np.sqrt((tcp_err**2).mean()):.2f} mm   max {tcp_err.max():.2f} mm",
                 f"command RMS (torque N*m or position rad): {np.round(np.sqrt((u**2).mean(0)), 3).tolist()}",
                 f"TASK-SPACE  validation ({len(line_dev)} straight-line segments): worst deviation from the line "
                 f"{max(line_dev):.2f} mm, worst end error {max(task_end):.2f} mm  (tol {tol_t:g} mm)  "
                 f"{'PASS' if task_ok else 'FAIL'}",
                 f"JOINT-SPACE validation ({len(joint_max)} quintic segments): worst joint error "
                 f"{max(joint_max) * 1000:.1f} mrad, worst end error {max(joint_end) * 1000:.1f} mrad  "
                 f"(tol {tol_j * 1000:g} mrad)  {'PASS' if joint_ok else 'FAIL'}",
                 "boxes in the bin of their colour: " + ", ".join(f"{c}={'YES' if v else 'NO'}" for c, v in sorted_flags.items()),
                 "SORTING SUCCESSFUL" if ok else "SORTING FAILED"]
        text = "\n".join(lines)
        print("\n" + text)
        with open(os.path.join(out, "summary.txt"), "w") as f:
            f.write(text + "\n")
        with open(os.path.join(out, "segments.csv"), "w") as f:
            f.write("phase,type,duration_s,max_error (mm for task-space deviation / mrad for joint-space),end_error (mm / mrad)\n")
            for r in rows:
                f.write(",".join(str(x) for x in r) + "\n")
        np.savez(os.path.join(out, "results.npz"), t=t, q=q, qd=qd, q_ref=q_ref, qd_ref=qd_ref,
                 u=u, tcp=tcp, tcp_ref=tcp_ref, grip=gq, grip_ref=g_ref)
        self._plots(out, t, q, q_ref, e, tcp, tcp_ref, tcp_err, label)
        print(f"results saved to {out}")

    def _shade(self, ax, t):
        cols = {"approach": "#dbe9ff", "grasp": "#ffe4c4", "release": "#ffe4c4", "transfer": "#e5f5e0"}
        starts = [(tp, s.split(": ")[1]) for tp, s in self.phases] + [(t[-1], "end")]
        for (t0, name), (t1, _) in zip(starts[:-1], starts[1:]):
            if name in cols:
                ax.axvspan(t0, t1, color=cols[name], alpha=0.6, lw=0)

    def _plots(self, out, t, q, q_ref, e, tcp, tcp_ref, tcp_err, label):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            self.get_logger().warn(f"plots skipped ({exc})")
            return
        fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        for j, n in enumerate(K.JOINT_NAMES):
            ax[0].plot(t, q_ref[:, j], lw=1, label=f"{n} ref")
            ax[0].plot(t, q[:, j], "--", lw=1)
            ax[1].plot(t, e[:, j] * 1000, lw=1, label=n)
        for a in ax:
            self._shade(a, t); a.grid(alpha=.3); a.legend(fontsize=7, ncol=5)
        ax[0].set_ylabel("joint [rad]  (solid ref, dashed measured)")
        ax[1].set_ylabel("tracking error [mrad]"); ax[1].set_xlabel("t [s]")
        fig.suptitle(f"Joint tracking - {label}"); fig.tight_layout()
        fig.savefig(os.path.join(out, "joint_tracking.png"), dpi=130); plt.close(fig)

        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        ax[0].plot(t, tcp_err); self._shade(ax[0], t)
        ax[0].set_xlabel("t [s]"); ax[0].set_ylabel("TCP position error [mm]"); ax[0].grid(alpha=.3)
        ax[1].plot(tcp_ref[:, 0], tcp_ref[:, 1], label="reference"); ax[1].plot(tcp[:, 0], tcp[:, 1], "--", label="measured")
        ax[1].set_xlabel("x [m]"); ax[1].set_ylabel("y [m]"); ax[1].axis("equal"); ax[1].legend(); ax[1].grid(alpha=.3)
        for color in C.COLORS:
            c = C.bin_center(color); ax[1].plot(*c, "s", color=color, ms=12, alpha=.5)
        fig.suptitle(f"Task-space tracking - {label}"); fig.tight_layout()
        fig.savefig(os.path.join(out, "tcp_tracking.png"), dpi=130); plt.close(fig)


def main(args=None):
    rclpy.init(args=args)
    node = EvaluationNode()
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
