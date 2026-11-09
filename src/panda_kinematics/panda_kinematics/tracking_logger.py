#!/usr/bin/env python3
"""
Tracking logger for Challenge 2 - records how well the simulated arm follows the plan.

For every command that is published (message k) it stores
  * the PLANNED joint positions for that step, and
  * the ACTUAL joint positions last reported by the simulator on /joint_state.
After the whole trajectory (plus a short settling time) it computes the errors, writes a CSV,
a figure and a metrics file, and exits.

Two kinds of run can be recorded (parameter `template`):
  template:=false   the extended runner (trajectory_experiment): setup move + rests + main move
  template:=true    the instructor's trajectory_node exactly as given: the plan is just
                    plan_trajectory(Q_START, Q_END, TRAJ_DURATION, traj_dt)

  ros2 run panda_kinematics tracking_logger --ros-args -p control_mode:=VELOCITY -p template:=true

Parameters: control_mode (POSITION|VELOCITY), template (bool), traj_dt (s, default = TRAJ_DT of
the template file), output_dir.   Outputs (output_dir/<control_mode>/): tracking.csv, tracking.png,
metrics.txt
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

from panda_kinematics.trajectory_node import (Q_END, Q_START, TRAJ_DT, TRAJ_DURATION,
                                              forward_kinematics, plan_trajectory)
from panda_kinematics.trajectory_experiment import PHASES, build_timeline

SETTLE_TIME = 2.0        # s of extra recording after the last planned sample


class TrackingLogger(Node):
    def __init__(self):
        super().__init__("tracking_logger")
        self.declare_parameter("control_mode", "POSITION")
        self.declare_parameter("output_dir", "challenge2_results")
        self.declare_parameter("template", False)
        self.declare_parameter("traj_dt", float(TRAJ_DT))
        self.mode = self.get_parameter("control_mode").value.upper()
        self.template = self.get_parameter("template").value
        self.dt = float(self.get_parameter("traj_dt").value)
        self.out = os.path.join(self.get_parameter("output_dir").value, self.mode)

        # The plan, identical to the one the node plays
        if self.template:
            _, self.q_plan, self.dq_plan, _ = plan_trajectory(Q_START, Q_END, TRAJ_DURATION, self.dt)
            self.phase = np.full(len(self.q_plan), 2)            # everything is "main motion"
        else:
            self.dt = float(TRAJ_DT)
            (_, self.q_plan, self.dq_plan, _, self.phase) = build_timeline()
        self.n = len(self.q_plan)
        self.n_total = self.n + int(SETTLE_TIME / self.dt)

        self.q_actual = np.zeros(7)
        self.rows = []                                   # (wall_time, k, q_actual[7])
        self.t0 = None
        self.done = False
        self.create_subscription(JointState, "joint_state", self.on_state, 10)
        self.create_subscription(Float64MultiArray, "joint_commands", self.on_command, 1000)
        self.get_logger().info(f"logging {self.mode} run ({'template node' if self.template else 'extended runner'}, "
                               f"dt = {self.dt}), {self.n_total} command messages expected")

    def on_state(self, msg):
        self.q_actual = np.array(msg.position[:7])       # latest simulator joint positions

    def on_command(self, msg):
        if self.done:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.t0 is None:
            self.t0 = now
        self.rows.append((now - self.t0, len(self.rows), *self.q_actual))
        if len(self.rows) >= self.n_total:
            self.done = True
            self.finalise()

    # ------------------------------------------------------------------ analysis
    def finalise(self):
        os.makedirs(self.out, exist_ok=True)
        dt = self.dt
        rows = np.array(self.rows)
        wall, k, q_act = rows[:, 0], rows[:, 1].astype(int), rows[:, 2:]
        kk = np.minimum(k, self.n - 1)                       # after the plan ends the goal is held
        q_pln = self.q_plan[kk]
        err = q_act - q_pln                                  # joint error, rad
        ph = self.phase[kk]
        t_plan = k * dt
        ee_pln = np.array([forward_kinematics(q)[:3, 3] for q in q_pln])
        ee_act = np.array([forward_kinematics(q)[:3, 3] for q in q_act])
        ee_err = np.linalg.norm(ee_act - ee_pln, axis=1) * 1000.0     # hand position error, mm

        np.savetxt(os.path.join(self.out, "tracking.csv"),
                   np.column_stack([wall, t_plan, ph, q_pln, q_act]), delimiter=",",
                   header="wall_s,plan_s,phase," + ",".join(f"plan_q{i+1}" for i in range(7)) + ","
                          + ",".join(f"act_q{i+1}" for i in range(7)))

        main = np.where((ph == 2) & (k < self.n))[0]         # the planned main motion (not the settling time)
        i0, i1 = main[0], main[-1]
        goal = self.q_plan[-1]
        tail = slice(-max(2, int(0.5 / dt)), None)           # last 0.5 s of the recording
        q_set = q_act[tail].mean(axis=0)
        moving = np.abs(Q_END - Q_START) > 0.1               # joints that really move

        m = {}
        m["rate"] = float((self.n - 1) / wall[min(self.n, len(wall)) - 1])
        m["rms"] = np.degrees(np.sqrt((err[main] ** 2).mean(axis=0)))
        m["max"] = np.degrees(np.abs(err[main]).max(axis=0))
        m["ee_rms"] = float(np.sqrt((ee_err[main] ** 2).mean()))
        m["ee_max"] = float(ee_err[main].max())
        m["start_offset"] = np.degrees(q_act[i0] - self.q_plan[kk[i0]])
        m["end_of_main"] = np.degrees(err[i1])
        m["settled"] = np.degrees(q_set - goal)
        m["settled_ee"] = float(np.linalg.norm(forward_kinematics(q_set)[:3, 3] - forward_kinematics(goal)[:3, 3]) * 1000)
        # how much of the planned joint displacement the arm really travelled during/after the main move
        # VELOCITY mode integrates RELATIVE motion (the target starts at 0), so the commanded
        # displacement is q_end - q_start; POSITION mode commands the ABSOLUTE goal.
        commanded = (goal - Q_START) if self.mode == "VELOCITY" else (goal - q_act[i0])
        m["delivered"] = ((q_set - q_act[i0]) / commanded)[moving]
        # equivalent time lag behind the plan on the joint that moves most
        j = int(np.argmax(np.abs(Q_END - Q_START)))
        shifts = range(0, min(int(0.4 / dt), len(q_act) - i1 - 1))
        best = min(shifts, key=lambda s: np.sum((q_act[main + s, j] - q_pln[main, j]) ** 2))
        m["lag_ms"] = best * dt * 1000

        names = [str(i + 1) for i in np.where(moving)[0]]
        lines = [f"mode: {self.mode}   ({'instructor template node' if self.template else 'extended runner'}, dt = {dt} s)",
                 f"publish rate achieved: {m['rate']:.0f} Hz (target {1 / dt:.0f})",
                 "main motion, per-joint RMS error [deg]: " + np.array2string(m["rms"], precision=2),
                 "main motion, per-joint max error [deg]: " + np.array2string(m["max"], precision=2),
                 f"main motion, hand position error: rms {m['ee_rms']:.1f} mm, max {m['ee_max']:.1f} mm",
                 "joint offset from the plan when the main motion starts [deg]: " + np.array2string(m["start_offset"], precision=2),
                 "joint error at end of main motion [deg]: " + np.array2string(m["end_of_main"], precision=2),
                 "settled joint error vs goal [deg]: " + np.array2string(m["settled"], precision=2),
                 f"settled hand error vs goal: {m['settled_ee']:.1f} mm",
                 f"delivered fraction of the commanded joint displacement (joints {','.join(names)}): "
                 + np.array2string(m["delivered"], precision=3),
                 f"equivalent lag behind the plan (joint {j + 1}): {m['lag_ms']:.0f} ms"]
        with open(os.path.join(self.out, "metrics.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        for line in lines:
            self.get_logger().info(line)

        # ---------------- figure
        fig, ax = plt.subplots(2, 2, figsize=(14, 8))
        tm = t_plan[main] - t_plan[main][0]
        for jj in (0, 1, 3, 5):
            ax[0, 0].plot(tm, np.degrees(q_pln[main, jj]), label=f"plan q{jj+1}")
            ax[0, 0].plot(tm, np.degrees(q_act[main, jj]), "--", label=f"actual q{jj+1}")
        ax[0, 0].set(title=f"{self.mode}: planned vs actual joints (main motion)", xlabel="t [s]", ylabel="deg")
        ax[0, 0].legend(fontsize=7, ncol=2)
        for jj in range(7):
            ax[0, 1].plot(t_plan, np.degrees(err[:, jj]), label=f"q{jj+1}")
        ax[0, 1].set(title="joint error (actual - plan), whole run", xlabel="plan time [s]", ylabel="deg")
        ax[0, 1].legend(fontsize=7, ncol=2)
        ax[1, 0].plot(t_plan, ee_err)
        ax[1, 0].set(title="hand position error", xlabel="plan time [s]", ylabel="mm")
        for p_i, name in enumerate(PHASES):
            idx = np.where(ph == p_i)[0]
            if len(idx) and not self.template:
                for a_ in (ax[0, 1], ax[1, 0]):
                    a_.axvspan(t_plan[idx[0]], t_plan[idx[-1]], alpha=0.07, color="C%d" % p_i)
                ax[1, 0].text(t_plan[idx[0]], ax[1, 0].get_ylim()[1] * 0.92, name, fontsize=7)
        ax[1, 1].plot(k, wall - k * dt)
        ax[1, 1].set(title="message timing: wall time minus ideal time", xlabel="message k", ylabel="s")
        fig.tight_layout()
        fig.savefig(os.path.join(self.out, "tracking.png"), dpi=130)
        plt.close(fig)
        self.get_logger().info(f"results saved to {self.out}")


def main(args=None):
    rclpy.init(args=args)
    node = TrackingLogger()
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
