#!/usr/bin/env python3
"""
Milestone 04 - compare the three motor-level controllers on the same sorting plan (no ROS).

Runs the full colour-sorting cycle in MuJoCo for: built-in position servo, PD + gravity
compensation, computed torque. Writes results/controller_comparison.png, results/plan_profiles.png
and results/controller_comparison.txt.

Run:  .venv/bin/python "Milestone 04/code/compare_controllers.py"
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
sys.path.insert(0, os.path.join(ROOT, "milestone_03"))
sys.path.insert(0, os.path.join(ROOT, "milestone_04"))
from milestone_03 import kinematics as K
from milestone_04 import offline, planning as P, tcp_kinematics as TK

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
NAMES = ["servo", "pd_gravity", "computed_torque"]
LABELS = {"servo": "position servo (built-in)", "pd_gravity": "PD + gravity compensation",
          "computed_torque": "computed torque"}

plan, runs, rows = None, {}, []
for name in NAMES:
    plan, log, in_bin, poses, _ = offline.simulate(name, plan=plan)
    m = offline.metrics(plan, log)
    runs[name] = (log, m, in_bin)
    rows.append(f"{LABELS[name]:<28s} joint RMS [mrad] {np.round(m['joint_rms_rad']*1000, 1).tolist()}  "
                f"TCP RMS {m['tcp_rms_mm']:.2f} mm  max {m['tcp_max_mm']:.2f} mm  "
                f"sorted {sum(in_bin.values())}/3")
    print(rows[-1])
with open(os.path.join(OUT, "controller_comparison.txt"), "w") as f:
    f.write(f"plan duration {plan.duration:.1f} s, control period 4 ms\n" + "\n".join(rows) + "\n")

# --- tracking comparison
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
for name in NAMES:
    log, m, _ = runs[name]
    err = np.linalg.norm(log["tcp"] - log["tcp_ref"], axis=1) * 1000
    ax[0].semilogy(log["t"], np.maximum(err, 1e-3), lw=1, label=LABELS[name])
ax[0].set_xlabel("t [s]"); ax[0].set_ylabel("TCP position error [mm] (log)"); ax[0].grid(alpha=.3, which="both"); ax[0].legend()
w = 0.25
for i, name in enumerate(NAMES):
    ax[1].bar(np.arange(5) + (i - 1) * w, runs[name][1]["joint_rms_rad"] * 1000, w, label=LABELS[name])
ax[1].set_xticks(range(5)); ax[1].set_xticklabels(K.JOINT_NAMES, rotation=15)
ax[1].set_ylabel("joint tracking RMS [mrad]"); ax[1].set_yscale("log"); ax[1].legend(); ax[1].grid(alpha=.3, axis="y")
fig.suptitle("Motor-level control comparison over the full sorting cycle")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "controller_comparison.png"), dpi=130); plt.close(fig)

# --- the planned profiles (joint-space and task-space segments)
fig, ax = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
for j, n in enumerate(K.JOINT_NAMES):
    ax[0].plot(plan.t, plan.q[:, j], lw=1, label=n)
    ax[1].plot(plan.t, plan.qd[:, j], lw=1)
    ax[2].plot(plan.t, plan.qdd[:, j], lw=1)
tcp = np.array([TK.tcp_position(q) for q in plan.q[::5]])
for i, n in enumerate("xyz"):
    ax[3].plot(plan.t[::5], tcp[:, i], lw=1, label=f"TCP {n}")
ax[3].plot(plan.t, plan.grip * 0.05, "k--", lw=1, label="gripper (x0.05)")
for a, lab in zip(ax, ["q [rad]", "qd [rad/s]", "qdd [rad/s^2]", "TCP [m]"]):
    a.set_ylabel(lab); a.grid(alpha=.3)
    for name, i0, i1, color in plan.segments:
        if name in ("descend", "lift", "lower", "retreat"):
            a.axvspan(plan.t[i0], plan.t[i1], color="#ffe4c4", alpha=.6, lw=0)
ax[0].legend(fontsize=7, ncol=5); ax[3].legend(fontsize=7, ncol=4); ax[3].set_xlabel("t [s]")
fig.suptitle("Planned references: quintic joint-space moves; task-space straight lines shaded")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "plan_profiles.png"), dpi=130); plt.close(fig)
print("saved to", OUT)
