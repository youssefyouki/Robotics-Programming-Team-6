#!/usr/bin/env python3
"""
Milestone 04 - verification of the joint-space and task-space trajectories (no simulator, no ROS).

Checks the mathematics of the plan used by the sorting cycle:
  A  quintic polynomial: boundary conditions (position, velocity, acceleration) at both ends
  B  joint-space segments: end points, zero velocity/acceleration at rest, peak speed within limit
  C  task-space segments: the TCP stays on the straight line (deviation), the tool stays vertical,
     the TCP speed profile is minimum-jerk, qd = J^-1 v agrees with the TCP velocity obtained by
     differentiating the IK solution
  D  whole plan: joint limits, q / qd / qdd consistency (qd = dq/dt, qdd = dqd/dt), reference
     continuity at every segment junction, every grasp/drop target reached

Run:  .venv/bin/python "Milestone 04/code/verify_trajectories.py"      -> results/trajectory_verification.txt
"""
import os
import sys
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
sys.path.insert(0, os.path.join(ROOT, "milestone_03"))
sys.path.insert(0, os.path.join(ROOT, "milestone_04"))
from milestone_03 import kinematics as K
from milestone_04 import cell as C, planning as P, tcp_kinematics as TK

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
lines, ok_all = [], []


def check(name, value, tol, unit, fmt="{:.3e}"):
    ok = value < tol
    ok_all.append(ok)
    lines.append(f"{'PASS' if ok else 'FAIL'}  {name:<62s} {fmt.format(value)} {unit} (tol {tol:g})")


rng = np.random.default_rng(0)
dt = P.DT
TASK = ("descend", "lift", "lower", "retreat")
JOINT = ("approach", "transfer", "home")

# ---------------------------------------------------------------- A quintic polynomial
err = 0.0
for _ in range(200):
    q0, q1 = rng.uniform(-2, 2, 2)
    v0, v1, a0, a1 = rng.uniform(-1, 1, 4)
    T = rng.uniform(0.5, 4)
    c = P.quintic_coefficients(q0, v0, a0, q1, v1, a1, T)
    s0, sT = P.quintic_eval(c, 0.0), P.quintic_eval(c, T)
    err = max(err, np.abs(np.array(s0) - [q0, v0, a0]).max(), np.abs(np.array(sT) - [q1, v1, a1]).max())
lines.append("--- A. quintic polynomial (200 random boundary-condition sets) ---")
check("boundary conditions (q, qd, qdd) at t=0 and t=T", err, 1e-9, "")

# ---------------------------------------------------------------- build the real plan
model_poses = {c: (C.box_start_position(c), C.BOX_START[c][1]) for c in C.COLORS}
plan = P.build_sorting_plan(np.zeros(5), model_poses)
q, qd, qdd, t = plan.q, plan.qd, plan.qdd, plan.t
segs = plan.segments

# ---------------------------------------------------------------- B joint-space segments
lines.append(f"--- B. joint-space segments (approach / transfer / home): {sum(s[0] in JOINT for s in segs)} segments ---")
rest_err, vmax_ratio, jump = 0.0, 0.0, 0.0
for name, i0, i1, color in segs:
    if name in JOINT:
        rest_err = max(rest_err, np.abs(qd[i1]).max(), np.abs(qdd[i1]).max())
        T_seg = (i1 - i0 + 1) * dt
        d = np.abs(q[i1] - q[i0 - 1]).max()
        # minimum-jerk quintic: peak speed 1.875 * distance / T
        vmax_ratio = max(vmax_ratio, np.abs(qd[i0:i1 + 1]).max() / max(1.875 * d / T_seg, 1e-9))
check("velocity and acceleration are zero at the end of every segment", rest_err, 1e-9, "")
check("peak joint speed vs analytic 1.875*distance/T (ratio - 1)", abs(vmax_ratio - 1), 1e-3, "", "{:.2e}")
check("peak joint speed below the planning limit", np.abs(qd).max() - 1.2 * 1.001, 0.0, "rad/s", "{:+.3f}")

# ---------------------------------------------------------------- C task-space segments
lines.append(f"--- C. task-space segments (descend / lift / lower / retreat): {sum(s[0] in TASK for s in segs)} segments ---")
dev, pitch_err, speed_err, jinv_err, end_err, rest_task = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
for name, i0, i1, color in segs:
    if name not in TASK:
        continue
    pts = np.array([TK.tcp_position(x) for x in q[i0 - 1:i1 + 1]])         # includes the start sample
    p0, p1 = pts[0], pts[-1]
    u = (p1 - p0) / np.linalg.norm(p1 - p0)
    rel = pts - p0
    dev = max(dev, np.linalg.norm(rel - np.outer(rel @ u, u), axis=1).max())
    pitch_err = max(pitch_err, max(abs(TK.tool_pitch(x) + np.pi / 2) for x in q[i0:i1 + 1:25]))
    # speed along the line must follow the minimum-jerk profile 30 tau^2 (1 - tau)^2 L / T
    L, T_seg = np.linalg.norm(p1 - p0), (i1 - i0 + 1) * dt
    tau = np.arange(1, len(pts)) / (len(pts) - 1)
    v_an = 30 * tau**2 * (1 - tau)**2 * L / T_seg
    v_fd = np.linalg.norm(np.gradient(pts, dt, axis=0, edge_order=2), axis=1)[1:]
    speed_err = max(speed_err, np.abs(v_fd - v_an).max())
    # TCP velocity from the Jacobian and the planned qd vs differentiating the TCP path
    for k in range(i0 + 5, i1 - 5, 40):
        v_j = TK.tcp_jacobian(q[k]) @ qd[k]
        v_d = (TK.tcp_position(q[k + 1]) - TK.tcp_position(q[k - 1])) / (2 * dt)
        jinv_err = max(jinv_err, np.linalg.norm(v_j - v_d))
    end_err = max(end_err, np.linalg.norm(p1 - TK.tcp_position(q[i1])))
    rest_task = max(rest_task, np.abs(qd[i1]).max(), np.abs(qdd[i1]).max())
check("TCP deviation from the straight line", dev * 1000, 0.001, "mm", "{:.2e}")
check("tool pitch stays vertical (-90 deg)", np.degrees(pitch_err), 1e-6, "deg", "{:.2e}")
check("TCP speed vs minimum-jerk profile 30 t^2 (1-t)^2 L/T", speed_err, 1e-4, "m/s")
check("qd (J^-1 v form) vs TCP velocity from differentiating the path", jinv_err, 1e-4, "m/s")
check("velocity and acceleration are zero at the end of every task segment", rest_task, 1e-6, "")

# ---------------------------------------------------------------- D whole plan
lines.append(f"--- D. whole plan: {plan.duration:.1f} s, {len(plan)} samples, {len(segs)} segments ---")
lo, hi = K.JOINT_LIMITS[:, 0], K.JOINT_LIMITS[:, 1]
check("joint limit violation", max((lo - q).max(), (q - hi).max(), 0.0), 1e-9, "rad")
qd_fd = np.gradient(q, dt, axis=0, edge_order=2)
check("qd consistent with dq/dt (max over the plan)", np.abs(qd_fd - qd).max(), 5e-3, "rad/s")
qdd_fd = np.gradient(qd, dt, axis=0, edge_order=2)
interior = np.ones(len(q), bool)                       # differentiating across a junction mixes two segments
for _, i0, i1, _ in segs:
    interior[max(0, i0 - 2):i0 + 3] = False
    interior[max(0, i1 - 2):i1 + 3] = False
interior[:3] = interior[-3:] = False
check("qdd consistent with dqd/dt (away from segment junctions)", np.abs(qdd_fd - qdd)[interior].max(), 2e-3, "rad/s^2")
junction = 0.0
for (_, i0, _, _), (_, j0, _, _) in zip(segs[:-1], segs[1:]):
    junction = max(junction, np.abs(q[j0] - q[j0 - 1]).max(), np.abs(qd[j0] - qd[j0 - 1]).max() * dt)
check("reference is continuous at every segment junction (step per sample)", junction, 5e-3, "rad")
reach = 0.0
for c in C.COLORS:
    pos, _ = model_poses[c]
    grasp = [s for s in segs if s[0] == "grasp" and s[3] == c][0]
    reach = max(reach, np.linalg.norm(TK.tcp_position(q[grasp[1]]) - pos))
    drop = [s for s in segs if s[0] == "release" and s[3] == c][0]
    bc = C.bin_center(c)
    reach = max(reach, np.linalg.norm(TK.tcp_position(q[drop[1]])[:2] - bc))
check("TCP reaches every box (grasp) and every bin (release)", reach * 1000, 0.01, "mm", "{:.2e}")

verdict = "ALL CHECKS PASSED" if all(ok_all) else "SOME CHECKS FAILED"
lines.append(verdict)
text = "\n".join(lines)
print(text)
open(os.path.join(OUT, "trajectory_verification.txt"), "w").write(text + "\n")
sys.exit(0 if all(ok_all) else 1)
