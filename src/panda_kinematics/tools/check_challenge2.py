#!/usr/bin/env python3
"""
Self-checks for Challenge 2 (run before the evaluation; every line prints PASS / FAIL).

  1. compute_jacobian        vs MuJoCo's own Jacobian (mj_jac at the hand origin)
  2. compute_jacobian        vs finite differences of the forward kinematics
  3. compute_jacobian_dot    vs finite differences of the Jacobian (analytic = numeric)
  4. plan_trajectory         array shapes, boundary conditions, consistency q -> dq -> ddq
  5. limits                  joint position / speed / acceleration limits respected
  6. singularity guard       a near-singular goal is backed off automatically
  7. collisions              no contacts along the whole timeline (MuJoCo)

Run (needs ROS sourced and mujoco importable):
  source /opt/ros/jazzy/setup.bash && source install/setup.bash
  export PYTHONPATH=$PWD/.venv/lib/python3.12/site-packages:$PYTHONPATH
  python3 src/panda_kinematics/tools/check_challenge2.py
"""
import os
import sys

import numpy as np
import mujoco

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from panda_kinematics import trajectory_node as T
from panda_kinematics import trajectory_experiment as E

SCENE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "mujoco_ros2",
                     "model", "franka_emika_panda", "scene.xml")
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name:<58s} {detail}")


rng = np.random.default_rng(0)
model = mujoco.MjModel.from_xml_path(SCENE)
data = mujoco.MjData(model)
hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")

# ------------------------------------------------------------------ 1-3 Jacobian checks
err_mj, err_fd, err_dot = [], [], []
for _ in range(200):
    q = rng.uniform(T.Q_MIN, T.Q_MAX)
    dq = rng.uniform(-1, 1, 7)
    J = T.compute_jacobian(q)

    data.qpos[:7] = q
    mujoco.mj_forward(model, data)
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, None, data.xpos[hand], hand)
    err_mj.append(np.abs(jacp[:, :7] - J).max())

    h = 1e-6
    J_fd = np.column_stack([
        (T.forward_kinematics(q + h * e)[:3, 3] - T.forward_kinematics(q - h * e)[:3, 3]) / (2 * h)
        for e in np.eye(7)])
    err_fd.append(np.abs(J_fd - J).max())

    err_dot.append(np.abs(T.compute_jacobian_dot(q, dq) - T.compute_jacobian_dot_numeric(q, dq)).max())

check("1. J matches MuJoCo mj_jac (200 random poses)", max(err_mj) < 1e-9, f"max err {max(err_mj):.1e}")
check("2. J matches finite-difference FK", max(err_fd) < 1e-7, f"max err {max(err_fd):.1e}")
check("3. Jdot analytic matches finite differences", max(err_dot) < 1e-6, f"max err {max(err_dot):.1e}")

# ------------------------------------------------------------------ 4 planner structure
dt = T.TRAJ_DT
t, q, dq, ddq = T.plan_trajectory(T.Q_START, T.Q_END, T.TRAJ_DURATION, dt)
check("4a. four arrays of equal length", len(t) == len(q) == len(dq) == len(ddq), f"N = {len(t)}")
check("4b. starts at q_start, ends at q_end", np.allclose(q[0], T.Q_START) and np.allclose(q[-1], T.Q_END))
check("4c. zero velocity at both ends", np.abs(dq[0]).max() < 1e-12 and np.abs(dq[-1]).max() < 1e-12)
check("4d. zero acceleration at both ends", np.abs(ddq[0]).max() < 1e-12 and np.abs(ddq[-1]).max() < 1e-12)
dq_fd = np.gradient(q, dt, axis=0)
ddq_fd = np.gradient(dq, dt, axis=0)
check("4e. dq is the derivative of q", np.abs(dq_fd[5:-5] - dq[5:-5]).max() < 1e-4,
      f"max err {np.abs(dq_fd[5:-5] - dq[5:-5]).max():.1e} rad/s")
check("4f. ddq is the derivative of dq", np.abs(ddq_fd[5:-5] - ddq[5:-5]).max() < 1e-3,
      f"max err {np.abs(ddq_fd[5:-5] - ddq[5:-5]).max():.1e} rad/s^2")

# ------------------------------------------------------------------ 5 limits
info = T.check_trajectory(q, dq, ddq)
check("5a. positions inside joint limits", np.all(q >= T.Q_MIN) and np.all(q <= T.Q_MAX),
      f"min margin {np.degrees(info['limit_margin_rad'].min()):.1f} deg")
check("5b. speeds below SAFETY * limit", info["peak_speed_ratio"].max() <= T.SAFETY + 1e-9,
      f"peak {100 * info['peak_speed_ratio'].max():.0f} % of limit")
check("5c. accelerations below SAFETY * limit", info["peak_accel_ratio"].max() <= T.SAFETY + 1e-9,
      f"peak {100 * info['peak_accel_ratio'].max():.0f} % of limit")
t_fast, q_fast, _, _ = T.plan_trajectory(T.Q_START, T.Q_END, 0.5, dt)
check("5d. too-short T is stretched automatically", t_fast[-1] > 1.5, f"asked 0.5 s, got {t_fast[-1]:.2f} s")
try:
    T.plan_trajectory(T.Q_START, T.Q_MAX + 0.1, 4.0, dt)
    check("5e. goal outside joint limits is rejected", False)
except ValueError:
    check("5e. goal outside joint limits is rejected", True)

# ------------------------------------------------------------------ 6 singularity guard
q_singular = np.array([0.0, 0.0, 0.0, -0.0698, 0.0, 0.0, 0.785])      # arm stretched straight up
sig_end, w_end = T.jacobian_metrics(q_singular)
_, q_g, _, _ = T.plan_trajectory(T.Q_START, q_singular, 6.0, dt)
metrics = np.array([T.jacobian_metrics(x) for x in q_g[::50]])
frac = np.linalg.norm(q_g[-1] - T.Q_START) / np.linalg.norm(q_singular - T.Q_START)
check("6a. requested goal is near-singular", w_end < T.W_MIN_ALLOWED,
      f"at goal: sigma_min {sig_end:.3f}, w {w_end:.4f} (threshold {T.W_MIN_ALLOWED})")
check("6b. guard backs the goal off", not np.allclose(q_g[-1], q_singular),
      f"planned move is {100 * frac:.0f} % of the requested one")
check("6c. guarded path stays above both thresholds",
      metrics[:, 0].min() >= T.SIGMA_MIN_ALLOWED - 1e-3 and metrics[:, 1].min() >= T.W_MIN_ALLOWED - 1e-3,
      f"min sigma_min {metrics[:, 0].min():.3f}, min w {metrics[:, 1].min():.3f}")
check("6d. the real Team 6 path needs no back-off", np.allclose(q[-1], T.Q_END),
      f"min sigma_min {info['sigma_min']:.3f}, min w {info['manipulability_min']:.3f}")

# ------------------------------------------------------------------ 7 collisions along the timeline
times, q_all, dq_all, ddq_all, phase = E.build_timeline()


def harmful_contacts(qq):
    """Contacts other than the two closed fingers touching each other (that is normal)."""
    data.qpos[:7] = qq
    data.qpos[7:] = 0.0                               # gripper closed, as commanded (GRIPPER_CTRL = 0)
    mujoco.mj_forward(model, data)
    out = []
    for c in data.contact[:data.ncon]:
        pair = {model.body(model.geom_bodyid[c.geom1]).name, model.body(model.geom_bodyid[c.geom2]).name}
        if pair != {"left_finger", "right_finger"}:
            out.append(tuple(sorted(pair)))
    return out


late = [i for i in range(0, len(q_all), 25) if phase[i] != 0 and harmful_contacts(q_all[i])]
check("7a. no contacts during hold_start / main / hold_end", not late, f"{len(late)} bad samples")
prep_bad = [times[i] for i in range(0, len(q_all), 25) if phase[i] == 0 and harmful_contacts(q_all[i])]
check("7b. only start-up overlap (zero pose) during prep, gone early",
      (not prep_bad) or max(prep_bad) < 0.25 * E.T_PREP,
      f"contact until t = {max(prep_bad) if prep_bad else 0:.2f} s of {E.T_PREP:.0f} s (hand overlaps link5 at q = 0)")

# ------------------------------------------------------------------ 8 code copied from Challenge 1
fk_err = []
for _ in range(200):
    q = rng.uniform(T.Q_MIN, T.Q_MAX)
    data.qpos[:7] = q
    mujoco.mj_forward(model, data)
    fk_err.append(np.linalg.norm(T.forward_kinematics(q)[:3, 3] - data.xpos[hand]))
check("8a. copied forward_kinematics matches MuJoCo hand position", max(fk_err) < 1e-9, f"max err {max(fk_err):.1e} m")
T_goal = T.forward_kinematics(T.Q_END)
q_ik, ok = T.inverse_kinematics(T_goal, T.Q_START)
pos_err = np.linalg.norm(T.forward_kinematics(q_ik)[:3, 3] - T_goal[:3, 3])
check("8b. copied inverse_kinematics reaches the Q_END pose", ok and pos_err < 1e-3, f"position error {pos_err:.1e} m")

print()
print("ALL CHECKS PASSED" if all(results) else "SOME CHECKS FAILED")
sys.exit(0 if all(results) else 1)
