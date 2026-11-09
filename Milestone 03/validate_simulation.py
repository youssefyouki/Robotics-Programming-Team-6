#!/usr/bin/env python3
"""
Milestone 03 - validate the kinematic equations against MuJoCo (pure Python).

Tests
  1. Forward position      : FK(q)                   vs simulator EE site position
  2. Forward velocity      : Jv(q) qd                vs simulator EE site velocity
  3. Forward acceleration  : Jv qdd + Jv_dot qd      vs simulator EE site acceleration
  4. Inverse (pos/vel/acc) : desired Cartesian circle -> IK, qd = J+ v, qdd = J+(a - Jdot qd),
                             joint values are fed to the simulator and the sensed EE motion
                             must reproduce the desired position / velocity / acceleration
  5. Dynamic run           : position-controlled simulation tracking sinusoidal joint
                             references; sensed EE velocity/acceleration vs the equations
                             evaluated on the simulator's own q, qd, qacc

Run:  .venv/bin/python "Milestone 03/validate_simulation.py"
Outputs: Milestone 03/results/*.png and results/summary.txt
"""
import os
import sys
import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "milestone_03"))
from milestone_03 import kinematics as K
from milestone_03 import sim_model as S

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

# tolerances ("simple tolerance" required by the milestone)
TOL_POS = 1e-5     # m
TOL_VEL = 1e-5     # m/s
TOL_ACC = 1e-4     # m/s^2

rng = np.random.default_rng(1)
model, data, SITE = S.load_model()
summary = []


def set_state(q, qd=None, qdd=None):
    data.qpos[:5] = q
    data.qvel[:5] = 0 if qd is None else qd
    mujoco.mj_forward(model, data)
    if qdd is not None:
        data.qacc[:5] = qdd
        data.qacc[5] = 0.0


def random_q(scale=0.7):
    return rng.uniform(K.JOINT_LIMITS[:, 0] * scale, K.JOINT_LIMITS[:, 1] * scale)


def report(name, errs, tol, unit):
    errs = np.asarray(errs)
    ok = errs.max() < tol
    line = (f"{name:<34s} max={errs.max():.3e} {unit}  mean={errs.mean():.3e} {unit}  "
            f"tol={tol:.0e}  {'PASS' if ok else 'FAIL'}")
    print(line)
    summary.append(line)
    return ok


# ------------------------------------------------------------------ tests 1-3 (random states)
N_RANDOM = 300
e_pos, e_vel, e_acc = [], [], []
for _ in range(N_RANDOM):
    q, qd, qdd = random_q(), rng.uniform(-2, 2, 5), rng.uniform(-5, 5, 5)
    set_state(q, qd, qdd)
    e_pos.append(np.linalg.norm(data.site_xpos[SITE] - K.ee_position(q)))
    v_sim, _ = S.site_velocity(model, data, SITE)
    e_vel.append(np.linalg.norm(v_sim - K.forward_velocity(q, qd)[0]))
    a_sim, _ = S.site_acceleration(model, data, SITE)
    e_acc.append(np.linalg.norm(a_sim - K.forward_acceleration(q, qd, qdd)[0]))

print(f"\n--- Forward kinematics vs simulator ({N_RANDOM} random states) ---")
ok = [report("1. forward position", e_pos, TOL_POS, "m"),
      report("2. forward velocity", e_vel, TOL_VEL, "m/s"),
      report("3. forward acceleration", e_acc, TOL_ACC, "m/s^2")]

# ------------------------------------------------------------------ test 4 (inverse kinematics)
T_CIRCLE = 2.0
CENTER = np.array([-0.15, 0.0, 0.12])
RADIUS = 0.03
W = 2 * np.pi / T_CIRCLE


def circle(t):
    c, s = np.cos(W * t), np.sin(W * t)
    p = CENTER + RADIUS * np.array([0, c, s])
    v = RADIUS * W * np.array([0, -s, c])
    a = -RADIUS * W**2 * np.array([0, c, s])
    return p, v, a


ts = np.linspace(0, T_CIRCLE, 201)
q_t, qd_t, qdd_t = [], [], []
q_prev = None
for t in ts:
    p_d, v_d, a_d = circle(t)
    q, ik_err = K.inverse_position(p_d, q0=q_prev)
    assert ik_err < 1e-6, f"IK failed at t={t:.2f} (err={ik_err:.2e}): target unreachable"
    qd = K.inverse_velocity(q, v_d)
    qdd = K.inverse_acceleration(q, qd, a_d)
    q_t.append(q); qd_t.append(qd); qdd_t.append(qdd)
    q_prev = q
q_t, qd_t, qdd_t = map(np.array, (q_t, qd_t, qdd_t))

p_des = np.array([circle(t)[0] for t in ts])
v_des = np.array([circle(t)[1] for t in ts])
a_des = np.array([circle(t)[2] for t in ts])
p_sim, v_sim_t, a_sim_t = [], [], []
for q, qd, qdd in zip(q_t, qd_t, qdd_t):
    set_state(q, qd, qdd)
    p_sim.append(data.site_xpos[SITE].copy())
    v_sim_t.append(S.site_velocity(model, data, SITE)[0])
    a_sim_t.append(S.site_acceleration(model, data, SITE)[0])
p_sim, v_sim_t, a_sim_t = map(np.array, (p_sim, v_sim_t, a_sim_t))

print("\n--- Inverse kinematics (Cartesian circle -> joint space -> simulator) ---")
ok += [report("4a. inverse position", np.linalg.norm(p_sim - p_des, axis=1), TOL_POS, "m"),
       report("4b. inverse velocity", np.linalg.norm(v_sim_t - v_des, axis=1), TOL_VEL, "m/s"),
       report("4c. inverse acceleration", np.linalg.norm(a_sim_t - a_des, axis=1), TOL_ACC, "m/s^2")]

fig, ax = plt.subplots(2, 3, figsize=(15, 7))
labels = ["x", "y", "z"]
for i in range(3):
    ax[0, 0].plot(ts, p_des[:, i], label=f"desired {labels[i]}")
    ax[0, 0].plot(ts, p_sim[:, i], "--", label=f"sim {labels[i]}")
    ax[0, 1].plot(ts, v_des[:, i], label=f"desired {labels[i]}")
    ax[0, 1].plot(ts, v_sim_t[:, i], "--", label=f"sim {labels[i]}")
    ax[0, 2].plot(ts, a_des[:, i], label=f"desired {labels[i]}")
    ax[0, 2].plot(ts, a_sim_t[:, i], "--", label=f"sim {labels[i]}")
for a_, title, unit in zip(ax[0], ["EE position", "EE velocity", "EE acceleration"],
                           ["m", "m/s", "m/s^2"]):
    a_.set_title(title); a_.set_xlabel("t [s]"); a_.set_ylabel(unit); a_.legend(fontsize=7); a_.grid(alpha=.3)
for j, n in enumerate(K.JOINT_NAMES):
    ax[1, 0].plot(ts, q_t[:, j], label=n)
    ax[1, 1].plot(ts, qd_t[:, j], label=n)
    ax[1, 2].plot(ts, qdd_t[:, j], label=n)
for a_, title, unit in zip(ax[1], ["joint angles (IK)", "joint velocities (J+ v)",
                                   "joint accelerations (J+(a - Jdot qd))"],
                           ["rad", "rad/s", "rad/s^2"]):
    a_.set_title(title); a_.set_xlabel("t [s]"); a_.set_ylabel(unit); a_.legend(fontsize=7); a_.grid(alpha=.3)
fig.suptitle("Inverse kinematics validation: desired circle vs simulator-sensed end effector")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "inverse_validation.png"), dpi=130)
plt.close(fig)

# ------------------------------------------------------------------ test 5 (dynamic simulation)
mujoco.mj_resetData(model, data)
q_c = np.array([0.0, -0.5, 0.6, 0.3, 0.0])
amp = np.array([0.6, 0.35, 0.4, 0.5, 0.8])
freq = np.array([0.3, 0.4, 0.5, 0.35, 0.25])          # Hz


def q_ref(t):
    return q_c + amp * np.sin(2 * np.pi * freq * t)


data.qpos[:5] = q_ref(0.0)
mujoco.mj_forward(model, data)
LOG_EVERY = 5
T_SIM = 8.0
steps = int(T_SIM / model.opt.timestep)
log = {k: [] for k in ["t", "v_sim", "v_eq", "a_sim", "a_eq", "p_sim", "p_eq"]}
for step in range(steps):
    data.ctrl[:5] = q_ref(data.time)
    data.ctrl[5] = 0.0
    mujoco.mj_step(model, data)
    if step % LOG_EVERY == 0:
        mujoco.mj_forward(model, data)               # consistent q, qd, qacc at this instant
        q, qd, qdd = data.qpos[:5].copy(), data.qvel[:5].copy(), data.qacc[:5].copy()
        log["t"].append(data.time)
        log["p_sim"].append(data.site_xpos[SITE].copy())
        log["p_eq"].append(K.ee_position(q))
        log["v_sim"].append(S.site_velocity(model, data, SITE)[0])
        log["v_eq"].append(K.forward_velocity(q, qd)[0])
        log["a_sim"].append(S.site_acceleration(model, data, SITE)[0])
        log["a_eq"].append(K.forward_acceleration(q, qd, qdd)[0])
log = {k: np.array(v) for k, v in log.items()}

print("\n--- Dynamic simulation (position-controlled, sinusoidal joint references) ---")
ok += [report("5a. position (sim vs FK)", np.linalg.norm(log["p_sim"] - log["p_eq"], axis=1), TOL_POS, "m"),
       report("5b. velocity (sim vs J qd)", np.linalg.norm(log["v_sim"] - log["v_eq"], axis=1), TOL_VEL, "m/s"),
       report("5c. acceleration (sim vs eq.)", np.linalg.norm(log["a_sim"] - log["a_eq"], axis=1), TOL_ACC, "m/s^2")]

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
for i in range(3):
    ax[0].plot(log["t"], log["p_eq"][:, i], label=f"FK {labels[i]}")
    ax[0].plot(log["t"], log["p_sim"][:, i], "--", label=f"sim {labels[i]}")
    ax[1].plot(log["t"], log["v_eq"][:, i], label=f"J qd {labels[i]}")
    ax[1].plot(log["t"], log["v_sim"][:, i], "--", label=f"sim {labels[i]}")
    ax[2].plot(log["t"], log["a_eq"][:, i], label=f"eq {labels[i]}")
    ax[2].plot(log["t"], log["a_sim"][:, i], "--", label=f"sim {labels[i]}")
for a_, title, unit in zip(ax, ["EE position", "EE velocity", "EE acceleration"], ["m", "m/s", "m/s^2"]):
    a_.set_title(title); a_.set_xlabel("t [s]"); a_.set_ylabel(unit); a_.legend(fontsize=7); a_.grid(alpha=.3)
fig.suptitle("Dynamic run: equations vs simulator-sensed end effector")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "dynamic_validation.png"), dpi=130)
plt.close(fig)

# ------------------------------------------------------------------ summary
verdict = "ALL TESTS PASSED" if all(ok) else "SOME TESTS FAILED"
print(f"\n{verdict}   (plots in {OUT})")
with open(os.path.join(OUT, "summary.txt"), "w") as f:
    f.write("\n".join(summary) + f"\n{verdict}\n")
sys.exit(0 if all(ok) else 1)
