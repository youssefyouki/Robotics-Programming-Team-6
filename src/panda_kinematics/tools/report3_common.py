#!/usr/bin/env python3
"""
Numbers and figures shared by the two Challenge 3 PDFs (make_report3.py = study guide,
make_submission_report3.py = report to hand in). Everything is computed from controller_node.py and from the
recorded runs in challenge3_results/, so the PDFs always match the code and the data.
"""
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_style import *            # noqa: F401,F403  (styles, helpers, story, TMP, HERE)
from panda_kinematics import controller_node as cn   # noqa: E402

RES = os.path.join(HERE, "..", "challenge3_results")
SCENE = os.path.join(HERE, "..", "..", "mujoco_ros2", "model", "franka_emika_panda", "scene.xml")

# ---------------------------------------------------------------- the plan and its numbers
q0, q1 = cn.Q_START, cn.Q_END
t_plan, q_plan, dq_plan, ddq_plan = cn.plan_trajectory(q0, q1, cn.TRAJ_DURATION, cn.TRAJ_DT)
# the plan ends with a controlled stop (zero-velocity samples at q_end); the numbers below describe the motion only
_moving = np.where(np.abs(dq_plan).max(axis=1) > 0)[0]
N_MOTION = int(_moving[-1]) + 2
t_plan, q_plan, dq_plan, ddq_plan = t_plan[:N_MOTION], q_plan[:N_MOTION], dq_plan[:N_MOTION], ddq_plan[:N_MOTION]
T_MOVE = t_plan[-1]
hand0, hand1 = cn.forward_kinematics(q0)[:3, 3], cn.forward_kinematics(q1)[:3, 3]


def torque_parts(q, dq, ddq):
    """Inertial, Coriolis/centrifugal and gravity torque at one plan sample."""
    M, Cg = cn.compute_dynamics(q, dq, ddq)
    g = cn.compute_dynamics(q, np.zeros(7), ddq)[1]
    return M @ ddq, Cg - g, g


parts = [torque_parts(a, b, c) for a, b, c in zip(q_plan, dq_plan, ddq_plan)]
TAU_INERT = np.array([p[0] for p in parts])
TAU_CORI = np.array([p[1] for p in parts])
TAU_GRAV = np.array([p[2] for p in parts])
RATIO = np.linalg.norm(TAU_INERT + TAU_CORI, axis=1) / np.linalg.norm(TAU_GRAV, axis=1)
SPEED_RATIO = np.abs(dq_plan).max(axis=0) / cn.V_MAX
ACCEL_RATIO = np.abs(ddq_plan).max(axis=0) / cn.A_MAX
TAU_TOTAL = TAU_INERT + TAU_CORI + TAU_GRAV
TAU_USE = np.abs(TAU_TOTAL).max(axis=0) / cn.TAU_MAX
delta = q1 - q0
JOINT_NAMES = [f"q{i + 1}" for i in range(7)]


# ---------------------------------------------------------------- recorded runs
RUNS = [("main", "model + FLC (submitted)"), ("decoupled", "decoupled inertia (diag M) + FLC"),
        ("no_coriolis", "no Coriolis term + FLC"), ("gravity_only", "gravity only + FLC"),
        ("flc_only", "FLC only (no model)"), ("model_only", "model only (no FLC)")]


def parse(label):
    txt = open(os.path.join(RES, label, "metrics.txt")).read()
    num = lambda pat: float(re.search(pat, txt).group(1))
    arr = lambda pat: np.array([float(x) for x in re.search(pat, txt, re.S).group(1).split()])
    return dict(rate=num(r"publish rate achieved: (\d+)"), speed=num(r"at (\d+) % of its speed limit"),
                rms=num(r"\(all joints\): rms ([\d.]+) deg"), mx=num(r"\(all joints\): rms [\d.]+ deg, max ([\d.]+) deg"),
                rms_late=num(r"first [\d.]+ s: rms ([\d.]+) deg"), mx_late=num(r"first [\d.]+ s: rms [\d.]+ deg, max ([\d.]+) deg"),
                pj=arr(r"per-joint rms error \[deg\]: \[(.*?)\]"), pj_late=arr(r"after start-up \[deg\]: \[(.*?)\]"),
                ee_rms=num(r"hand position error: rms ([\d.]+) mm"), ee_max=num(r"hand position error: rms [\d.]+ mm, max ([\d.]+) mm"),
                final=arr(r"final joint error vs goal \[deg\]: \[(.*?)\]"), tau_use=num(r"peak commanded torque: (\d+) %"),
                dtau=num(r"torque step between commands ([\d.]+) Nm"))


def load(label):
    d = np.loadtxt(os.path.join(RES, label, "tracking.csv"), delimiter=",")
    return dict(wall=d[:, 0], t=d[:, 1], q_des=d[:, 2:9], q=d[:, 9:16], dq_des=d[:, 16:23], dq=d[:, 23:30], tau=d[:, 30:37])


R = {k: parse(k) for k, _ in RUNS}                        # first run of every controller (used for the figures)

# Several controllers were run more than once on the real bridge (label, label_r2, label_r3, ...): the tables use the mean.
GROUPS = {k: [d for d in sorted(os.listdir(RES)) if d == k or re.fullmatch(rf"{k}_r\d+", d)] for k, _ in RUNS}
_NUM = ["rms", "mx", "rms_late", "mx_late", "ee_rms", "ee_max", "tau_use", "dtau", "rate"]
RM = {}
for _k, _labels in GROUPS.items():
    _runs = [parse(_l) for _l in _labels]
    RM[_k] = {f: float(np.mean([r[f] for r in _runs])) for f in _NUM}
    RM[_k]["pj_late"] = np.mean([r["pj_late"] for r in _runs], axis=0)
    RM[_k]["pj"] = np.mean([r["pj"] for r in _runs], axis=0)
    RM[_k]["final"] = np.mean([r["final"] for r in _runs], axis=0)
    RM[_k]["lo"] = min(r["rms_late"] for r in _runs)
    RM[_k]["hi"] = max(r["rms_late"] for r in _runs)
    RM[_k]["n"] = len(_runs)

# size of the coupling terms along the plan (norms over the 7 joints, ignoring the first/last samples where ddq ~ 0)
_off, _inertial, _cori = [], [], []
for _a, _b, _c in zip(q_plan, dq_plan, ddq_plan):
    _M, _Cg = cn.compute_dynamics(_a, _b, _c)
    _g = cn.compute_dynamics(_a, np.zeros(7), _c)[1]
    _inertial.append(_M @ _c)
    _off.append((_M - np.diag(np.diag(_M))) @ _c)
    _cori.append(_Cg - _g)
_off, _inertial, _cori = map(np.array, (_off, _inertial, _cori))
_core = slice(5, -5)
with np.errstate(invalid="ignore", divide="ignore"):
    OFF_SHARE = float(np.nanmean((np.linalg.norm(_off, axis=1) / np.linalg.norm(_inertial, axis=1))[_core]))
CORI_PEAK = float(np.linalg.norm(_cori, axis=1).max())
CORI_PEAK_J = _cori.__abs__().max(axis=0)
INERT_PEAK = float(np.linalg.norm(_inertial, axis=1).max())
DYN_CHECK = open(os.path.join(RES, "check_dynamics.txt")).read()
OFFLINE = open(os.path.join(RES, "offline_sim.txt")).read()


def check_value(n):
    """max error printed on line n of check_dynamics.txt"""
    m = re.search(rf"{n}\. .*?max err ([\d.e+-]+)", DYN_CHECK)
    return m.group(1) if m else "?"


# ---------------------------------------------------------------- figures
def savefig(fig_, name):
    p = os.path.join(TMP, name)
    fig_.tight_layout()
    fig_.savefig(p, dpi=150)
    plt.close(fig_)
    return p


def fig_architecture():
    fig_, ax = plt.subplots(figsize=(11, 4.4))
    ax.set_xlim(0, 11); ax.set_ylim(0, 4.4); ax.axis("off")

    def box(x, y, w, h, text, c):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=c, ec="#333", lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.6)

    box(0.1, 1.2, 3.5, 2.3, "controller_node (Python)\n\nplan_trajectory -> q_des, dq_des, ddq_des\ntau = M(q) ddq_des + C(q,dq) dq + g(q)\n      + FLC( q_des - q , dq_des - dq )\n(100 Hz)", "#cfe8e6")
    box(4.3, 1.2, 3.1, 2.3, "mujoco_node (C++ bridge)\nTORQUE + torque_passthrough\n\nctrl = (tau - bias) / gain\nso the servo produces exactly tau\nEuler integrator, 1 kHz\ntorque held until the next message", "#dfe7f1")
    box(8.1, 1.2, 2.8, 2.3, "MuJoCo physics\nunmodified panda.xml\n\n7 position servos\n(force = kp ctrl - kp q - kv dq)\njoint damping 1, armature 0.1", "#f6e7c8")
    ax.annotate("", xy=(4.3, 2.75), xytext=(3.6, 2.75), arrowprops=dict(arrowstyle="->", lw=1.6))
    ax.text(3.95, 3.0, "/joint_commands\n7 torques [Nm],\ngripper, 0", ha="center", fontsize=7.5)
    ax.annotate("", xy=(8.1, 2.75), xytext=(7.4, 2.75), arrowprops=dict(arrowstyle="->", lw=1.6))
    ax.annotate("", xy=(3.6, 1.8), xytext=(4.3, 1.8), arrowprops=dict(arrowstyle="->", lw=1.6, color="#1b7f79"))
    ax.text(3.95, 1.35, "/joint_state\nq, dq", ha="center", fontsize=7.5, color="#1b7f79")
    ax.annotate("", xy=(7.4, 1.8), xytext=(8.1, 1.8), arrowprops=dict(arrowstyle="->", lw=1.6, color="#1b7f79"))
    ax.text(5.5, 0.35, "Before the first torque command the bridge holds the initial pose (the arm does not collapse);\n"
                       "after the trajectory the template node holds with gravity compensation only.", ha="center", fontsize=8, color="#444")
    return savefig(fig_, "arch3.png")


def fig_lagrange():
    fig_, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_xlim(0, 11); ax.set_ylim(0, 4.2); ax.axis("off")

    def box(x, y, w, h, text, c="#dfe7f1"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=c, ec="#333", lw=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.2)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.3))

    box(0.1, 1.7, 2.3, 1.2, "link data (panda.xml)\nmass, centre of mass,\ninertia of each link", "#f6e7c8")
    box(0.1, 0.2, 2.3, 1.1, "DH frames (Challenge 1)\naxes z_j and points o_j", "#f6e7c8")
    box(3.1, 1.0, 2.4, 1.6, "Jacobians of every\ncentre of mass\nJv_i (3x7), Jw_i (3x7)")
    box(6.2, 2.9, 4.6, 1.1, "kinetic energy  T = 1/2 dq' M(q) dq\nM = sum_i ( m_i Jv_i' Jv_i + Jw_i' I_i Jw_i ) + rotor inertia", "#cfe8e6")
    box(6.2, 1.6, 4.6, 1.0, "potential energy  V = sum_i m_i g z_i(q)\ngravity torque  g(q) = dV/dq = sum_i m_i g Jv_i[z]", "#cfe8e6")
    box(6.2, 0.1, 4.6, 1.2, "Coriolis / centrifugal  C(q,dq) dq  from the derivatives of M\n"
                             "(C dq)_i = sum_jk [ dM_ij/dq_k - 1/2 dM_jk/dq_i ] dq_j dq_k", "#cfe8e6")
    arrow(2.4, 2.3, 3.1, 1.9); arrow(2.4, 0.8, 3.1, 1.5)
    arrow(5.5, 2.2, 6.2, 3.4); arrow(5.5, 1.8, 6.2, 2.1); arrow(5.5, 1.4, 6.2, 0.8)
    ax.text(5.5, 3.95, "tau = M(q) ddq + C(q,dq) dq + g(q)", fontsize=10, ha="center", fontweight="bold")
    return savefig(fig_, "lagrange3.png")


def fig_dyn_validation():
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(1)
    damping = model.dof_damping[:7]
    a_all, b_all, n = [], [], 0
    while n < 150:
        q = rng.uniform(cn.Q_LIM_LOW * 0.95, cn.Q_LIM_HIGH * 0.95)
        dq, ddq = rng.uniform(-2, 2, 7), rng.uniform(-8, 8, 7)
        data.qpos[:7], data.qpos[7:] = q, cn.FINGER_OPEN
        data.qvel[:], data.qvel[:7] = 0.0, dq
        mujoco.mj_forward(model, data)
        if data.ncon > 0:
            continue
        n += 1
        data.qacc[:], data.qacc[:7] = 0.0, ddq
        mujoco.mj_inverse(model, data)
        M, Cg = cn.compute_dynamics(q, dq, ddq)
        a_all += list(data.qfrc_inverse[:7])
        b_all += list(M @ ddq + Cg + damping * dq)
    a_all, b_all = np.array(a_all), np.array(b_all)
    fig_, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(a_all, b_all, ".", ms=3)
    lim = np.abs(a_all).max() * 1.05
    ax[0].plot([-lim, lim], [-lim, lim], "k", lw=0.8)
    ax[0].set(xlabel="MuJoCo inverse dynamics [Nm]", ylabel="analytic  M ddq + C dq + g  [Nm]", title="1050 joint torques, 150 random states")
    ax[1].hist(np.log10(np.abs(a_all - b_all) + 1e-16), bins=30, color="#1b7f79")
    ax[1].set(xlabel="log10 |difference| [Nm]", ylabel="count", title="difference is numerical noise")
    for a in ax:
        a.grid(alpha=0.3)
    return savefig(fig_, "dynval3.png")


def fig_fast():
    fig_, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for j in range(7):
        ax[0].plot(t_plan, np.degrees(q_plan[:, j]), label=JOINT_NAMES[j])
        ax[1].plot(t_plan, 100 * np.abs(dq_plan[:, j]) / cn.V_MAX[j])
    ax[0].set(title="planned joint angles", xlabel="t [s]", ylabel="deg"); ax[0].legend(fontsize=7, ncol=2)
    ax[1].set(title="joint speed as % of the Franka limit", xlabel="t [s]", ylabel="%")
    ax[2].plot(t_plan, np.linalg.norm(TAU_INERT + TAU_CORI, axis=1), label="inertial + Coriolis", color="#c0392b", lw=2)
    ax[2].plot(t_plan, np.linalg.norm(TAU_GRAV, axis=1), label="gravity", color="#1b7f79", lw=2)
    ax[2].set(title="torque norm over the 7 joints", xlabel="t [s]", ylabel="Nm"); ax[2].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    return savefig(fig_, "fast3.png")


def flc_output(e, de, joint=0):
    """FLC torque for a single (e, de) pair of one joint, using the submitted fuzzy_controller."""
    saved = cn._flc_state["u"].copy()
    cn._flc_state["u"][:] = 0.0
    out = cn.fuzzy_controller(np.full(7, e), np.full(7, de), 0.01)[joint]
    cn._flc_state["u"][:] = saved
    return out


def fig_flc_design():
    x = np.linspace(-1, 1, 401)
    mu = cn._memberships(x)
    names = ["LN", "MN", "S", "MP", "LP"]
    fig_, ax = plt.subplots(1, 3, figsize=(12.5, 3.7))
    for i, n_ in enumerate(names):
        ax[0].plot(x, mu[:, i], label=n_, lw=2)
    ax[0].set(title="membership functions (same 5 sets for e, de and the output)", xlabel="normalised value", ylabel="degree of membership")
    ax[0].legend(fontsize=8, ncol=5, loc="upper center")
    # rule table
    table_ = cn._RULES
    ax[1].imshow(table_, cmap="coolwarm", origin="upper")
    for a_ in range(5):
        for b_ in range(5):
            ax[1].text(b_, a_, names[table_[a_, b_]], ha="center", va="center", fontsize=9)
    ax[1].set_xticks(range(5)); ax[1].set_xticklabels(names); ax[1].set_yticks(range(5)); ax[1].set_yticklabels(names)
    ax[1].set(xlabel="change of error de", ylabel="error e", title="rule base: output set (25 rules)")
    # control surface of joint 1
    e = np.linspace(-0.08, 0.08, 41)
    de = np.linspace(-2.5, 2.5, 41)
    Z = np.array([[flc_output(a, b, 0) for a in e] for b in de])
    c = ax[2].contourf(np.degrees(e), de, Z, levels=15, cmap="coolwarm")
    plt.colorbar(c, ax=ax[2], label="correction torque joint 1 [Nm]")
    ax[2].set(xlabel="error e [deg]", ylabel="change of error de [rad/s]", title="control surface of the FLC (joint 1)")
    return savefig(fig_, "flcdesign3.png")


# worked inference example (normalised inputs): e_n = 0.6, de_n = -0.3
WORK_E_N, WORK_DE_N = 0.6, -0.3


def worked_example():
    en, dn = WORK_E_N, WORK_DE_N
    mu_e = cn._memberships(np.array(en))
    mu_d = cn._memberships(np.array(dn))
    names = ["LN", "MN", "S", "MP", "LP"]
    fired = []
    for a in range(5):
        for b in range(5):
            w = min(mu_e[a], mu_d[b])
            if w > 1e-9:
                fired.append((names[a], names[b], names[cn._RULES[a, b]], float(w), int(cn._RULES[a, b])))
    strength = np.zeros(5)
    for _, _, _, w, o in fired:
        strength[o] = max(strength[o], w)
    combined = np.minimum(strength[None, :], cn._OUTPUT_SETS).max(axis=1)
    u = float((combined * cn._U_GRID).sum() / combined.sum())
    check_u = flc_output(en * cn.FLC_E_SCALE[0], dn * cn.FLC_DE_SCALE[0], 0) / cn.FLC_U_SCALE[0]
    assert abs(u - check_u) < 1e-9, (u, check_u)
    return mu_e, mu_d, fired, strength, combined, u


def fig_worked():
    mu_e, mu_d, fired, strength, combined, u = worked_example()
    names = ["LN", "MN", "S", "MP", "LP"]
    x = np.linspace(-1, 1, 401)
    mu = cn._memberships(x)
    fig_, ax = plt.subplots(1, 3, figsize=(12.5, 3.6))
    ax[0].plot(x, mu, lw=1.2)
    ax[0].axvline(WORK_E_N, color="k", ls="--"); ax[0].set(title=f"e = {WORK_E_N:+.1f} (normalised)", xlabel="e / E_scale")
    for i in range(5):
        if mu_e[i] > 1e-9:
            ax[0].plot([WORK_E_N], [mu_e[i]], "ko"); ax[0].text(WORK_E_N + 0.03, mu_e[i], f"{names[i]} {mu_e[i]:.1f}", fontsize=8)
    ax[1].plot(x, mu, lw=1.2)
    ax[1].axvline(WORK_DE_N, color="k", ls="--"); ax[1].set(title=f"de = {WORK_DE_N:+.1f} (normalised)", xlabel="de / DE_scale")
    for i in range(5):
        if mu_d[i] > 1e-9:
            ax[1].plot([WORK_DE_N], [mu_d[i]], "ko"); ax[1].text(WORK_DE_N + 0.03, mu_d[i], f"{names[i]} {mu_d[i]:.1f}", fontsize=8)
    for i in range(5):
        ax[2].plot(cn._U_GRID, cn._OUTPUT_SETS[:, i], color="#bbbbbb", lw=1)
    ax[2].fill_between(cn._U_GRID, combined, color="#c0392b", alpha=0.45, label="clipped sets, combined (max)")
    ax[2].axvline(u, color="k", lw=2, label=f"centroid u = {u:+.3f}")
    ax[2].set(title="output: clip each set at its rule strength, take the union", xlabel="u (normalised output)")
    ax[2].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    return savefig(fig_, "worked3.png")


def fig_tracking():
    d = load("main")
    fig_, ax = plt.subplots(2, 2, figsize=(12.5, 7))
    for j in range(7):
        ax[0, 0].plot(d["t"], np.degrees(d["q_des"][:, j]), color=f"C{j}", lw=1.6)
        ax[0, 0].plot(d["t"], np.degrees(d["q"][:, j]), color=f"C{j}", ls="--", lw=1.2)
        ax[0, 1].plot(d["t"], np.degrees(d["q"][:, j] - d["q_des"][:, j]), label=JOINT_NAMES[j])
        ax[1, 0].plot(d["t"], d["tau"][:, j], label=f"tau{j + 1}")
    ee = [np.linalg.norm(cn.forward_kinematics(a)[:3, 3] - cn.forward_kinematics(b)[:3, 3]) * 1000 for a, b in zip(d["q"], d["q_des"])]
    ax[1, 1].plot(d["t"], ee)
    ax[0, 0].set(title="desired (solid) vs actual (dashed) joint angles", xlabel="time [s]", ylabel="deg")
    ax[0, 1].set(title="joint error (actual - desired)", xlabel="time [s]", ylabel="deg", ylim=(-8, 8)); ax[0, 1].legend(fontsize=7, ncol=4)
    ax[1, 0].set(title="commanded torques", xlabel="time [s]", ylabel="Nm"); ax[1, 0].legend(fontsize=7, ncol=4)
    ax[1, 1].set(title="hand position error", xlabel="time [s]", ylabel="mm")
    for a in ax.flat:
        a.grid(alpha=0.3); a.axvline(T_MOVE, color="k", ls=":", lw=1)
    return savefig(fig_, "tracking3.png")


def fig_ablation():
    fig_, ax = plt.subplots(1, 2, figsize=(12.5, 4))
    labels = [n for _, n in RUNS]
    vals = [RM[k]["rms_late"] for k, _ in RUNS]
    colors = ["#1b7f79", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#c0392b"]
    ax[0].barh(range(len(vals)), vals, color=colors)
    ax[0].set_yticks(range(len(vals))); ax[0].set_yticklabels(labels, fontsize=8); ax[0].invert_yaxis()
    ax[0].set_xscale("log"); ax[0].set(xlabel="RMS tracking error after start-up [deg] (log scale)", title="same trajectory, real bridge, six controllers")
    for i, v in enumerate(vals):
        ax[0].text(v * 1.1, i, f"{v:.2f}", va="center", fontsize=8)
    for (k, name), c in zip(RUNS, colors):
        if k in ("main", "flc_only", "model_only", "gravity_only"):
            d = load(k)
            ax[1].plot(d["t"], np.linalg.norm(np.degrees(d["q"] - d["q_des"]), axis=1), label=name, color=c, lw=1.8)
    ax[1].set(title="joint error norm over time", xlabel="time [s]", ylabel="deg", yscale="log"); ax[1].legend(fontsize=7)
    ax[1].axvline(T_MOVE, color="k", ls=":", lw=1)
    for a in ax:
        a.grid(alpha=0.3, which="both")
    return savefig(fig_, "ablation3.png")


def fig_perjoint():
    fig_, ax = plt.subplots(figsize=(8.5, 3.6))
    w = 0.14
    for i, (k, name) in enumerate(RUNS[:5]):
        ax.bar(np.arange(7) + (i - 2) * w, RM[k]["pj_late"], w, label=name)
    ax.set(xticks=range(7), xticklabels=JOINT_NAMES, ylabel="RMS error after start-up [deg]", title="per-joint error (model-only is off the chart: 11-47 deg)")
    ax.set_yscale("log"); ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")
    return savefig(fig_, "perjoint3.png")


def fig_two_link():
    """Small 2-link planar arm used to explain the Lagrangian steps (formulas in the text)."""
    fig_, ax = plt.subplots(figsize=(4.6, 3.6))
    th1, th2 = 0.9, 0.7
    p1 = np.array([np.cos(th1), np.sin(th1)])
    p2 = p1 + np.array([np.cos(th1 + th2), np.sin(th1 + th2)]) * 0.8
    ax.plot([0, p1[0], p2[0]], [0, p1[1], p2[1]], "-", lw=3, color="#1f3a5f")
    ax.plot([0], [0], "ko", ms=7)
    ax.plot(*p1, "s", color="#c0392b", ms=11); ax.plot(*p2, "s", color="#c0392b", ms=11)
    ax.text(-0.15, -0.2, "joint 1 (q1)", fontsize=8)
    ax.text(p1[0] + 0.08, p1[1] - 0.02, "m1 at the end of link 1\n(joint 2 is here, angle q2)", fontsize=8, color="#c0392b")
    ax.text(p2[0] + 0.08, p2[1] - 0.02, "m2 at the end of link 2", fontsize=8, color="#c0392b")
    ax.text(0.2, 0.18, "l1", fontsize=9); ax.text((p1[0] + p2[0]) / 2 - 0.2, (p1[1] + p2[1]) / 2 + 0.02, "l2", fontsize=9)
    ax.annotate("", xy=(-0.55, 0.05), xytext=(-0.55, 0.75), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(-0.52, 0.35, "gravity", fontsize=8)
    ax.set_xlim(-0.8, 2.6); ax.set_ylim(-0.4, 2.1)
    ax.set_aspect("equal"); ax.axis("off")
    return savefig(fig_, "twolink3.png")
