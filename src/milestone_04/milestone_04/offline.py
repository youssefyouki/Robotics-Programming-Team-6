#!/usr/bin/env python3
"""
In-process (no ROS) closed-loop run of the sorting cycle: plan -> controller -> MuJoCo.

Used to tune and compare the controllers and to produce the report figures; the ROS 2 nodes
run exactly the same planner, controllers and cell model.
"""
import numpy as np
import mujoco

from milestone_03 import kinematics as K
from milestone_04 import cell as C, control, planning as P, tcp_kinematics as TK

CTRL_STEPS = 2      # control period = 2 simulator steps = 4 ms (250 Hz, same as the ROS nodes)


def evaluate_boxes(model, data):
    poses = C.box_poses(model, data)
    return {c: C.box_in_bin(p, c) for c, (p, _) in poses.items()}, poses


def simulate(controller_name, plan=None, tail=1.5, callback=None, verbose=False, offscreen=None):
    """callback(model, data, k, segment_name) is called once per control tick (e.g. to render frames)."""
    ctrl = control.make_controller(controller_name)
    model, data = C.load_cell(actuation=ctrl.kind, offscreen=offscreen)
    C.reset_to_home(model, data)
    tcp_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    if plan is None:
        plan = P.build_sorting_plan(np.zeros(5), C.box_poses(model, data))
    data.ctrl[5] = plan.grip[0]
    dt = model.opt.timestep
    assert abs(dt - P.DT) < 1e-12, "plan sample period must equal the simulator time step"

    n_ctrl = int((plan.duration + tail) / (CTRL_STEPS * dt))
    log = {k: [] for k in ["t", "q", "qd", "q_ref", "qd_ref", "tau", "tcp", "tcp_ref", "grip", "grip_ref"]}
    for k in range(n_ctrl):
        t = data.time
        q_ref, qd_ref, qdd_ref, g_ref = plan.sample(t)
        q, qd = data.qpos[:5].copy(), data.qvel[:5].copy()
        u = ctrl(q, qd, q_ref, qd_ref, qdd_ref, grip_q=data.qpos[5])
        data.ctrl[:5] = u
        data.ctrl[5] = g_ref
        log["t"].append(t); log["q"].append(q); log["qd"].append(qd)
        log["q_ref"].append(q_ref); log["qd_ref"].append(qd_ref); log["tau"].append(u)
        log["tcp"].append(data.site_xpos[tcp_site].copy()); log["tcp_ref"].append(TK.tcp_position(q_ref))
        log["grip"].append(data.qpos[5]); log["grip_ref"].append(g_ref)
        for _ in range(CTRL_STEPS):
            mujoco.mj_step(model, data)
        name, color = plan.segment_at(t)
        if callback is not None:
            callback(model, data, k, name)
        if verbose and k % 250 == 0:
            print(f"t={t:5.1f}s {name:9s}", end="\r")
    log = {k: np.array(v) for k, v in log.items()}
    in_bin, poses = evaluate_boxes(model, data)
    return plan, log, in_bin, poses, (model, data)


def metrics(plan, log):
    e = log["q_ref"] - log["q"]
    tcp_err = np.linalg.norm(log["tcp"] - log["tcp_ref"], axis=1)
    return {
        "joint_rms_rad": np.sqrt((e**2).mean(0)),
        "joint_max_rad": np.abs(e).max(0),
        "tcp_rms_mm": float(np.sqrt((tcp_err**2).mean()) * 1000),
        "tcp_max_mm": float(tcp_err.max() * 1000),
        "torque_rms": np.sqrt((log["tau"]**2).mean(0)),
    }
