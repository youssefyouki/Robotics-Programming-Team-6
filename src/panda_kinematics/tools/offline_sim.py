#!/usr/bin/env python3
"""
Offline (no ROS bridge) simulation of the Challenge 3 control loop, used to tune the fuzzy controller quickly and to
test ideas in seconds. It reproduces what the real bridge does:
  * the same unmodified Panda model, 1 ms steps, Euler integrator
  * the controller runs every 10 ms on the latest state; its torque is HELD until the next command
  * a commanded torque is turned into the position-servo target that produces exactly that torque
  * an optional latency (ms) between reading the state and the torque reaching the actuators (the real loop has about 5-10 ms)
  * until the first command the initial pose is held; after the trajectory the template holds with gravity compensation only

  python3 src/panda_kinematics/tools/offline_sim.py [--delay 8]
"""
import argparse
import os
import sys

import numpy as np
import mujoco

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from panda_kinematics import controller_node as cn

SCENE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "mujoco_ros2", "model", "franka_emika_panda", "scene.xml")


def simulate(control_fn, q_init=None, delay_ms=0, hold_s=1.5, t_start=0.5):
    """Returns (time, q, dq, q_des, tau, n_motion) sampled at the controller rate."""
    t, q_traj, dq_traj, ddq_traj = cn.plan_trajectory(cn.Q_START, cn.Q_END, cn.TRAJ_DURATION, cn.TRAJ_DT)
    m = mujoco.MjModel.from_xml_path(SCENE)
    m.opt.timestep, m.opt.integrator = 0.001, mujoco.mjtIntegrator.mjINT_EULER
    d = mujoco.MjData(m)
    if q_init is not None:
        d.qpos[:7] = q_init
    d.qpos[7:] = 0.0
    q_hold = d.qpos[:7].copy()
    cn._flc_state["u"][:] = 0.0
    n_plan = len(t)                                          # motion + the controlled stop appended by the planner
    n_motion = int(np.where(np.abs(dq_traj).max(axis=1) > 0)[0][-1]) + 2     # samples of the motion itself
    n_total = n_plan + int(hold_s / cn.TRAJ_DT)
    pending, tau_applied, tau, k, log = [], None, np.zeros(7), 0, []
    t_next = t_start
    for step in range(int((t_start + n_total * cn.TRAJ_DT) / 0.001)):
        if d.time >= t_next - 1e-9:
            q, dq = d.qpos[:7].copy(), d.qvel[:7].copy()
            if k < n_plan:
                tau = control_fn(q, dq, q_traj[k], dq_traj[k], ddq_traj[k])
                q_des = q_traj[k]
            else:
                tau = cn.compute_dynamics(q, np.zeros(7), np.zeros(7))[1]       # the template's hold: gravity only
                q_des = q_traj[-1]
            pending.append((step + delay_ms, tau.copy()))
            log.append((d.time, q, dq, q_des, tau.copy()))
            k += 1
            t_next += cn.TRAJ_DT
        while pending and pending[0][0] <= step:
            tau_applied = pending.pop(0)[1]
        for i in range(7):                                                      # torque -> servo target (the bridge does the same)
            if tau_applied is None:
                d.ctrl[i] = q_hold[i]
            else:
                b = m.actuator_biasprm[i, :3]
                d.ctrl[i] = (tau_applied[i] - (b[0] + b[1] * d.qpos[i] + b[2] * d.qvel[i])) / m.actuator_gainprm[i, 0]
        d.ctrl[7] = cn.GRIPPER_CTRL
        mujoco.mj_step(m, d)
    cols = list(zip(*log))
    return (np.array(cols[0]), np.array(cols[1]), np.array(cols[2]), np.array(cols[3]), np.array(cols[4]), n_motion)


def report(label, res):
    _, q, dq, q_des, tau, n = res
    e = np.degrees(q - q_des)[:n]
    late = e[int(0.3 / cn.TRAJ_DT):]
    print(f"{label:46s} rms {np.sqrt((e ** 2).mean()):6.2f}  max {np.abs(e).max():6.2f} | after start-up rms {np.sqrt((late ** 2).mean()):6.2f} max {np.abs(late).max():6.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay", type=int, default=8, help="latency in ms between state and torque (default 8)")
    a = ap.parse_args()
    ctrl = cn.compute_torques

    def variant(**flags):
        def fn(q, dq, qd, dqd, ddqd):
            saved = {k: getattr(cn, k) for k in flags}
            for k, v in flags.items():
                setattr(cn, k, v)
            try:
                return ctrl(q, dq, qd, dqd, ddqd)
            finally:
                for k, v in saved.items():
                    setattr(cn, k, v)
        return fn

    print(f"errors in degrees over the planned motion, latency {a.delay} ms\n")
    report("model + FLC (submitted controller)", simulate(variant(), delay_ms=a.delay))
    report("model only (no FLC)", simulate(variant(USE_FLC=False), delay_ms=a.delay))
    report("FLC only (no model)", simulate(variant(USE_MODEL=False), delay_ms=a.delay))
    report("decoupled inertia (diagonal M) + FLC", simulate(variant(MODEL_TERMS="decoupled"), delay_ms=a.delay))
    report("no Coriolis term + FLC", simulate(variant(MODEL_TERMS="no_coriolis"), delay_ms=a.delay))
    report("gravity only + FLC", simulate(variant(MODEL_TERMS="gravity"), delay_ms=a.delay))
    print("\nWhy does the model alone drift? the unmodelled joint damping of panda.xml (1 Nms/rad):")
    damping = lambda q, dq, qd, dqd, ddqd: variant(USE_FLC=False)(q, dq, qd, dqd, ddqd) + 1.0 * dqd
    report("model only, perfect start", simulate(variant(USE_FLC=False), q_init=cn.Q_START, delay_ms=0))
    report("model + damping term, perfect start (no FLC)", simulate(damping, q_init=cn.Q_START, delay_ms=0))
    report("model + damping term, 8 ms latency (no FLC)", simulate(damping, q_init=cn.Q_START, delay_ms=8))


if __name__ == "__main__":
    main()
