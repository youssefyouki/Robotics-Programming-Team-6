#!/usr/bin/env python3
"""
Challenge 3, part (a): compare the analytical Lagrangian model with MuJoCo's built-in dynamics.

For random joint states (q, dq, ddq) inside the joint limits it checks
  1. M(q)                  vs MuJoCo's joint-space inertia matrix (mj_fullM)
  2. C(q,dq) dq + g(q)     vs MuJoCo's bias force qfrc_bias
  3. tau = M ddq + C dq + g  vs MuJoCo's INVERSE dynamics (mj_inverse): the torque MuJoCo says is needed to produce ddq
  4. the DH frames of Challenge 1 vs the body frames of panda.xml: they must differ by one constant transform per link
  5. energy: d/dt( kinetic energy ) = dq . (tau - g)  for the model (the Coriolis terms do no work)

Run (needs ROS sourced and mujoco importable):
  source /opt/ros/jazzy/setup.bash && source install/setup.bash
  export PYTHONPATH=$PWD/.venv/lib/python3.12/site-packages:$PYTHONPATH
  python3 src/panda_kinematics/tools/check_dynamics.py
"""
import os
import sys
import time

import numpy as np
import mujoco

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from panda_kinematics import controller_node as cn

SCENE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "mujoco_ros2", "model", "franka_emika_panda", "scene.xml")
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name:<62s} {detail}")


model = mujoco.MjModel.from_xml_path(SCENE)
data = mujoco.MjData(model)
rng = np.random.default_rng(0)
damping = model.dof_damping[:7]
lo, hi = cn.Q_LIM_LOW, cn.Q_LIM_HIGH

e_M, e_C, e_tau, e_rel = [], [], [], []
samples = 0
while samples < 300:
    q = rng.uniform(lo * 0.95, hi * 0.95)
    dq = rng.uniform(-2.0, 2.0, 7)
    ddq = rng.uniform(-8.0, 8.0, 7)
    data.qpos[:7], data.qpos[7:] = q, cn.FINGER_OPEN          # gripper open, as in the experiments
    data.qvel[:], data.qvel[:7] = 0.0, dq
    mujoco.mj_forward(model, data)
    if data.ncon > 0:
        continue                                               # skip poses in contact (the model has no contact forces)
    samples += 1
    M9 = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, M9)
    M, Cg = cn.compute_dynamics(q, dq, ddq)
    e_M.append(np.abs(M9[:7, :7] - M).max())
    e_C.append(np.abs(data.qfrc_bias[:7] - Cg).max())
    # inverse dynamics: qfrc_inverse = M qacc + bias - passive (joint damping) when nothing is in contact
    data.qacc[:], data.qacc[:7] = 0.0, ddq
    mujoco.mj_inverse(model, data)
    tau_mj = data.qfrc_inverse[:7] + 0.0
    tau_model = M @ ddq + Cg + damping * dq                    # model + the joint damping that panda.xml adds
    e_tau.append(np.abs(tau_mj - tau_model).max())
    e_rel.append(np.abs(tau_mj - tau_model).max() / max(np.abs(tau_mj).max(), 1e-9))

check("1. M(q) matches MuJoCo's inertia matrix (300 random poses)", max(e_M) < 1e-6, f"max err {max(e_M):.1e} kg m^2")
check("2. C dq + g matches MuJoCo's bias force", max(e_C) < 1e-6, f"max err {max(e_C):.1e} Nm")
check("3. M ddq + C dq + g matches MuJoCo inverse dynamics", max(e_tau) < 1e-6, f"max err {max(e_tau):.1e} Nm (largest relative {max(e_rel):.1e})")

# 4. DH frames vs XML body frames
T_body = np.eye(4)
chain = []
for pos, quat in cn._BODY_OFFSETS:
    step = np.eye(4)
    step[:3, :3] = cn._quat_to_R(*quat)
    step[:3, 3] = pos
    chain.append(step)
C0 = []
frames0 = cn._frames(np.zeros(7))
T = np.eye(4)
for i, step in enumerate(chain):
    T = T @ step
    C0.append(np.linalg.inv(frames0[i + 1]) @ T)
worst = 0.0
for _ in range(100):
    q = rng.uniform(lo, hi)
    frames = cn._frames(q)
    T = np.eye(4)
    for i, step in enumerate(chain):
        rot = np.eye(4)
        c, s = np.cos(q[i]), np.sin(q[i])
        rot[:2, :2] = [[c, -s], [s, c]]
        T = T @ step @ rot
        worst = max(worst, np.abs(T - frames[i + 1] @ C0[i]).max())
check("4. XML body frame = DH frame x constant transform, every link", worst < 1e-12, f"max err {worst:.1e}")

# 5. energy balance of the model:  d/dt(1/2 dq' M dq) = dq' (tau - g)
h = 1e-6
worst = 0.0
for _ in range(100):
    q = rng.uniform(lo * 0.9, hi * 0.9)
    dq = rng.uniform(-2, 2, 7)
    ddq = rng.uniform(-5, 5, 7)
    M, Cg = cn.compute_dynamics(q, dq, ddq)
    g = cn.compute_dynamics(q, np.zeros(7), ddq)[1]
    tau = M @ ddq + Cg
    KE = lambda a, b: 0.5 * b @ (cn.compute_dynamics(a, b, ddq)[0] - cn.ARMATURE * np.eye(7)) @ b
    dKE = (KE(q + h * dq, dq + h * ddq) - KE(q - h * dq, dq - h * ddq)) / (2 * h)
    # the rotor inertia is excluded from KE above, so remove its (simple) contribution from the power balance
    power = dq @ (tau - g) - cn.ARMATURE * dq @ ddq
    worst = max(worst, abs(dKE - power))
check("5. power balance: d(kinetic energy)/dt = dq.(tau - g)", worst < 1e-5, f"max err {worst:.1e} W")

t0 = time.perf_counter()
for _ in range(200):
    cn.compute_dynamics(q, dq, ddq)
ms = (time.perf_counter() - t0) / 200 * 1000
check("6. compute_dynamics fits in the 10 ms control period", ms < 8.0, f"{ms:.1f} ms per call")

print()
print("ALL CHECKS PASSED" if all(results) else "SOME CHECKS FAILED")
sys.exit(0 if all(results) else 1)
