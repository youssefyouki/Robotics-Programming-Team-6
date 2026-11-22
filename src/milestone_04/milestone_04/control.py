#!/usr/bin/env python3
"""
Motor-level joint controllers for the five arm joints (Milestone 04).

Each controller maps (q, qd, q_ref, qd_ref, qdd_ref) -> command u (5 values):

  ServoController     u = q_ref                       (position set-point; the simulator's built-in
                                                       joint servo kp=100, kv=10 does the rest)
  PDGravityController u = tau = Kp e + Kd ed + g(q)   (joint PD with gravity compensation)
  ComputedTorqueController
                      u = tau = M(q) (qdd_ref + Kd ed + Kp e) + h(q, qd)
                                                      (inverse dynamics: cancels inertia, Coriolis
                                                       and gravity, leaves linear error dynamics
                                                       e'' + Kd e' + Kp e = 0)

with e = q_ref - q, ed = qd_ref - qd. M is the joint-space inertia matrix (including rotor
armature) and h = C(q,qd) qd + g(q) the bias force, both taken from the MuJoCo model of the robot
(the controller keeps its own copy, as a real model-based controller would). A feed-forward term
for the joints' dry friction (friction loss) is added to the torque controllers.
"""
import numpy as np
import mujoco
from milestone_04 import cell as C

N = C.ARM_JOINTS


class _Dynamics:
    """Inertia matrix and bias force of the robot from its own MuJoCo model."""

    def __init__(self):
        self.model, self.d = C.load_cell(actuation="position", with_cell=False)
        self.friction = self.model.dof_frictionloss[:N].copy()
        self._M = np.zeros((self.model.nv, self.model.nv))
        self.grip_q = 0.0

    def update(self, q, qd):
        d = self.d
        d.qpos[:N] = q
        d.qpos[N] = self.grip_q
        d.qvel[:N] = qd
        d.qvel[N] = 0.0
        mujoco.mj_forward(self.model, d)

    def inertia(self):
        try:
            mujoco.mj_fullM(self.model, self.d, self._M)            # MuJoCo >= 4 signature
        except TypeError:
            mujoco.mj_fullM(self.model, self._M, self.d.qM)         # MuJoCo 3.x signature
        return self._M[:N, :N].copy()

    def bias(self):
        return self.d.qfrc_bias[:N].copy()

    def gravity(self):
        """Gravity torque only (zero velocity)."""
        d = self.d
        saved = d.qvel[:N].copy()
        d.qvel[:N] = 0.0
        mujoco.mj_forward(self.model, d)
        g = d.qfrc_bias[:N].copy()
        d.qvel[:N] = saved
        return g


class ServoController:
    kind = "position"
    name = "servo"
    label = "Position servo (built-in)"

    def describe(self):
        return {"name": self.name, "label": self.label, "kind": self.kind,
                "law": "u = q_ref   (set-point of the joint servo)\nservo torque: tau = kp (q_ref - q) - kv qd,  kp = 100, kv = 10",
                "gains": "kp = 100, kv = 10 (per joint, in the model)"}

    def __call__(self, q, qd, q_ref, qd_ref, qdd_ref, grip_q=0.0):
        return np.asarray(q_ref, float).copy()


class PDGravityController:
    kind = "torque"
    name = "pd_gravity"
    label = "PD + gravity compensation"

    def describe(self):
        return {"name": self.name, "label": self.label, "kind": self.kind,
                "law": "tau = Kp e + Kd ed + g(q) + friction feed-forward\ne = q_ref - q,  ed = qd_ref - qd",
                "gains": f"Kp = {self.kp.astype(int).tolist()}, Kd = {self.kd.astype(int).tolist()}"}

    def __init__(self, kp=(110, 110, 110, 70, 45), kd=(16, 16, 16, 11, 8), friction_ff=True):
        self.kp, self.kd = np.array(kp, float), np.array(kd, float)
        self.dyn = _Dynamics()
        self.friction_ff = friction_ff

    def __call__(self, q, qd, q_ref, qd_ref, qdd_ref, grip_q=0.0):
        self.dyn.grip_q = grip_q
        self.dyn.update(q, qd)
        tau = self.kp * (q_ref - q) + self.kd * (qd_ref - qd) + self.dyn.gravity()
        if self.friction_ff:
            tau += self.dyn.friction * np.tanh(qd_ref / 0.02)
        return tau


class ComputedTorqueController:
    kind = "torque"
    name = "computed_torque"
    label = "Computed torque"

    def describe(self):
        return {"name": self.name, "label": self.label, "kind": self.kind,
                "law": "tau = M(q) (qdd_ref + Kd ed + Kp e) + h(q, qd) + friction feed-forward\nerror dynamics: e'' + Kd e' + Kp e = 0",
                "gains": f"wn = {np.sqrt(self.kp).astype(int).tolist()} rad/s (critically damped)"}

    def __init__(self, wn=(22, 22, 22, 22, 22), friction_ff=True):
        wn = np.array(wn, float)
        self.kp, self.kd = wn**2, 2 * wn                  # critically damped error dynamics
        self.dyn = _Dynamics()
        self.friction_ff = friction_ff

    def __call__(self, q, qd, q_ref, qd_ref, qdd_ref, grip_q=0.0):
        self.dyn.grip_q = grip_q
        self.dyn.update(q, qd)
        M, h = self.dyn.inertia(), self.dyn.bias()
        a = qdd_ref + self.kd * (qd_ref - qd) + self.kp * (q_ref - q)
        tau = M @ a + h
        if self.friction_ff:
            tau += self.dyn.friction * np.tanh(qd_ref / 0.02)
        return tau


CONTROLLERS = {"servo": ServoController, "pd_gravity": PDGravityController,
               "computed_torque": ComputedTorqueController}


def make_controller(name):
    return CONTROLLERS[name]()
