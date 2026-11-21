#!/usr/bin/env python3
"""
Sorting cell for Milestone 04 (pure MuJoCo Python API).

Koch v1.1 + a static conveyor with three coloured boxes (red, green, blue) in front of the
arm and three bins around it. Boxes are free bodies, so they are grasped by contact.
Layout is chosen from the reachability of a top-down grasp (TCP within ~0.14 m of the base
axis) and from the joint limits: base_rotation is limited to +-2.2 rad, so a bin exactly
behind the robot is unreachable; the third bin sits rear-right (the right side reaches
further round than the left because of the shoulder's sideways offset).

load_cell(actuation): 'position' keeps the built-in joint position servos;
                      'torque' turns the five arm actuators into direct torque inputs
                      (force = ctrl), used by the motor-level controllers.
"""
import numpy as np
import mujoco
from milestone_03 import sim_model as S
from milestone_04.tcp_kinematics import TCP_LOCAL, BOX_SIZE

J1_XY = np.array([0.012, 0.0])          # base_rotation axis in the world frame
BELT_CENTER = np.array([-0.103, 0.0])
BELT_HALF = np.array([0.035, 0.14, 0.01])
BELT_TOP_Z = 0.02
BOX_Z = BELT_TOP_Z + BOX_SIZE / 2       # box centre height when resting on the belt

BOX_COLORS = {"red": [0.85, 0.15, 0.15, 1], "green": [0.15, 0.7, 0.25, 1], "blue": [0.15, 0.3, 0.85, 1]}
# colour -> (belt y position, yaw [rad])
BOX_START = {"red": (-0.05, 0.0), "green": (0.0, 0.35), "blue": (0.05, -0.3)}

BIN_RADIUS = 0.115
BIN_OUTER = 0.07
BIN_WALL = 0.004
BIN_WALL_H = 0.024
# colour -> base yaw q1 pointing at the bin (arm extends along (-cos q1, sin q1))
BIN_YAW = {"red": -1.35, "green": 1.5, "blue": -2.15}
COLORS = list(BOX_COLORS)

ARM_JOINTS = 5


def bin_center(color):
    q1 = BIN_YAW[color]
    return J1_XY + BIN_RADIUS * np.array([-np.cos(q1), np.sin(q1)])


def box_start_position(color):
    y, _ = BOX_START[color]
    return np.array([BELT_CENTER[0], y, BOX_Z])


def _add_cell(spec):
    w = spec.worldbody
    box = mujoco.mjtGeom.mjGEOM_BOX

    w.add_geom(name="belt", type=box, pos=[*BELT_CENTER, BELT_TOP_Z - BELT_HALF[2]],
               size=BELT_HALF, rgba=[0.18, 0.18, 0.2, 1], friction=[1.0, 0.005, 0.0001])
    for s in (-1, 1):
        for ly in (-0.12, 0.12):
            w.add_geom(name=f"belt_leg_{s}_{ly}", type=box, contype=0, conaffinity=0,
                       pos=[BELT_CENTER[0] + s * 0.03, ly, (BELT_TOP_Z - 0.02) / 2 + 0.002],
                       size=[0.004, 0.004, 0.001], rgba=[0.4, 0.4, 0.45, 1])

    for color in COLORS:
        c = bin_center(color)
        rgba = BOX_COLORS[color][:3] + [1]
        light = [min(1.0, v + 0.35) for v in rgba[:3]] + [1]
        half = BIN_OUTER / 2
        w.add_geom(name=f"bin_{color}_floor", type=box, pos=[c[0], c[1], BIN_WALL / 2],
                   size=[half, half, BIN_WALL / 2], rgba=light, friction=[1.0, 0.005, 0.0001])
        for name, off, size in [("n", [0, half - BIN_WALL / 2], [half, BIN_WALL / 2]),
                                ("s", [0, -half + BIN_WALL / 2], [half, BIN_WALL / 2]),
                                ("e", [half - BIN_WALL / 2, 0], [BIN_WALL / 2, half]),
                                ("w", [-half + BIN_WALL / 2, 0], [BIN_WALL / 2, half])]:
            w.add_geom(name=f"bin_{color}_{name}", type=box,
                       pos=[c[0] + off[0], c[1] + off[1], BIN_WALL + BIN_WALL_H / 2],
                       size=[size[0], size[1], BIN_WALL_H / 2], rgba=rgba)

    for color in COLORS:
        y, yaw = BOX_START[color]
        body = w.add_body(name=f"box_{color}", pos=[BELT_CENTER[0], y, BOX_Z + 1e-3],
                          quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        body.add_freejoint()
        body.add_geom(name=f"box_{color}_geom", type=box, size=[BOX_SIZE / 2] * 3,
                      rgba=BOX_COLORS[color], friction=[1.0, 0.01, 0.001], condim=4,
                      mass=0.01)


def load_cell(actuation="position", scene_xml=None, with_cell=True, offscreen=None):
    """Returns (model, data). Sites: 'ee' (wrist_roll origin) and 'tcp' (between the pads).
    with_cell=False loads the bare robot (used by model-based controllers).
    offscreen=(width, height) enlarges the offscreen framebuffer for rendered images."""
    spec = mujoco.MjSpec.from_file(scene_xml or S.default_scene_xml())
    finger = spec.body("gripper_static_finger")
    finger.add_site(name="ee", pos=[0, 0, 0], size=[0.004, 0, 0], rgba=[1, 0, 0, 1])
    finger.add_site(name="tcp", pos=TCP_LOCAL.tolist(), size=[0.003, 0, 0], rgba=[1, 1, 0, 1])
    if with_cell:
        _add_cell(spec)
    if offscreen:
        spec.visual.global_.offwidth = max(offscreen[0], 640)
        spec.visual.global_.offheight = max(offscreen[1], 480)
        spec.visual.quality.offsamples = 8            # multisample anti-aliasing for the offscreen view
    model = spec.compile()
    if actuation == "torque":
        for i in range(ARM_JOINTS):       # force = ctrl (direct joint torque)
            model.actuator_gainprm[i, :] = 0.0
            model.actuator_gainprm[i, 0] = 1.0
            model.actuator_biasprm[i, :] = 0.0
            model.actuator_biastype[i] = int(mujoco.mjtBias.mjBIAS_NONE)
            model.actuator_ctrllimited[i] = 0
    elif actuation != "position":
        raise ValueError("actuation must be 'position' or 'torque'")
    data = mujoco.MjData(model)
    return model, data


def box_body_ids(model):
    return {c: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"box_{c}") for c in COLORS}


def box_poses(model, data):
    """{colour: (position[3], yaw)} read from the simulator."""
    out = {}
    for c, b in box_body_ids(model).items():
        R = data.xmat[b].reshape(3, 3)
        out[c] = (data.xpos[b].copy(), float(np.arctan2(R[1, 0], R[0, 0])))
    return out


def box_in_bin(pos, color):
    """True if a box centre lies inside the bin of the given colour."""
    c = bin_center(color)
    half = BIN_OUTER / 2 - BIN_WALL
    return abs(pos[0] - c[0]) < half and abs(pos[1] - c[1]) < half and pos[2] < BIN_WALL + BIN_WALL_H


def reset_to_home(model, data, q=None):
    mujoco.mj_resetData(model, data)
    if q is not None:
        data.qpos[:ARM_JOINTS] = q
    mujoco.mj_forward(model, data)
