#!/usr/bin/env python3
"""
Loads the Koch v1.1 MuJoCo model for Milestone 03 using only the Python bindings.

- adds an 'ee' site at the origin of the wrist_roll body (the FK end-effector point)
- optionally builds the industrial environment (conveyor, table, person, chair, tool rack)

The arm is small (reach ~0.25 m), so the environment is scaled to a desktop pick-and-place cell.
"""
import os
import numpy as np
import mujoco

_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE_SCENE = os.path.join(_HERE, "..", "..", "mujoco_ros2", "model",
                             "low_cost_robot_arm", "scene.xml")


def default_scene_xml():
    """Scene of the installed mujoco_ros2 package, falling back to the source tree."""
    try:
        from ament_index_python.packages import get_package_share_directory
        path = os.path.join(get_package_share_directory("mujoco_ros2"),
                            "model", "low_cost_robot_arm", "scene.xml")
        if os.path.exists(path):
            return path
    except Exception:
        pass
    return os.path.normpath(_SOURCE_SCENE)


# Conveyor geometry (belt runs along world Y, in front of the arm, i.e. -X side)
BELT_X = -0.17
BELT_HALF_WIDTH = 0.035
BELT_HALF_LENGTH = 0.30
BELT_TOP_Z = 0.03


def _add_environment(spec):
    w = spec.worldbody

    def box(name, pos, size, rgba, mocap=False):
        if mocap:
            body = w.add_body(name=name, pos=pos, mocap=True)
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=size, rgba=rgba,
                          contype=0, conaffinity=0)
            return body
        w.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size, rgba=rgba,
                   contype=0, conaffinity=0)

    # --- conveyor: belt, side rails, legs, end rollers
    box("belt", [BELT_X, 0, BELT_TOP_Z - 0.005], [BELT_HALF_WIDTH, BELT_HALF_LENGTH, 0.005],
        [0.15, 0.15, 0.17, 1])
    for s in (-1, 1):
        box(f"rail_{s}", [BELT_X + s * (BELT_HALF_WIDTH + 0.004), 0, BELT_TOP_Z - 0.003],
            [0.004, BELT_HALF_LENGTH, 0.008], [0.7, 0.7, 0.75, 1])
        for ly in (-0.26, 0.26):
            box(f"leg_{s}_{ly}", [BELT_X + s * 0.03, ly, (BELT_TOP_Z - 0.01) / 2],
                [0.004, 0.004, (BELT_TOP_Z - 0.01) / 2], [0.4, 0.4, 0.45, 1])
    for ly in (-1, 1):
        w.add_geom(name=f"roller_{ly}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   pos=[BELT_X, ly * BELT_HALF_LENGTH, BELT_TOP_Z - 0.008],
                   size=[0.008, BELT_HALF_WIDTH, 0], euler=[0, 90, 0],
                   rgba=[0.5, 0.5, 0.55, 1], contype=0, conaffinity=0)

    # --- workpieces riding on the belt (mocap so the demo can move them)
    colors = [[0.85, 0.2, 0.2, 1], [0.2, 0.4, 0.85, 1], [0.95, 0.8, 0.15, 1], [0.2, 0.7, 0.3, 1]]
    for i, c in enumerate(colors):
        box(f"workpiece_{i}", [BELT_X, -0.25 + i * 0.12, BELT_TOP_Z + 0.01], [0.01, 0.01, 0.01],
            c, mocap=True)

    # --- bin on the arm's right side where parts are dropped
    box("bin_floor", [0.0, 0.17, 0.003], [0.04, 0.04, 0.003], [0.35, 0.35, 0.4, 1])
    for name, pos, size in [("bin_a", [0.04, 0.17, 0.015], [0.002, 0.04, 0.015]),
                            ("bin_b", [-0.04, 0.17, 0.015], [0.002, 0.04, 0.015]),
                            ("bin_c", [0.0, 0.21, 0.015], [0.04, 0.002, 0.015]),
                            ("bin_d", [0.0, 0.13, 0.015], [0.04, 0.002, 0.015])]:
        box(name, pos, size, [0.35, 0.35, 0.4, 1])

    # --- tool rack behind the arm
    box("rack_back", [0.14, 0.0, 0.05], [0.004, 0.09, 0.05], [0.55, 0.45, 0.3, 1])
    for i, y in enumerate((-0.06, -0.02, 0.02, 0.06)):
        w.add_geom(name=f"tool_{i}", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                   pos=[0.125, y, 0.06], size=[0.004, 0.02, 0], euler=[0, 90, 0],
                   rgba=[0.8, 0.8, 0.85, 1], contype=0, conaffinity=0)

    # --- operator (worker) standing at the end of the belt, and a chair
    person_pos = [-0.17, -0.45, 0.0]
    w.add_geom(name="person_legs", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
               fromto=[person_pos[0], person_pos[1], 0.02, person_pos[0], person_pos[1], 0.17],
               size=[0.035, 0, 0], rgba=[0.2, 0.25, 0.5, 1], contype=0, conaffinity=0)
    w.add_geom(name="person_torso", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
               fromto=[person_pos[0], person_pos[1], 0.19, person_pos[0], person_pos[1], 0.36],
               size=[0.045, 0, 0], rgba=[0.95, 0.55, 0.1, 1], contype=0, conaffinity=0)
    w.add_geom(name="person_head", type=mujoco.mjtGeom.mjGEOM_SPHERE,
               pos=[person_pos[0], person_pos[1], 0.43], size=[0.04, 0, 0],
               rgba=[0.9, 0.75, 0.6, 1], contype=0, conaffinity=0)
    w.add_geom(name="person_helmet", type=mujoco.mjtGeom.mjGEOM_SPHERE,
               pos=[person_pos[0], person_pos[1], 0.445], size=[0.042, 0, 0],
               rgba=[1.0, 0.9, 0.1, 1], contype=0, conaffinity=0)

    chair = [0.2, -0.3, 0.0]
    box("chair_seat", [chair[0], chair[1], 0.075], [0.05, 0.05, 0.006], [0.3, 0.3, 0.35, 1])
    box("chair_back", [chair[0] + 0.045, chair[1], 0.13], [0.006, 0.05, 0.055], [0.3, 0.3, 0.35, 1])
    for dx in (-0.04, 0.04):
        for dy in (-0.04, 0.04):
            box(f"chair_leg_{dx}_{dy}", [chair[0] + dx, chair[1] + dy, 0.035],
                [0.004, 0.004, 0.035], [0.2, 0.2, 0.22, 1])


def load_model(environment=False, scene_xml=None):
    """Returns (model, data, ee_site_id). The 'ee' site marks the FK end-effector point."""
    spec = mujoco.MjSpec.from_file(scene_xml or default_scene_xml())
    spec.body("gripper_static_finger").add_site(
        name="ee", pos=[0, 0, 0], size=[0.004, 0, 0], rgba=[1, 0, 0, 1])
    if environment:
        _add_environment(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    return model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee")


def site_velocity(model, data, site_id):
    """World-frame linear and angular velocity of the ee site: (v, w)."""
    res = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, site_id, res, 0)
    return res[3:].copy(), res[:3].copy()


def site_acceleration(model, data, site_id):
    """World-frame linear and angular acceleration of the ee site: (a, alpha).
    Requires data.qacc to be set; recomputes accelerations from it."""
    mujoco.mj_rnePostConstraint(model, data)
    res = np.zeros(6)
    mujoco.mj_objectAcceleration(model, data, mujoco.mjtObj.mjOBJ_SITE, site_id, res, 0)
    # MuJoCo returns proper acceleration (accelerometer-like), i.e. true acceleration minus
    # gravity; add gravity back to get the kinematic acceleration of the point.
    return res[3:] + model.opt.gravity, res[:3].copy()


BELT_SPEED = 0.02   # m/s, workpieces ride along +Y


def move_belt(model, data, dt, speed=BELT_SPEED):
    """Workpieces (mocap bodies) ride the belt and wrap around."""
    for i in range(4):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"workpiece_{i}")
        mid = model.body_mocapid[body]
        y = data.mocap_pos[mid][1] + speed * dt
        if y > BELT_HALF_LENGTH - 0.02:
            y = -BELT_HALF_LENGTH + 0.02
        data.mocap_pos[mid][1] = y
