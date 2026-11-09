#!/usr/bin/env python3
"""
Milestone 03 - standalone (no ROS) environment demo in the MuJoCo viewer.

The arm performs the Cartesian pick-and-place cycle (conveyor -> bin) driven by closed-loop
inverse velocity kinematics:   qd = J+ ( v_d + Kp (p_d - p(q)) )
The ROS 2 version of the same pipeline is `ros2 launch milestone_03 milestone_03.launch.py`.

Note: the controlled point is the wrist_roll frame origin (the FK end-effector), so the
gripper open/close is illustrative rather than a physical grasp.

Run:   .venv/bin/python "Milestone 03/demo_viewer.py"            (interactive viewer)
       .venv/bin/python "Milestone 03/demo_viewer.py" --headless  (no window, prints stats)
"""
import os
import sys
import time
import numpy as np
import mujoco

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "milestone_03"))
from milestone_03 import kinematics as K
from milestone_03 import sim_model as S
from milestone_03.trajectories import PickPlaceTrajectory, GRIPPER_OPEN

KP = 20.0                       # task-space position feedback gain [1/s]


def run(headless):
    model, data, site = S.load_model(environment=True)
    dt = model.opt.timestep
    q_cmd = np.zeros(5)
    data.ctrl[5] = GRIPPER_OPEN
    mujoco.mj_forward(model, data)

    traj = PickPlaceTrajectory(K.ee_position(q_cmd), loop=not headless)
    viewer = None
    if not headless:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data)
        viewer.cam.lookat[:] = [-0.05, 0.05, 0.1]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.0, 215, -25

    max_err, next_print, wall0 = 0.0, 0.0, time.time()
    try:
        while (viewer is None and data.time < traj.duration) or (viewer is not None and viewer.is_running()):
            p_d, v_d, a_d, grip = traj.sample(data.time)
            qd = K.inverse_velocity(q_cmd, v_d + KP * (p_d - K.ee_position(q_cmd)))
            q_cmd = np.clip(q_cmd + qd * dt, K.JOINT_LIMITS[:, 0], K.JOINT_LIMITS[:, 1])
            data.ctrl[:5], data.ctrl[5] = q_cmd, grip
            mujoco.mj_step(model, data)
            S.move_belt(model, data, dt)

            if data.time >= next_print:
                v_sim, _ = S.site_velocity(model, data, site)
                err = np.linalg.norm(data.site_xpos[site] - p_d)
                max_err = max(max_err, err)
                print(f"t={data.time:6.2f}s EE={np.round(data.site_xpos[site], 3)} "
                      f"track_err={err*1000:5.1f}mm |v|={np.linalg.norm(v_sim):.4f} m/s")
                next_print += 0.5
            if viewer is not None:
                viewer.sync()
                lag = data.time - (time.time() - wall0)
                if lag > 0:
                    time.sleep(lag)
    finally:
        if viewer is not None:
            viewer.close()
    print(f"\nmax tracking error: {max_err*1000:.1f} mm")


if __name__ == "__main__":
    run(headless="--headless" in sys.argv)
