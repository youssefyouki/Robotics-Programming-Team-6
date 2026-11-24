#!/usr/bin/env python3
"""
Automated Challenge 3 run: real C++ bridge (TORQUE mode) + controller_node + control_logger.

The experiments switch parts of the controller on and off through three constants in controller_node.py
(USE_MODEL, USE_FLC, MODEL_TERMS). To leave the submitted file untouched this script runs a TEMPORARY copy with
those constants changed.

  source /opt/ros/jazzy/setup.bash && source install/setup.bash
  python3 src/panda_kinematics/tools/run_control.py main
  python3 src/panda_kinematics/tools/run_control.py model_only --no-flc
  python3 src/panda_kinematics/tools/run_control.py flc_only --no-model
  python3 src/panda_kinematics/tools/run_control.py decoupled --terms decoupled

Results: <out>/<label>/{metrics.txt, tracking.png, tracking.csv}
"""
import argparse
import os
import re
import signal
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
NODE_FILE = os.path.join(HERE, "..", "panda_kinematics", "controller_node.py")
SCENE = os.path.join(HERE, "..", "..", "mujoco_ros2", "model", "franka_emika_panda", "scene.xml")


def make_variant(use_model, use_flc, terms):
    text = open(NODE_FILE).read()
    for name, value in (("USE_MODEL", use_model), ("USE_FLC", use_flc)):
        text, n = re.subn(rf"(?m)^{name} = (True|False)", f"{name} = {value}", text)
        assert n == 1, name
    text, n = re.subn(r'(?m)^MODEL_TERMS = "[a-z_]+"', f'MODEL_TERMS = "{terms}"', text)
    assert n == 1
    fd, path = tempfile.mkstemp(suffix="_controller_node.py")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    return path


def spawn(cmd, log=None):
    sink = open(log, "w") if log else subprocess.DEVNULL
    return subprocess.Popen(cmd, start_new_session=True, stdout=sink, stderr=subprocess.STDOUT if log else subprocess.DEVNULL)


def stop(p):
    if p.poll() is None:
        try:
            os.killpg(p.pid, signal.SIGINT)
            p.wait(timeout=5)
        except Exception:
            os.killpg(p.pid, signal.SIGKILL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--no-flc", action="store_true")
    ap.add_argument("--terms", default="full", choices=["full", "no_coriolis", "decoupled", "gravity"])
    ap.add_argument("--vis-hz", type=int, default=20)
    ap.add_argument("--out", default="challenge3_results")
    ap.add_argument("--timeout", type=float, default=60.0)
    a = ap.parse_args()

    out = os.path.abspath(os.path.join(a.out, a.label))
    os.makedirs(out, exist_ok=True)
    procs, variant = [], None
    try:
        procs.append(spawn(["ros2", "run", "mujoco_ros2", "mujoco_node", os.path.abspath(SCENE), "--ros-args",
                            "-p", "control_mode:=TORQUE", "-p", "torque_passthrough:=true",
                            "-p", "simulation_frequency:=1000", "-p", f"visualisation_frequency:={a.vis_hz}",
                            "-p", "joint_state_topic_name:=joint_state", "-p", "joint_command_topic_name:=joint_commands",
                            "-p", "camera_focal_point:=[0.3,0.0,0.5]", "-p", "camera_distance:=2.2",
                            "-p", "camera_azimuth:=135.0", "-p", "camera_elevation:=-15.0"], os.path.join(out, "bridge.log")))
        time.sleep(3.0)
        logger = spawn(["ros2", "run", "panda_kinematics", "control_logger", "--ros-args", "-p", f"output_dir:={out}"],
                       os.path.join(out, "logger.log"))
        procs.append(logger)
        time.sleep(2.0)
        variant = make_variant(not a.no_model, not a.no_flc, a.terms)
        procs.append(spawn([sys.executable, variant], os.path.join(out, "node.log")))
        t0 = time.time()
        while logger.poll() is None and time.time() - t0 < a.timeout:
            time.sleep(0.5)
        print(("finished" if logger.poll() is not None else "TIMED OUT"), "->", out)
    finally:
        for p in reversed(procs):
            stop(p)
        if variant:
            os.remove(variant)


if __name__ == "__main__":
    main()
