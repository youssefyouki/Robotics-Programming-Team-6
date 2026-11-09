#!/usr/bin/env python3
"""
Automated Challenge 2 run: real C++ bridge + node + tracking logger, one configuration at a time.

The instructor's template switches mode by commenting / uncommenting two lines in tick(). To run
both modes without editing the submitted file, this script makes a TEMPORARY copy of
trajectory_node.py with exactly those lines swapped (and optionally another TRAJ_DT), runs it,
and leaves the real file untouched.

  source /opt/ros/jazzy/setup.bash && source install/setup.bash
  python3 src/panda_kinematics/tools/run_experiment.py template POSITION --label template_1khz
  python3 src/panda_kinematics/tools/run_experiment.py template VELOCITY --label template_20hz --traj-dt 0.05
  python3 src/panda_kinematics/tools/run_experiment.py experiment VELOCITY --label runner --vis-hz 5

  runner   'template'   the instructor's trajectory_node.py exactly as submitted (arm starts at all zeros)
           'experiment' trajectory_experiment: slow set-up move to Q_START + rests + main move
Results: <out>/<label>/<MODE>/{metrics.txt, tracking.png, tracking.csv}
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
NODE_FILE = os.path.join(HERE, "..", "panda_kinematics", "trajectory_node.py")
SCENE = os.path.join(HERE, "..", "..", "mujoco_ros2", "model", "franka_emika_panda", "scene.xml")

# the two places in tick() the template asks you to switch (exact text of the template)
SWITCH = [
    ("            cmd.data = Q_END.tolist() + [GRIPPER_CTRL, 0.0]",
     "            # cmd.data = Q_END.tolist() + [GRIPPER_CTRL, 0.0]"),
    ("            # cmd.data = [0.0] * 7 + [GRIPPER_CTRL, 0.0]",
     "            cmd.data = [0.0] * 7 + [GRIPPER_CTRL, 0.0]"),
    ("        q_target = self.q_traj[self.step_idx]",
     "        # q_target = self.q_traj[self.step_idx]"),
    ("        # q_target = self.dq_traj[self.step_idx]",
     "        q_target = self.dq_traj[self.step_idx]"),
]


def make_variant(mode, traj_dt):
    text = open(NODE_FILE).read()
    if mode == "VELOCITY":
        for old, new in SWITCH:
            assert text.count(old) == 1, f"template line not found exactly once: {old.strip()}"
            text = text.replace(old, new)
    if traj_dt is not None:
        text, n = re.subn(r"(?m)^TRAJ_DT\s*=\s*[0-9.eE+-]+", f"TRAJ_DT       = {traj_dt}", text)
        assert n == 1
    fd, path = tempfile.mkstemp(suffix="_trajectory_node.py")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    return path


def spawn(cmd, log=None):
    """Start a process in its own process group; its output goes to `log` (a file path) if given."""
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
    ap.add_argument("runner", choices=["template", "experiment"])
    ap.add_argument("mode", choices=["POSITION", "VELOCITY"])
    ap.add_argument("--label", required=True)
    ap.add_argument("--traj-dt", type=float, default=None, help="override TRAJ_DT (template runner only)")
    ap.add_argument("--vis-hz", type=int, default=20, help="bridge render rate")
    ap.add_argument("--out", default="challenge2_results")
    ap.add_argument("--timeout", type=float, default=90.0)
    a = ap.parse_args()

    out = os.path.abspath(os.path.join(a.out, a.label))
    os.makedirs(os.path.join(out, a.mode), exist_ok=True)
    logfile = lambda name: os.path.join(out, a.mode, name + ".log")
    bridge = spawn(["ros2", "run", "mujoco_ros2", "mujoco_node", os.path.abspath(SCENE), "--ros-args",
                    "-p", f"control_mode:={a.mode}", "-p", "simulation_frequency:=1000",
                    "-p", f"visualisation_frequency:={a.vis_hz}",
                    "-p", "joint_state_topic_name:=joint_state", "-p", "joint_command_topic_name:=joint_commands",
                    "-p", "camera_focal_point:=[0.3,0.0,0.5]", "-p", "camera_distance:=2.2",
                    "-p", "camera_azimuth:=135.0", "-p", "camera_elevation:=-15.0"])
    procs = [bridge]
    variant = None
    try:
        time.sleep(3.0)                                  # let the bridge come up
        dt_for_logger = a.traj_dt
        logger_cmd = ["ros2", "run", "panda_kinematics", "tracking_logger", "--ros-args",
                      "-p", f"control_mode:={a.mode}", "-p", f"template:={'true' if a.runner == 'template' else 'false'}",
                      "-p", f"output_dir:={out}"]
        if a.runner == "template" and dt_for_logger is not None:
            logger_cmd += ["-p", f"traj_dt:={dt_for_logger}"]
        logger = spawn(logger_cmd, logfile("logger"))
        procs.append(logger)
        time.sleep(2.0)                                  # logger must be listening before the first command
        if a.runner == "template":
            variant = make_variant(a.mode, a.traj_dt)
            node = spawn([sys.executable, variant], logfile("node"))
        else:
            node = spawn(["ros2", "run", "panda_kinematics", "trajectory_experiment",
                          "--ros-args", "-p", f"control_mode:={a.mode}"])
        procs.append(node)
        t0 = time.time()
        while logger.poll() is None and time.time() - t0 < a.timeout:
            time.sleep(0.5)
        ok = logger.poll() is not None
        print(("finished" if ok else "TIMED OUT"), "->", os.path.join(out, a.mode))
    finally:
        for p in reversed(procs):
            stop(p)
        if variant:
            os.remove(variant)


if __name__ == "__main__":
    main()
