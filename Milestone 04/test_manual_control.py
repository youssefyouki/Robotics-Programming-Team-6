#!/usr/bin/env python3
"""
Milestone 04 - test of the commands the control panel sends (joint angles, Cartesian targets, jog,
stop, pause / resume / reset, sorting) against the running closed-loop system.

It starts the simulator, planner and controller nodes itself (headless, as fast as possible), sends
the same JSON commands as the panel and checks the result in the simulator.

Run (ROS sourced):   python3 "Milestone 04/test_manual_control.py"
"""
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String

from milestone_04 import cell as C, tcp_kinematics as TK, topics as T

procs = [subprocess.Popen(c, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
         for c in (["ros2", "run", "milestone_04", "cell_sim_node", "--ros-args", "-p", "viewer:=false", "-p", "realtime:=false", "-p", "actuation:=torque"],
                   ["ros2", "run", "milestone_04", "sorting_node", "--ros-args", "-p", "auto_start:=false"],
                   ["ros2", "run", "milestone_04", "controller_node", "--ros-args", "-p", "controller:=computed_torque"])]
results = []


def finish(code):
    for p in procs:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    sys.exit(code)


try:
    rclpy.init()
    n = Node("manual_control_test")
    st = {"q": None, "t": 0.0, "epoch": 0, "g": 0.0, "status": [], "boxes": None}

    def on_state(m):
        st.update(q=np.array(m.data[1:6]), t=m.data[0], g=m.data[11], epoch=m.data[12])
    n.create_subscription(Float64MultiArray, T.STATE, on_state, 200)
    n.create_subscription(String, T.STATUS, lambda m: st["status"].append(m.data), 50)
    n.create_subscription(Float64MultiArray, T.BOXES, lambda m: st.update(boxes=[bool(m.data[5 + 5 * i]) for i in range(3)]), 10)
    pc = n.create_publisher(String, T.COMMAND, 10)
    ps = n.create_publisher(String, T.SIM_CONTROL, 10)

    def spin(sec):
        t0 = time.time()
        while time.time() - t0 < sec:
            rclpy.spin_once(n, timeout_sec=0.01)

    def wait_sim(dt):
        t0 = st["t"]
        while st["t"] < t0 + dt:
            rclpy.spin_once(n, timeout_sec=0.05)

    def cmd(**k):
        pc.publish(String(data=json.dumps(k)))
        spin(0.2)

    def sim(**k):
        ps.publish(String(data=json.dumps(k)))
        spin(0.2)

    def ok(cond, msg):
        results.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + msg)

    t_start = time.time()
    while st["q"] is None:
        rclpy.spin_once(n, timeout_sec=0.1)
        if time.time() - t_start > 60:
            print("FAIL simulator did not start")
            finish(1)

    # joint-space commands
    st["status"].clear(); cmd(cmd="goto_joint", q=[0.5, -0.3, 0.4, -1.0, 0.2]); spin(0.5)
    ok(any("below the work surface" in x for x in st["status"]), "joint target below the work surface is rejected")
    tgt = np.array([0.4, 0.0, -0.1, -1.6, 0.3]); cmd(cmd="goto_joint", q=tgt.tolist(), grip=-0.6); wait_sim(4.0)
    ok(np.abs(st["q"] - tgt).max() < 0.01 and abs(st["g"] + 0.6) < 0.05,
       f"goto_joint reaches the joint angles (error {np.abs(st['q'] - tgt).max():.4f} rad) and the gripper ({st['g']:.2f})")
    # task-space commands
    p = np.array([-0.103, 0.0, 0.085]); cmd(cmd="goto_tcp", p=p.tolist(), pitch=-np.pi / 2, roll=None, linear=False); wait_sim(4.0)
    err = np.linalg.norm(TK.tcp_position(st["q"]) - p) * 1000
    ok(err < 2.0 and abs(np.degrees(TK.tool_pitch(st["q"])) + 90) < 1, f"goto_tcp (joint motion): TCP error {err:.2f} mm, tool pointing down")
    p2 = p + np.array([0.0, 0.04, -0.02]); cmd(cmd="goto_tcp", p=p2.tolist(), pitch=-np.pi / 2, linear=True); wait_sim(3.5)
    err = np.linalg.norm(TK.tcp_position(st["q"]) - p2) * 1000
    ok(err < 2.0, f"goto_tcp (straight line): TCP error {err:.2f} mm")
    z0 = TK.tcp_position(st["q"])[2]; cmd(cmd="jog", d=[0, 0, 0.01]); wait_sim(1.5)
    dz = (TK.tcp_position(st["q"])[2] - z0) * 1000
    ok(abs(dz - 10) < 1.5, f"jog +10 mm in z moves the tool {dz:.2f} mm")
    st["status"].clear(); cmd(cmd="goto_tcp", p=[-0.3, 0, 0.3], pitch=-np.pi / 2); spin(0.5)
    ok(any("cannot do that" in x for x in st["status"]), "unreachable Cartesian target is rejected")
    # signal builder (constants, steps and waveforms as joint inputs)
    cmd(cmd="goto_joint", q=[0.0, -0.3, 0.3, -1.0, 0.0]); wait_sim(4.0)
    q_start = st["q"].copy()
    peak = {}
    for kind, amp, freq in (("sine", 0.25, 0.5), ("square", 0.2, 0.5), ("triangle", 0.2, 0.5), ("step", 0.2, 0.5), ("constant", 0.15, 0.5)):
        cmd(cmd="signal", joint=2, kind=kind, amp=amp, freq=freq, duration=4.0)
        hi_dev = 0.0
        t_end = st["t"] + 5.5
        while st["t"] < t_end:
            rclpy.spin_once(n, timeout_sec=0.02)
            hi_dev = max(hi_dev, abs(st["q"][1] - q_start[1]))
        peak[kind] = hi_dev
        ok(abs(hi_dev - amp) < 0.05 * amp + 0.01 and abs(st["q"][1] - q_start[1]) < 0.01,
           f"signal builder '{kind}' (amplitude {amp} rad): joint 2 peaked at {hi_dev:.3f} rad and returned to its start")
    st["status"].clear(); cmd(cmd="signal", joint=5, kind="sine", amp=6.0, freq=0.5, duration=3.0); spin(0.5)
    ok(any("joint limits" in x for x in st["status"]), "an amplitude beyond the joint limits is reduced with a warning")
    wait_sim(5.0)
    # simulation control
    sim(cmd="pause"); spin(0.5); t_a = st["t"]; spin(1.0)
    ok(abs(st["t"] - t_a) < 0.02, f"pause freezes the simulation clock ({t_a:.2f} s -> {st['t']:.2f} s)")
    cmd(cmd="goto_joint", q=[0, 0, 0, 0, 0]); spin(0.5)
    sim(cmd="resume"); wait_sim(1.0)
    ok(st["t"] > t_a + 0.9, "resume continues the simulation")
    e0 = st["epoch"]; sim(cmd="reset"); spin(1.0); wait_sim(1.0)
    ok(st["epoch"] == e0 + 1 and np.abs(st["q"]).max() < 0.05, "reset returns the arm to the home pose")
    # sorting program, stop in the middle, restart, repeat after reset
    st["status"].clear(); cmd(cmd="start_sorting"); wait_sim(8.0)
    cmd(cmd="hold"); wait_sim(1.5); q_a = st["q"].copy(); wait_sim(1.0)
    ok(np.abs(st["q"] - q_a).max() < 1e-3, "stop (hold) decelerates and holds the pose")
    cmd(cmd="start_sorting"); t0 = st["t"]
    while not any("finished" in x for x in st["status"]) and st["t"] - t0 < 90:
        rclpy.spin_once(n, timeout_sec=0.05)
    wait_sim(1.0)
    ok(st["boxes"] == [True, True, True], f"sorting resumed after a stop sorts all boxes {st['boxes']}")
    st["status"].clear(); cmd(cmd="start_sorting"); spin(0.5)
    ok(any("already" in x for x in st["status"]), "nothing is re-sorted when every box is already in its bin")
    sim(cmd="reset"); spin(1.0); wait_sim(0.5); st["status"].clear(); cmd(cmd="start_sorting"); t0 = st["t"]
    while not any("finished" in x for x in st["status"]) and st["t"] - t0 < 90:
        rclpy.spin_once(n, timeout_sec=0.05)
    wait_sim(1.0)
    ok(st["boxes"] == [True, True, True], f"reset + full cycle sorts all boxes {st['boxes']}")
    print("ALL TESTS PASSED" if all(results) else "SOME TESTS FAILED")
    rclpy.shutdown()
    finish(0 if all(results) else 1)
except KeyboardInterrupt:
    finish(1)
