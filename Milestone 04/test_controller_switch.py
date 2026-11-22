#!/usr/bin/env python3
"""
Milestone 04 - test of live controller switching and of the live validation readout.

Starts the simulator, planner and controller nodes (headless, fast) and checks that
  1. every controller can be selected while the system runs and reports its description,
  2. the same manual move is tracked best by computed torque and worst by the position servo,
  3. switching controllers in the middle of the sorting cycle does not break it,
  4. every finished trajectory segment is validated (task-space and joint-space).

Run (ROS sourced):   python3 "Milestone 04/test_controller_switch.py"
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
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray, String

from milestone_04 import topics as T

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
    n = Node("controller_switch_test")
    st = {"t": 0.0, "q": None, "info": None, "mon": [], "valid": [], "status": [], "boxes": None}
    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
    n.create_subscription(Float64MultiArray, T.STATE, lambda m: st.update(t=m.data[0], q=np.array(m.data[1:6])), 200)
    n.create_subscription(String, T.CONTROLLER_INFO, lambda m: st.update(info=json.loads(m.data)), latched)
    n.create_subscription(Float64MultiArray, T.MONITOR, lambda m: st["mon"].append(list(m.data)), 50)
    n.create_subscription(String, T.VALIDATION, lambda m: st["valid"].append(json.loads(m.data)), 100)
    n.create_subscription(String, T.STATUS, lambda m: st["status"].append(m.data), 50)
    n.create_subscription(Float64MultiArray, T.BOXES, lambda m: st.update(boxes=[bool(m.data[5 + 5 * i]) for i in range(3)]), 10)
    pc = n.create_publisher(String, T.COMMAND, 10)
    pk = n.create_publisher(String, T.CONTROLLER, 10)

    def spin(sec):
        t0 = time.time()
        while time.time() - t0 < sec:
            rclpy.spin_once(n, timeout_sec=0.01)

    def wait_sim(dt):
        t0 = st["t"]
        while st["t"] < t0 + dt:
            rclpy.spin_once(n, timeout_sec=0.05)

    def select(name):
        pk.publish(String(data=name))
        spin(0.4)

    def cmd(**k):
        pc.publish(String(data=json.dumps(k)))
        spin(0.2)

    def ok(cond, msg):
        results.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + msg)

    t0 = time.time()
    while st["q"] is None or st["info"] is None:
        rclpy.spin_once(n, timeout_sec=0.1)
        if time.time() - t0 > 60:
            print("FAIL system did not start")
            finish(1)

    # 1 + 2: each controller on the same manual move
    worst = {}
    for name in ("servo", "pd_gravity", "computed_torque"):
        select(name)
        ok(st["info"] and st["info"]["name"] == name and "law" in st["info"], f"selecting '{name}' -> description: {st['info']['label']}")
        cmd(cmd="goto_joint", q=[0, 0, 0, 0, 0]); wait_sim(3.0)           # same start for every controller
        st["mon"].clear()
        cmd(cmd="goto_joint", q=[0.6, -0.4, 0.4, -1.4, 0.3]); wait_sim(2.2)
        worst[name] = max(m[2] for m in st["mon"]) * 1000
    print("     max joint tracking error on the same fast move [mrad]:", {k: round(v, 1) for k, v in worst.items()})
    ok(worst["computed_torque"] < worst["pd_gravity"] < worst["servo"], "computed torque < PD + gravity < position servo (live monitor)")

    # 3 + 4: sorting with controller changes in the middle
    cmd(cmd="goto_joint", q=[0, 0, 0, 0, 0]); wait_sim(3.0)
    select("computed_torque"); st["valid"].clear(); st["status"].clear()
    cmd(cmd="start_sorting")
    while not any("plan ready" in x for x in st["status"]):         # planning runs in the background
        rclpy.spin_once(n, timeout_sec=0.05)
    t_start = st["t"]                                              # the cycle starts when the plan is ready
    schedule = [(8, "servo"), (14, "pd_gravity"), (20, "computed_torque"), (26, "pd_gravity"), (31, "computed_torque")]
    for when, name in schedule:
        while st["t"] - t_start < when:
            rclpy.spin_once(n, timeout_sec=0.05)
        select(name)
    while not any("finished" in x for x in st["status"]) and st["t"] - t_start < 120:
        rclpy.spin_once(n, timeout_sec=0.05)
    wait_sim(1.0)
    ok(st["boxes"] == [True, True, True], f"sorting completed with {len(schedule)} controller changes mid-cycle, boxes {st['boxes']}")
    spaces = {v["space"] for v in st["valid"]}
    ok({"task", "joint"} <= spaces and len(st["valid"]) >= 18,
       f"{len(st['valid'])} segments validated live ({sum(v['space'] == 'task' for v in st['valid'])} task-space, "
       f"{sum(v['space'] == 'joint' for v in st['valid'])} joint-space)")
    by_ctrl = {}
    for v in st["valid"]:
        by_ctrl.setdefault(v["controller"], []).append(v["ok"])
    print("     validation results per active controller:", {k: f"{sum(x)}/{len(x)} pass" for k, x in by_ctrl.items()})
    print("ALL TESTS PASSED" if all(results) else "SOME TESTS FAILED")
    rclpy.shutdown()
    finish(0 if all(results) else 1)
except KeyboardInterrupt:
    finish(1)
