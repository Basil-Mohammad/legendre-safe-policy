"""ROS 2 (Jazzy, CycloneDDS) in-the-loop pipeline for the Unicycle environment.

     /obs (Float32MultiArray)          /cmd_raw (Twist)              /cmd_exec (Twist)
 plant ------------------------> policy ----------------> bridge ------------------> plant
 (MuJoCo, 20 Hz wall clock)    (SB3 actor + action map)  (fault injector + watchdog)   (latest command is held)

  plant   : steps the MuJoCo unicycle every PERIOD seconds with the latest /cmd_exec (zero-order hold); publishes the packed
            row observation (lsp2.rows layout, float32) after each step; publishes /episode (JSON) at episode end and /lat
            (obs-to-command latency measured on receipt of the first command that follows an observation).
  policy  : subscribes /obs, runs the trained policy (+ map for environment-side methods), publishes the command as
            geometry_msgs/Twist (linear.x = w0 V_MAX, angular.z = w1 W_MAX).
  bridge  : fault injector (command blackout bursts, as lsp2.faults 'drop') and watchdog (timeout -> zero Twist = fallback).
Run through scripts/ros_eval.py, which launches the nodes as separate processes.
"""
from __future__ import annotations

import json, sys, time, argparse
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, String

QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1)
V_MAX, W_MAX = 0.4, 2.0
P_ON, L_BURST, T_WD = 0.03, 6, 2


class Plant(Node):
    def __init__(self, a):
        super().__init__("plant")
        import torch
        from . import envs as E
        torch.set_default_dtype(torch.float32)
        self.env = E.Unicycle(a.cond, (10_000 + a.seed) * 100)
        self.env.enforce = not a.fault_run
        self.o, _ = self.env.reset()
        self.r0 = self.env.restorations
        self.cmd = np.zeros(2)
        self.t_obs = None
        self.n_ep, self.target = 0, a.episodes
        self.stats = dict(c=0, s=False, fb=0, n=0, viol=0.0)
        self.pub = self.create_publisher(Float32MultiArray, "/obs", QOS)
        self.pe = self.create_publisher(String, "/episode", QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE))
        self.pl = self.create_publisher(Float32MultiArray, "/lat", QOS)
        self.create_subscription(Twist, "/cmd_exec", self.on_cmd, QOS)
        self.t0 = time.perf_counter() + a.start_delay
        self.ticks, self.late = 0, 0
        self.period = a.period
        self.create_timer(a.period, self.tick)

    def on_cmd(self, m):
        self.cmd = np.array([m.linear.x / V_MAX, m.angular.z / W_MAX])
        if self.t_obs is not None:
            lat = time.perf_counter() - self.t_obs
            self.t_obs = None
            self.pl.publish(Float32MultiArray(data=[float(lat) * 1e3]))

    def publish_obs(self):
        self.t_obs = time.perf_counter()
        self.pub.publish(Float32MultiArray(data=self.o.tolist()))

    def tick(self):
        if time.perf_counter() < self.t0:
            if self.ticks == 0:
                self.publish_obs()
            return
        self.ticks += 1
        w = np.clip(self.cmd, -1, 1)
        rows = self.env._rows64
        if rows is not None and not self.env.enforce:
            A, b, ok, _ = rows
            self.stats["viol"] = max(self.stats["viol"], float(max(0.0, np.max(np.where(ok, A @ w - b, -np.inf)))))
        o, r, te, tr, info = self.env.step(w)
        st = self.stats
        st["c"] += int(info["collided"]); st["fb"] += int(info["fallback"]); st["n"] += 1; st["s"] |= bool(info["success"])
        if te or tr:
            self.pe.publish(String(data=json.dumps(dict(collided=st["c"] > 0, success=st["s"], fb_steps=st["fb"],
                                                        steps=st["n"], viol=st["viol"], restorations=self.env.restorations - self.r0))))
            self.n_ep += 1
            self.r0 = None
            self.stats = dict(c=0, s=False, fb=0, n=0, viol=0.0)
            o, _ = self.env.reset()
            self.r0 = self.env.restorations
        self.o = o
        self.publish_obs()


class Policy(Node):
    def __init__(self, a):
        super().__init__("policy")
        import torch
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)
        from .train import build, ENVS                                  # noqa: F401
        from .sb3_ext import MapLayer, PATHWISE
        from stable_baselines3 import PPO, SAC
        self.torch = torch
        self.model = (PPO if a.algo == "ppo" else SAC).load(a.model, device="cpu")
        self.layer = None
        if a.algo == "ppo" or a.method in ("se",):
            spec = ENVS["unicycle"]("A", 0).spec
            self.layer = MapLayer(spec, a.method, a.scale)
        self.pub = self.create_publisher(Twist, "/cmd_raw", QOS)
        self.create_subscription(Float32MultiArray, "/obs", self.on_obs, QOS)
        self.ms = []

    def on_obs(self, m):
        t0 = time.perf_counter()
        obs = np.asarray(m.data, dtype=np.float32)[None]
        act, _ = self.model.predict(obs, deterministic=True)
        if self.layer is not None:
            w = self.layer(self.torch.as_tensor(act, dtype=self.torch.float32), self.torch.as_tensor(obs)).numpy()[0]
        else:
            w = act[0]
        w = np.clip(w.astype(np.float64), -1, 1)
        msg = Twist(); msg.linear.x = float(w[0] * V_MAX); msg.angular.z = float(w[1] * W_MAX)
        self.pub.publish(msg)
        self.ms.append((time.perf_counter() - t0) * 1e3)


class Bridge(Node):
    def __init__(self, a):
        super().__init__("bridge")
        self.mode = a.bridge
        self.rng = np.random.default_rng(a.seed + 77)
        self.burst, self.held = 0, 0
        self.pub = self.create_publisher(Twist, "/cmd_exec", QOS)
        self.create_subscription(Twist, "/cmd_raw", self.on_cmd, QOS)
        self.last_rx = time.perf_counter()
        self.last_tx = Twist()
        self.timeout = a.watchdog_timeout
        self.create_timer(0.01, self.watchdog)
        self.dropped = 0

    def on_cmd(self, m):
        if self.mode in ("drop", "drop_wd"):
            if self.burst == 0 and self.rng.random() < P_ON:
                self.burst, self.held = L_BURST, 0
            if self.burst > 0:
                self.burst -= 1
                self.dropped += 1
                return                                         # message lost: nothing published
        self.last_rx = time.perf_counter()
        self.last_tx = m
        self.pub.publish(m)

    def watchdog(self):
        if self.mode == "drop_wd" and time.perf_counter() - self.last_rx > self.timeout:
            self.pub.publish(Twist())                          # certified fallback: stop
            self.last_rx = time.perf_counter() - self.timeout + 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("role", choices=["plant", "policy", "bridge"])
    ap.add_argument("--cond", default="A"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=30); ap.add_argument("--period", type=float, default=0.05)
    ap.add_argument("--start_delay", type=float, default=4.0); ap.add_argument("--fault_run", action="store_true")
    ap.add_argument("--algo", default="ppo"); ap.add_argument("--method", default="sp"); ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--model", default=""); ap.add_argument("--bridge", default="none")
    ap.add_argument("--watchdog_timeout", type=float, default=0.12)
    a = ap.parse_args()
    rclpy.init()
    node = dict(plant=Plant, policy=Policy, bridge=Bridge)[a.role](a)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    if a.role == "policy" and node.ms:
        print("POLICY_MS", json.dumps(dict(mean=float(np.mean(node.ms)), p99=float(np.percentile(node.ms, 99)))), flush=True)
    if a.role == "plant":
        print("PLANT_TICKS", node.ticks, flush=True)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
