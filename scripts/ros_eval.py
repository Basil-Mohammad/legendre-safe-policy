"""Launch the ROS 2 pipeline (separate processes), collect N episodes, write a JSON record.
  source ~/ros2_ws/install/setup.bash; export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_LOCALHOST_ONLY=1
  python scripts/ros_eval.py --model M.zip --algo ppo --method sp --scale 1 --seed 0 --bridge none --episodes 30 --out r.json
(run with the ROS Python 3.12 venv)"""
import argparse, json, os, signal, subprocess, sys, time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32MultiArray, String

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--algo", required=True); ap.add_argument("--method", required=True)
ap.add_argument("--scale", type=float, default=1.0); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--bridge", default="none"); ap.add_argument("--episodes", type=int, default=30)
ap.add_argument("--period", type=float, default=0.05); ap.add_argument("--cond", default="A"); ap.add_argument("--out", required=True)
ap.add_argument("--timeout", type=float, default=900)
a = ap.parse_args()
py = sys.executable
common = ["--seed", str(a.seed), "--cond", a.cond]
procs = []
logs = {}
def launch(role, extra):
    f = open(f"{a.out}.{role}.log", "w")
    p = subprocess.Popen([py, "-m", "lsp2.ros_nodes", role] + common + extra, stdout=f, stderr=subprocess.STDOUT)
    procs.append(p); logs[role] = f
launch("bridge", ["--bridge", a.bridge])
launch("policy", ["--algo", a.algo, "--method", a.method, "--scale", str(a.scale), "--model", a.model])
time.sleep(5.0)
launch("plant", ["--episodes", str(a.episodes), "--period", str(a.period), "--start_delay", "3"] + (["--fault_run"] if a.bridge != "none" else []))
rclpy.init()
n = Node("collector")
eps, lats = [], []
n.create_subscription(String, "/episode", lambda m: eps.append(json.loads(m.data)), QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE))
n.create_subscription(Float32MultiArray, "/lat", lambda m: lats.append(m.data[0]), QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))
t0 = time.time()
while len(eps) < a.episodes and time.time() - t0 < a.timeout:
    rclpy.spin_once(n, timeout_sec=0.1)
for p in procs:
    p.send_signal(signal.SIGINT)
time.sleep(2)
for p in procs:
    try: p.wait(timeout=10)
    except Exception: p.kill()
for f in logs.values(): f.close()
pol = [l for l in open(f"{a.out}.policy.log") if l.startswith("POLICY_MS")]
res = dict(method=a.method, algo=a.algo, scale=a.scale, seed=a.seed, bridge=a.bridge, period=a.period, episodes=len(eps),
           success=float(np.mean([e["success"] for e in eps])) if eps else None,
           collided=float(np.mean([e["collided"] for e in eps])) if eps else None,
           viol=float(np.mean([e["viol"] for e in eps])) if eps else None,
           fb_steps=float(np.mean([e["fb_steps"] for e in eps])) if eps else None,
           restorations=float(np.mean([e["restorations"] for e in eps])) if eps else None,
           lat_ms=dict(n=len(lats), median=float(np.median(lats)), p95=float(np.percentile(lats, 95)), p99=float(np.percentile(lats, 99)),
                       max=float(np.max(lats))) if lats else None,
           policy_ms=json.loads(pol[0].split(" ", 1)[1]) if pol else None,
           load1=os.getloadavg()[0], wall_s=time.time() - t0)
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps(res))
rclpy.shutdown()
