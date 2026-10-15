"""Kaggle shard runner (CPU notebook, Internet ON).  Usage in a notebook cell:

    !python /kaggle/input/<dataset>/kaggle/run_shard.py --shard K1
Seven shards K1..K7 (see SHARDS); use "Save Version -> Save & Run All (Commit)" so that /kaggle/working is kept.
navsi, fetch and unicycle/ppo were run in the sandbox and are not part of the shards.
"""
import argparse, glob, os, subprocess, sys, shutil

SHARDS = {
    # seven shards of roughly 3-6 hours each on 4 CPU cores (estimates; Kaggle allows two notebooks at a time, 12 h per run)
    "K1": [("navdi", "sac", "sp,gauge,se,pen2")],
    "K2": [("navdi", "sac", "legendre,legendre_st"), ("navdi", "ppo", "")],
    "K3": [("unicycle", "sac", "sp,gauge,se,pen2")],
    "K4": [("unicycle", "sac", "legendre,legendre_st")],
    "K5": [("panda", "sac", "sp,gauge")],
    "K6": [("panda", "sac", "se,pen2"), ("panda", "ppo", "")],
    "K7": [("panda", "sac", "legendre,legendre_st")],
}
ap = argparse.ArgumentParser(); ap.add_argument("--shard", required=True); ap.add_argument("--workers", type=int, default=4)
a = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__))
src = os.path.dirname(here)
work = "/kaggle/working/lsp"
if not os.path.exists(work):
    shutil.copytree(src, work, ignore=shutil.ignore_patterns("__pycache__", "results*"))
os.chdir(work)
sh = lambda c: subprocess.run(c, shell=True, check=True)
sh(f"{sys.executable} -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu")
sh(f"{sys.executable} -m pip install -q stable-baselines3==2.9.0 gymnasium==1.4.0 gymnasium-robotics==1.4.2 mujoco==3.3.7 scipy pandas")
need_panda = any(e == "panda" for e, _, _ in SHARDS[a.shard])
if need_panda and not os.path.exists("/kaggle/working/menagerie"):
    sh("git clone -q https://github.com/google-deepmind/mujoco_menagerie /kaggle/working/menagerie && "
       "cd /kaggle/working/menagerie && git checkout -q 0059d4335f8156206f63a35662313385f7ad6d74")
env = dict(os.environ, PYTHONPATH=work, MENAGERIE_PATH="/kaggle/working/menagerie", OMP_NUM_THREADS="1", MUJOCO_GL="egl")
for e, algos, methods in SHARDS[a.shard]:
    cmd = [sys.executable, "scripts/study6_driver.py", "--env", e, "--algos", algos, "--workers", str(a.workers),
           "--out", "/kaggle/working/results_v6"] + (["--methods", methods] if methods else [])
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, env=env, check=False)
shutil.make_archive(f"/kaggle/working/study6_{a.shard}_results", "zip", "/kaggle/working/results_v6")
print("finished", a.shard)
