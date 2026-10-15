# Running Study 6 shards on Kaggle (CPU, free tier)
Seven shards K1..K7 (3-6 h each on 4 cores), two notebooks at a time:
round 1: K1 + K3 | round 2: K2 + K4 | round 3: K5 + K7 | round 4: K6.
1. Datasets -> New Dataset -> upload `legendre-safe-policy-code-v6.zip` (Kaggle extracts it) -> title `lsp-code-v6` -> Create.
2. New Notebook -> Add Input -> your dataset. Settings: Accelerator **None**, Internet **On**.
3. Smoke test (2 min): `!python /kaggle/input/lsp-code-v6/kaggle/run_shard.py --shard K1` ; when you see training lines/no import error, stop it.
   If the path differs: `!find /kaggle/input -name run_shard.py`.
4. Save Version -> Save & Run All (Commit). It runs in the background (max 12 h). Start the second notebook of the round the same way.
5. When a version finishes: open it -> Output -> download `study6_<shard>_results.zip`.
Each shard is resume-safe within one session.
