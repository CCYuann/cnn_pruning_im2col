#!/usr/bin/env python3
"""Queue every arm of the frozen plan on two GPUs, then pick lambda and finish group."""
import json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
SLOTS = [0, 1]*4
LAMBDAS = ["3e-5", "1e-4", "3e-4"]
ARMS = ["dense", "unstructured", "ft_const", "rewind", "rewind_r90", "rewind_B", "rewind_kd", "gradual"]


def command(arm, seed, lam=None, tag=""):
    args = [sys.executable, "scripts/retrain_recovery_worker.py", "--arm", arm, "--seed", str(seed), "--root", str(ROOT)]
    return args+(["--group-lambda", lam, "--tag", tag] if lam else [])


def run_all(jobs):
    active, failed = {}, []
    while jobs or active:
        for slot, gpu in enumerate(SLOTS):
            if slot not in active and jobs:
                name, args = jobs.pop(0)
                log = open(ROOT/"logs"/f"{name}.log", "w")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
                active[slot] = (name, subprocess.Popen(args, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT), log)
        for slot, (name, process, log) in list(active.items()):
            if process.poll() is not None:
                log.close()
                print(name, "rc", process.returncode, flush=True)
                if process.returncode:
                    failed.append(name)
                del active[slot]
        time.sleep(10)
    return failed


pilots = [(f"group_pilot_{l}", command("group", 0, l, f"_lam{l}")) for l in LAMBDAS]
rest = [(f"{a}_{s}", command(a, s)) for s in range(3) for a in ARMS]
failed = run_all(pilots+rest)
if any(n.startswith("group_pilot") for n in failed):
    sys.exit(f"pilot failed: {failed}")
scores = {l: json.loads((ROOT/f"seed0_group_lam{l}/result.json").read_text())["final_validation"] for l in LAMBDAS}
best = max(LAMBDAS, key=lambda l: scores[l])
(ROOT/"LAMBDA_SELECTION.json").write_text(json.dumps(dict(scores=scores, selected=best), indent=2))
failed += run_all([(f"group_{s}", command("group", s, best, f"_lam{best}")) for s in (1, 2)])
print("failed:", failed)
sys.exit(1 if failed else 0)
