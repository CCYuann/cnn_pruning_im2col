#!/usr/bin/env python3
"""Run seed0 lambda candidates, select per method, then finish seeds 1 and 2."""
import json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
PRIOR = PROJECT/"runs/retrain_recovery_r0p95"
SLOTS = [0, 1, 0, 1]
CANDIDATES = {"group": ["1e-5", "2e-5"], "group_gradual": ["1e-5", "3e-5"]}


def job(arm, seed, lam):
    tag = f"_lam{lam}"
    args = [sys.executable, "scripts/retrain_recovery_worker.py", "--arm", arm, "--seed", str(seed), "--root", str(ROOT),
            "--group-lambda", lam, "--tag", tag]
    return f"{arm}{tag}_{seed}", args


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


def score(arm, lam):
    root = PRIOR if (arm, lam) == ("group", "3e-5") else ROOT
    return json.loads((root/f"seed0_{arm}_lam{lam}/result.json").read_text())["final_validation"]


failed = run_all([job(a, 0, l) for a, values in CANDIDATES.items() for l in values])
if failed:
    sys.exit(f"seed0 failed: {failed}")
choice = {}
for arm, values in CANDIDATES.items():
    pool = values+(["3e-5"] if arm == "group" else [])
    choice[arm] = dict(scores={l: score(arm, l) for l in pool})
    choice[arm]["selected"] = max(pool, key=lambda l: choice[arm]["scores"][l])
(ROOT/"LAMBDA_SELECTION.json").write_text(json.dumps(choice, indent=2))
print(json.dumps(choice), flush=True)
jobs = [job(a, s, c["selected"]) for a, c in choice.items() for s in (1, 2)
        if not (a == "group" and c["selected"] == "3e-5")]
failed = run_all(jobs)
print("failed:", failed)
sys.exit(1 if failed else 0)
