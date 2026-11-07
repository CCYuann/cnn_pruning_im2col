#!/usr/bin/env python3
"""Run jobs from an appendable queue file, at most --per-gpu compute processes per GPU.

Queue line: <name> <TAB> <shell command>. A job whose <logs>/<name>.rc exists is not rerun.
Other people's and pre-existing GPU processes count toward the limit.
A job with <logs>/<name>.adopted is being watched by queue_handoff.py and is not started here.
"""
import argparse, os, signal, subprocess, time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--queue", type=Path, required=True)
parser.add_argument("--logs", type=Path, required=True)
parser.add_argument("--per-gpu", type=int, default=4)
parser.add_argument("--hours", type=float, default=14)
parser.add_argument("--cwd", type=Path, default=Path.cwd())
args = parser.parse_args()
args.logs.mkdir(parents=True, exist_ok=True)
deadline, active = time.time()+args.hours*3600, {}


def gpu_load():
    uuid = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    ids = {line.split(", ")[1].strip(): int(line.split(", ")[0]) for line in uuid.strip().splitlines()}
    load = {g: 0 for g in ids.values()}
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    for line in apps.strip().splitlines():
        load[ids[line.split(", ")[0].strip()]] += 1
    return load


def pending():
    rows = [l.split("\t", 1) for l in args.queue.read_text().splitlines() if l.strip() and not l.startswith("#")]
    return [(n, c) for n, c in rows if n not in active and not (args.logs/f"{n}.rc").exists() and not (args.logs/f"{n}.adopted").exists()]


while time.time() < deadline:
    for name, (process, log) in list(active.items()):
        code = process.poll()
        if code is not None:
            log.close()
            (args.logs/f"{name}.rc").write_text(str(code))
            print(time.strftime("%H:%M"), name, "rc", code, flush=True)
            del active[name]
    if (args.logs/"STOP").exists() and not active:
        break
    load = gpu_load()
    for gpu in load:  # launched jobs may not appear in nvidia-smi yet
        load[gpu] = max(load[gpu], sum(1 for _, (p, _l) in active.items() if getattr(p, "gpu", None) == gpu))
    for name, command in pending():
        gpu = min(load, key=load.get)
        if load[gpu] >= args.per_gpu or (args.logs/"STOP").exists():
            break
        log = open(args.logs/f"{name}.log", "w")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
        process = subprocess.Popen(command, shell=True, cwd=args.cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        process.gpu = gpu
        active[name] = (process, log)
        load[gpu] += 1
        print(time.strftime("%H:%M"), "start", name, "gpu", gpu, flush=True)
    time.sleep(30)
