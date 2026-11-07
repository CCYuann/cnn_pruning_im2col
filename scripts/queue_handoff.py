#!/usr/bin/env python3
"""Replace an expiring gpu_queue daemon without restarting its running jobs.

When the old PID exits, each job with no .rc whose exact command is still running is marked .adopted,
a new daemon starts, and each adopted job gets rc 0 only if it left its result.json or checkpoint.
"""
import os, re, subprocess, sys, time
from pathlib import Path

old_pid, queue, logs = int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])


def dataset_result(command):
    value = lambda key: re.search(rf"--{key} (\S+)", command).group(1)
    name = f"{value('dataset')}_{value('model')}_seed{value('seed')}_{value('arm')}"
    if value("arm") != "dense":
        name += f"_r{round(float(value('ratio'))*100)}"
    name += ("_floor" if "--q-floor" in command else "")+("_stem" if "--stem-keep" in command else "")
    return Path("runs/other_datasets")/name/"result.json"


def alive(pid):
    return Path(f"/proc/{pid}").exists()


while alive(old_pid):
    time.sleep(10)
rows = [line.split("\t", 1) for line in queue.read_text().splitlines() if line.strip()]
adopted = {}
for name, command in rows:
    if (logs/f"{name}.rc").exists():
        continue
    # The shell wrapper from Popen(shell=True) may exec or stay; match the python process itself.
    found = subprocess.run(["pgrep", "-fx", re.escape(command)], capture_output=True, text=True).stdout.split()
    if found:
        (logs/f"{name}.adopted").touch()
        adopted[name] = ([int(p) for p in found], command)
print(time.strftime("%H:%M"), "adopted", sorted(adopted), flush=True)
out = open(queue.parent/"queue2.out", "a")
subprocess.Popen([sys.executable, "scripts/gpu_queue.py", "--queue", str(queue), "--logs", str(logs), "--per-gpu", "4", "--hours", "14", "--cwd", "."],
                 stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
while adopted:
    for name, (pids, command) in list(adopted.items()):
        if any(alive(p) for p in pids):
            continue
        target = re.search(r"--dense-checkpoint-out (\S+)", command)
        if "dataset_pruning_worker" in command:
            done = dataset_result(command).exists()
        elif target:
            done = Path(target.group(1)).exists()
        else:
            root, seed, arm, tag = (re.search(rf"--{k} (\S+)", command) for k in ("root", "seed", "arm", "tag"))
            done = (Path(root.group(1))/f"seed{seed.group(1)}_{arm.group(1)}{tag.group(1) if tag else ''}/result.json").exists()
        (logs/f"{name}.rc").write_text("0" if done else "1")
        (logs/f"{name}.adopted").unlink()
        print(time.strftime("%H:%M"), name, "rc", 0 if done else 1, flush=True)
        del adopted[name]
    time.sleep(10)
