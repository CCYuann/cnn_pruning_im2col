#!/usr/bin/env python3
"""Plan O3: V3 cycles of one-shot global-magnitude unstructured masks (cluster q = union of the 8 filters' positions)."""
import csv, subprocess, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import retrain_recovery_worker as rw

core = rw.core
ROOT = Path("runs/other_datasets")
CYCLES = Path(__file__).resolve().parents[2]/"mask_aware_conv_accelerator/scripts/estimate_network_cycles.py"
TMP = Path(sys.argv[1])
rows = []
for model in ("resnet34", "googlenet", "densenet121", "vgg16_bn", "resnet56"):
    for seed in (0, 1, 2):
        net = core.make_model(model, 100, 32)
        net.load_state_dict(torch.load(ROOT/f"checkpoints/cifar100_{model}_seed{seed}.pt", map_location="cpu", weights_only=False))
        specs = core.collect_conv_specs(net, 32)
        for ratio in (.98, .99):
            full = rw.unstructured_masks(specs, ratio, "cpu")
            masks = {s.name: rw.cluster_view(full[s.name].float()).any(1) for s in specs}
            path = TMP/f"{model}_{seed}_{round(ratio*100)}.pt"
            torch.save(masks, path)
            out = TMP/"c.csv"
            subprocess.run([sys.executable, str(CYCLES), "--geometry", str(ROOT/f"geometry_cifar100_{model}.json"), "--masks", f"u={path}", "--out", str(out)], check=True, stdout=subprocess.DEVNULL)
            got = {r["label"]: float(r["ms_50mhz"]) for r in csv.DictReader(out.open())}
            rows.append(dict(model=model, seed=seed, ratio=round(ratio*100), ms=got["u"], dense_ms=got["dense"]))
            print(rows[-1], flush=True)
with open(ROOT/"PLAN_O3_UNSTRUCTURED_CYCLES.csv", "w") as f:
    w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)
