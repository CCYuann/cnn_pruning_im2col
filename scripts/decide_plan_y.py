#!/usr/bin/env python3
"""Plan Y (hardware S5): cluster vs unstructured vs 2:4 on CIFAR-100, descriptive only (no pass/fail).

Prints per model and ratio: Dense, mean test of each arm, paired gaps (cluster - other), V3 predicted ms and kept
weights. Unstructured and 2:4 are costed on the union of each 8-filter cluster's positions, which is what V3 runs.
Writes runs/other_datasets/PLAN_Y_RESULTS.json. Missing runs print as pending.
"""
import csv, json, subprocess, sys
from pathlib import Path

R = Path("runs/other_datasets")
C = Path(__file__).resolve().parents[2]/"mask_aware_conv_accelerator/scripts/estimate_network_cycles.py"
TMP = R/"plan_y_tmp.csv"
ARMS = {50: {"cluster": ("gradual_r50_floor", "masks.pt"), "unstructured": ("gradual_unstructured_r50", "union_masks.pt"), "2:4": ("gradual_nm24_r50", "union_masks.pt")},
        90: {"cluster": ("gradual_r90_floor", "masks.pt"), "unstructured": ("gradual_unstructured_r90", "union_masks.pt")}}


def ms(model, path):
    subprocess.run([sys.executable, str(C), "--geometry", str(R/f"geometry_cifar100_{model}.json"), "--masks", f"a={path}", "--out", str(TMP)],
                   check=True, stdout=subprocess.DEVNULL)
    rows = {r["label"]: float(r["ms_50mhz"]) for r in csv.DictReader(TMP.open())}
    return rows["a"], rows["dense"]


def main():
    out = {}
    for model in ("resnet34", "resnet56"):
        dense = [json.loads((R/f"cifar100_{model}_seed{s}_dense/result.json").read_text())["final_test"]*100 for s in range(3)]
        for ratio, arms in ARMS.items():
            row = dict(dense=sum(dense)/3)
            tests = {}
            for arm, (suffix, mask_file) in arms.items():
                runs = [R/f"cifar100_{model}_seed{s}_{suffix}" for s in range(3)]
                if not all((p/"result.json").exists() for p in runs):
                    row[arm] = "pending"
                    continue
                results = [json.loads((p/"result.json").read_text()) for p in runs]
                tests[arm] = [r["final_test"]*100 for r in results]
                cycles = [ms(model, p/mask_file) for p in runs]
                row[arm] = dict(test=sum(tests[arm])/3, tests=tests[arm], sparsity=sum(r["sparsity"] for r in results)/3,
                                kept=sum(r["kept"] for r in results)/3, ms=sum(c[0] for c in cycles)/3, dense_ms=cycles[0][1])
            for other in ("unstructured", "2:4"):
                if "cluster" in tests and other in tests:
                    d = [a-b for a, b in zip(tests["cluster"], tests[other])]
                    row[f"cluster-{other}"] = dict(diffs=d, mean=sum(d)/3)
            out[f"{model}_r{ratio}"] = row
            print(f"\n{model} {ratio}%  Dense {row['dense']:.2f}  (CIFAR-100, seeds 0/1/2, n=3)")
            for arm in arms:
                v = row[arm]
                if v == "pending":
                    print(f"  {arm:13s} pending")
                else:
                    print(f"  {arm:13s} test {v['test']:.2f}  sparsity {v['sparsity']*100:.2f}%  kept {v['kept']/1e6:.3f}M  V3 {v['ms']:.2f} ms (dense {v['dense_ms']:.2f})")
            for other in ("unstructured", "2:4"):
                g = row.get(f"cluster-{other}")
                if g:
                    print(f"  cluster - {other:12s} {'/'.join(f'{x:+.2f}' for x in g['diffs'])} (mean {g['mean']:+.2f}pp)")
    TMP.unlink(missing_ok=True)
    (R/"PLAN_Y_RESULTS.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
