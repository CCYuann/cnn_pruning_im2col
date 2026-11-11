#!/usr/bin/env python3
"""CIFAR-100 accuracy against V3 predicted cycles, all models, 90-99%, seeds 0/1/2 (plans M-Q). Writes CURVE_CIFAR100.csv."""
import csv, json, subprocess, sys
from pathlib import Path

R = Path("runs/other_datasets")
C = Path(__file__).resolve().parents[2]/"mask_aware_conv_accelerator/scripts/estimate_network_cycles.py"
TMP = R/"curve_tmp.csv"
ARMS = {"rewind": "rewind_r{r}", "ft_const": "ft_const_r{r}", "floor": "gradual_r{r}_floor", "fill15": "gradual_r{r}_floor_n15_free",
        "floor_kd": "gradual_r{r}_floor_kd", "fill15_kd": "gradual_r{r}_floor_n15_free_kd", "unstructured": "unstructured_r{r}",
        "gradual_unstructured": "gradual_unstructured_r{r}", "rewind_floor": "rewind_r{r}_floor"}


def ms(model, path):
    subprocess.run([sys.executable, str(C), "--geometry", str(R/f"geometry_cifar100_{model}.json"), "--masks", f"a={path}", "--out", str(TMP)], check=True, stdout=subprocess.DEVNULL)
    rows = {r["label"]: float(r["ms_50mhz"]) for r in csv.DictReader(TMP.open())}
    return rows["a"], rows["dense"]




def build():
    rows = []
    for model in ("resnet34", "resnet56", "vgg16_bn", "googlenet", "densenet121"):
        dense = [json.loads((R/f"cifar100_{model}_seed{s}_dense/result.json").read_text())["final_test"]*100 for s in (0, 1, 2)]
        for ratio in (90, 92, 95, 98, 99):
            for arm, pattern in ARMS.items():
                runs = [R/f"cifar100_{model}_seed{s}_{pattern.format(r=ratio)}" for s in (0, 1, 2)]
                if not all((d/"result.json").exists() for d in runs):
                    continue
                res = [json.loads((d/"result.json").read_text()) for d in runs]
                masks = [d/("union_masks.pt" if "unstructured" in arm else "masks.pt") for d in runs]
                cyc = [ms(model, m) for m in masks if m.exists()]
                acc = [r["final_test"]*100 for r in res]
                rows.append(dict(model=model, ratio=ratio, arm=arm, test=round(sum(acc)/3, 3), seeds="/".join(f"{a:.2f}" for a in acc),
                                 dense=round(sum(dense)/3, 3), sparsity=round(sum(r.get("sparsity", 0) for r in res)/3, 4),
                                 ms=round(sum(c for c, _ in cyc)/len(cyc), 3) if cyc else "", dense_ms=round(cyc[0][1], 3) if cyc else "",
                                 ms_note="" if len(cyc) == 3 else f"cycles from {len(cyc)} seeds"))
                print(rows[-1], flush=True)
    TMP.unlink(missing_ok=True)
    with open(R/"CURVE_CIFAR100.csv", "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)



def fill_gaps():
    """Fill cycles the masks above cannot give: one-shot unstructured from PLAN_O3_UNSTRUCTURED_CYCLES.csv (exact, the same
    global-magnitude mask from the Dense checkpoint), gradual_unstructured 98/99% from the seed 0 rerun that kept union masks."""
    path = R/"CURVE_CIFAR100.csv"
    rows = list(csv.DictReader(path.open()))
    oneshot = {}
    for r in csv.DictReader((R/"PLAN_O3_UNSTRUCTURED_CYCLES.csv").open()):
        oneshot.setdefault((r["model"], r["ratio"]), []).append(float(r["ms"]))
    for row in rows:
        if row["ms"]:
            continue
        key = (row["model"], row["ratio"])
        if row["arm"] == "unstructured" and key in oneshot:
            v = oneshot[key]
            row["ms"], row["ms_note"] = round(sum(v)/len(v), 3), f"one-shot global magnitude masks, seeds 0-{len(v)-1}"
        elif row["arm"] == "gradual_unstructured":
            mask = R/f"cifar100_{row['model']}_seed0_gradual_unstructured_r{row['ratio']}_union/union_masks.pt"
            if mask.exists():
                row["ms"], _ = ms(row["model"], mask)
                row["ms"], row["ms_note"] = round(row["ms"], 3), "seed 0 rerun (--tag _union) only"
        if row["ms"] and not row["dense_ms"]:
            row["dense_ms"] = next(r["dense_ms"] for r in rows if r["model"] == row["model"] and r["dense_ms"])
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)
    TMP.unlink(missing_ok=True)


if __name__ == "__main__":
    if "--fill-only" not in sys.argv:
        build()
    fill_gaps()
