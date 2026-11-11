#!/usr/bin/env python3
"""Plan N: N1 gradual+floor vs rewind (VGG16-BN, ResNet-56), N2 structure cost vs gradual_unstructured, N3 KD at 98%. CIFAR-100, seeds 0/1/2, FP32 test."""
import csv, json, subprocess, sys
from pathlib import Path

ROOT = Path("runs/other_datasets")
CYCLES = Path(__file__).resolve().parents[2]/"mask_aware_conv_accelerator/scripts/estimate_network_cycles.py"
SEEDS = (0, 1, 2)


def run(model, seed, suffix):
    path = ROOT/f"cifar100_{model}_seed{seed}_{suffix}"/"result.json"
    return json.loads(path.read_text()) if path.exists() else None


def accs(model, suffix):
    rs = [run(model, s, suffix) for s in SEEDS]
    return None if None in rs else [r["final_test"]*100 for r in rs]


def ms(model, suffix):
    masks = [f"{suffix}_{s}={ROOT}/cifar100_{model}_seed{s}_{suffix}/masks.pt" for s in SEEDS]
    out = Path("/tmp/claude-1006/-home-yenmh3-11457019/340af228-100b-4826-aa9e-e2c9efbbd0e3/scratchpad/cyc_n.csv")
    subprocess.run([sys.executable, str(CYCLES), "--geometry", str(ROOT/f"geometry_cifar100_{model}.json"), "--masks", *masks, "--out", str(out)], check=True, stdout=subprocess.DEVNULL)
    rows = list(csv.DictReader(out.open()))
    vals = [float(r["ms_50mhz"]) for r in rows if r["label"] != "dense"]
    return sum(vals)/len(vals), next(float(r["ms_50mhz"]) for r in rows if r["label"] == "dense")


def diff(a, b):
    d = [x-y for x, y in zip(a, b)]
    return d, sum(d)/len(d)


fmt = lambda xs: "/".join(f"{x:+.2f}" for x in xs)
m = lambda xs: sum(xs)/len(xs)
out = {}
for model in ("resnet34", "googlenet", "densenet121", "vgg16_bn", "resnet56"):
    dense = accs(model, "dense")
    print(f"\n== {model}: Dense {m(dense):.2f}" if dense else f"\n== {model}")
    for ratio in (90, 98, 99):
        rw, gf = accs(model, f"rewind_r{ratio}"), accs(model, f"gradual_r{ratio}_floor")
        un = accs(model, f"unstructured_r{ratio}") if ratio > 90 else None
        gu = accs(model, f"gradual_unstructured_r{ratio}") if ratio > 90 else None
        line = f" {ratio}%"
        if rw: line += f" rewind {m(rw):.2f}"
        if gf: line += f" floor {m(gf):.2f}"
        if un: line += f" unstr {m(un):.2f}"
        if gu: line += f" grad_unstr {m(gu):.2f}"
        rec = dict(rewind=rw, floor=gf, unstructured=un, gradual_unstructured=gu)
        if rw and gf:
            d, mu = diff(gf, rw)
            ok = mu >= 0.3 and all(x > 0 for x in d)
            line += f" | N1 floor-rewind {fmt(d)} ({mu:+.2f}) {'PASS' if ok else 'fail'}"
            rec["n1"] = dict(d=d, mean=mu, pass_=ok)
        if gu and gf:
            d, mu = diff(gf, gu)
            line += f" | N2 floor-grad_unstr {fmt(d)} ({mu:+.2f})"
            rec["n2"] = dict(d=d, mean=mu)
        if ratio == 98:
            kd = accs(model, "gradual_r98_floor_kd")
            if kd and gf:
                d, mu = diff(kd, gf)
                ok = mu >= 0.3 and all(x > 0 for x in d)
                line += f" | N3 kd {m(kd):.2f} kd-floor {fmt(d)} ({mu:+.2f}) {'PASS' if ok else 'fail'}"
                rec["n3"] = dict(kd=kd, d=d, mean=mu, pass_=ok)
        if rw and gf:
            rec["ms_rewind"], dms = ms(model, f"rewind_r{ratio}")
            rec["ms_floor"], _ = ms(model, f"gradual_r{ratio}_floor")
            rec["ms_dense"] = dms
            line += f" | ms rewind {rec['ms_rewind']:.2f} floor {rec['ms_floor']:.2f} (dense {dms:.2f})"
        print(line)
        out[f"{model}_{ratio}"] = rec
    out[f"{model}_dense"] = dense
json.dump(out, open(ROOT/"PLAN_N_RESULTS.json", "w"), indent=1)
