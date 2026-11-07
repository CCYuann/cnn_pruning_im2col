#!/usr/bin/env python3
"""Derived-output audits only: numerical zeros, C-vs-B paired gains, provenance."""

import json
import shutil
import statistics
from pathlib import Path

import torch

import run_compensation_controls as c


def main():
    torch.set_num_threads(4)
    c.freeze_sources()
    if json.loads((c.ROOT/"status.json").read_text())["state"]!="complete":
        raise RuntimeError("Complete training and the reporting finalizer first")
    rows=[]
    layer_rows=[]
    for seed in c.SEEDS:
        for variant in c.VARIANTS:
            masks=c.worker.branch_masks(seed,variant)
            for budget in [10,30]:
                if budget==30 and variant not in c.PRIMARY:
                    continue
                if budget==10 and variant in c.worker.REUSED:
                    path=c.OLD/f"seed{seed}_{c.worker.REUSED[variant]}/training_state.pt"
                else:
                    path=c.ROOT/f"seed{seed}_{variant}"/("epoch10_state.pt" if budget==10 and variant in c.PRIMARY else "training_state.pt")
                state=torch.load(path,map_location="cpu",weights_only=False,mmap=True)
                if state["epoch"]!=budget:
                    raise ValueError("Wrong checkpoint budget")
                total=kept=numeric=forbidden=0
                for key,weight in state["model"].items():
                    if not key.endswith(".weight") or weight.ndim!=4:
                        continue
                    name=key[:-7]
                    mask=torch.ones_like(weight,dtype=torch.bool) if masks is None else masks[name]
                    count=int(mask.sum())
                    nz=int(torch.count_nonzero(weight))
                    bad=int(torch.count_nonzero(weight[~mask]))
                    if bad:
                        raise ValueError("Forbidden nonzero weights")
                    total+=weight.numel(); kept+=count; numeric+=nz; forbidden+=bad
                    layer_rows.append(dict(seed=seed,variant=variant,ft_epochs=budget,layer=name,total=weight.numel(),
                        structural_kept=count,numerical_nonzero=nz,kept_but_zero=count-nz,forbidden_nonzero=bad))
                rows.append(dict(seed=seed,variant=variant,ft_epochs=budget,total_conv_weights=total,structural_kept=kept,
                    numerical_nonzero=numeric,kept_but_zero=kept-numeric,forbidden_nonzero=forbidden,
                    structural_sparsity=1-kept/total,numerical_sparsity=1-numeric/total,
                    checkpoint=str(path.relative_to(c.PROJECT)),checkpoint_sha256=c.common.sha(path)))
    c.common.write_csv(c.ROOT/"numerical_sparsity_audit.csv",rows)
    c.common.write_csv(c.ROOT/"numerical_sparsity_layers.csv",layer_rows)
    paired=[]
    for budget in [0,1,3,10,30]:
        diffs=[(c.accuracy(seed,"C",budget)-c.accuracy(seed,"B",budget))*100 for seed in c.SEEDS]
        paired.append(dict(ft_epochs=budget,mean_C_minus_B_pp=statistics.mean(diffs),paired_sample_std_pp=statistics.stdev(diffs),
            positive_seeds=sum(d>0 for d in diffs),differences_json=json.dumps(diffs)))
    c.common.write_csv(c.ROOT/"C_vs_B_paired.csv",paired)
    source=Path(__file__).resolve()
    target=c.ROOT/"source_snapshot/analysis_audit"/source.relative_to(c.PROJECT)
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)
    c.common.write_json(c.ROOT/"DERIVED_AUDIT.json",dict(passed=True,checkpoints=len(rows),layers=len(layer_rows),
        source_sha256=c.common.sha(source),training_changed=False,test_evaluated=False,
        note="Numerical zeros inside scheduled kept positions do not imply additional skipped hardware cycles"))
    c.package()
    print(json.dumps(dict(checkpoints=len(rows),forbidden_nonzeros=sum(r["forbidden_nonzero"] for r in rows),
        total_kept_but_zero=sum(r["kept_but_zero"] for r in rows)),indent=2))


if __name__=="__main__":
    main()
