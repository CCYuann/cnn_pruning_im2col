#!/usr/bin/env python3
"""Report-only NumPy RNG audit fix and predeclared accuracy-gap decomposition."""

import argparse
import fcntl
import json
import shutil
import statistics
from pathlib import Path

import numpy as np
import torch


def equal_nested(a,b):
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):
        return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
    if isinstance(a,dict):
        return isinstance(b,dict) and a.keys()==b.keys() and all(equal_nested(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):
        return isinstance(b,(list,tuple)) and len(a)==len(b) and all(equal_nested(x,y) for x,y in zip(a,b))
    return bool(a==b)


def gap_reports(c):
    rows=[]
    for seed in c.SEEDS:
        acc={v:c.accuracy(seed,v,10)*100 for v in c.VARIANTS}
        shared_gap=acc["quota_unstructured"]-acc["A"]
        rows.append(dict(seed=seed,dense_extra_ft10_pct=acc["dense_extra"],global_unstructured_ft10_pct=acc["global_unstructured"],
            quota_unstructured_ft10_pct=acc["quota_unstructured"],A_ft10_pct=acc["A"],B_ft10_pct=acc["B"],C_ft10_pct=acc["C"],
            dense_to_global_gap_pp=acc["dense_extra"]-acc["global_unstructured"],matched_shared_mask_gap_pp=shared_gap,
            matched_remaining_B_gap_pp=acc["quota_unstructured"]-acc["B"],matched_remaining_C_gap_pp=acc["quota_unstructured"]-acc["C"],
            B_recovery_of_matched_gap_pct=100*(acc["B"]-acc["A"])/shared_gap if shared_gap>1e-8 else None))
    c.common.write_csv(c.ROOT/"gap_decomposition.csv",rows)
    numerator=statistics.mean(r["B_ft10_pct"]-r["A_ft10_pct"] for r in rows)
    denominator=statistics.mean(r["matched_shared_mask_gap_pp"] for r in rows)
    c.common.write_json(c.ROOT/"gap_summary.json",dict(
        mean_dense_to_global_gap_pp=statistics.mean(r["dense_to_global_gap_pp"] for r in rows),
        mean_matched_shared_mask_gap_pp=denominator,mean_matched_remaining_B_gap_pp=statistics.mean(r["matched_remaining_B_gap_pp"] for r in rows),
        B_recovery_ratio_of_mean_matched_gaps_pct=100*numerator/denominator if denominator>1e-8 else None,
        caution="Empirical reference gaps, not mathematical accuracy bounds or relative error-rate reductions"))


def bn_difference_report(c):
    rows=[]
    summaries=[]
    for seed in c.SEEDS:
        for budget in [10,30]:
            a_path=c.OLD/f"seed{seed}_original/training_state.pt" if budget==10 else c.ROOT/f"seed{seed}_A/training_state.pt"
            bn_path=c.ROOT/f"seed{seed}_BN"/("epoch10_state.pt" if budget==10 else "training_state.pt")
            a=torch.load(a_path,map_location="cpu",weights_only=False,mmap=True)
            b=torch.load(bn_path,map_location="cpu",weights_only=False,mmap=True)
            parameters_equal=True
            max_buffer_delta=0.
            for key in a["model"]:
                if key.endswith("num_batches_tracked"):
                    continue
                same=torch.equal(a["model"][key],b["model"][key])
                is_running=key.endswith(("running_mean","running_var"))
                if not is_running and not same:
                    parameters_equal=False
                if not same:
                    delta=float((a["model"][key]-b["model"][key]).abs().max())
                    if is_running:
                        max_buffer_delta=max(max_buffer_delta,delta)
                    rows.append(dict(seed=seed,ft_epochs=budget,key=key,is_running_buffer=is_running,max_abs_difference=delta,
                        differing_values=int(torch.count_nonzero(a["model"][key]!=b["model"][key]))))
            summaries.append(dict(seed=seed,ft_epochs=budget,all_parameters_bitexact=parameters_equal,
                max_running_buffer_difference=max_buffer_delta,optimizer_bitexact=equal_nested(a["optimizer"],b["optimizer"]),
                scaler_bitexact=equal_nested(a["scaler"],b["scaler"]),rng_bitexact=equal_nested(a["rng"],b["rng"]),
                validation_difference_pp=(c.accuracy(seed,"BN",budget)-c.accuracy(seed,"A",budget))*100))
    c.common.write_csv(c.ROOT/"BN_A_DIFFERENCE_DETAIL.csv",rows,fields=["seed","ft_epochs","key","is_running_buffer","max_abs_difference","differing_values"])
    c.common.write_json(c.ROOT/"BN_A_PARAMETER_AUDIT.json",summaries)


def main():
    import run_compensation_controls as c
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-controller",action="store_true")
    args=parser.parse_args()
    c.ROOT.mkdir(parents=True,exist_ok=True)
    if args.wait_controller:
        print("Waiting for the frozen training controller to release its study lock",flush=True)
        with (c.ROOT/"study.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
    for seed in c.SEEDS:
        for variant in c.VARIANTS:
            budget=30 if variant in c.PRIMARY else 10
            if not (c.ROOT/f"seed{seed}_{variant}/ft{budget}_result.json").exists():
                raise RuntimeError("Finish training before running this report-only finalizer")
    c.equal_nested=equal_nested
    files=[Path(__file__).resolve(),c.PROJECT/"tests/test_compensation_report_audit.py"]
    for source in files:
        target=c.ROOT/"source_snapshot"/"reporting_fix"/source.relative_to(c.PROJECT)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
    c.common.write_json(c.ROOT/"REPORTING_FIX.json",dict(time=c.common.stamp(),
        issue="Original reporting-only equal_nested did not handle NumPy arrays in saved RNG state",
        remedy="Use np.array_equal in the in-memory finalizer only; frozen worker/controller sources and all training states unchanged",
        sources={str(p.relative_to(c.PROJECT)):c.common.sha(p) for p in files},training_recomputed=False))
    c.event("report_only_numpy_audit_fix",training_recomputed=False)
    c.main()
    gap_reports(c)
    bn_difference_report(c)
    log=c.ROOT/"EXECUTION_LOG.md"
    with log.open("a",encoding="utf-8") as stream:
        stream.write("\n## 報表問題與修正\n\n原報表equal_nested對NumPy RNG array使用==，會產生ambiguous truth ValueError。訓練方法及所有checkpoint無影響。以獨立report-only finalizer用np.array_equal覆核，原凍結程式不變，新增來源與回歸測試記於REPORTING_FIX.json。沒有因此重算訓練。gap_decomposition.csv另明確拆解global參考與同配額非共享mask差距，均非數學上界。\n")
    c.package()


if __name__=="__main__":
    main()
