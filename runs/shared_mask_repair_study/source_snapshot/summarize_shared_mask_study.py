#!/usr/bin/env python3
"""Audit and package the complete bounded shared-mask study, including negative results."""

import json
import shutil
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path

import torch

import run_fair_criterion_r0p9 as common


PROJECT=common.PROJECT
ROOT=PROJECT/"runs/shared_mask_repair_study"


def main():
    if not (ROOT/"adjustment_result.json").exists():
        raise RuntimeError("The bounded adjustment must finish before final packaging")
    audits=[]
    split=torch.randperm(50000,generator=torch.Generator().manual_seed(0))[:5000].tolist()
    for directory in sorted(ROOT.glob("v*/*_seed*")):
        if not (directory/"done.json").exists():
            continue
        masks=torch.load(directory/"masks.pt",map_location="cpu",weights_only=False)
        for variant,layers in masks.items():
            for name,mask in layers.items():
                if mask.dtype!=torch.bool or not torch.equal(mask.sum(1),masks["l1"][name].sum(1)):
                    raise ValueError(f"Mask quota changed: {directory}/{variant}/{name}")
        idx=json.loads((directory/"calibration_indices.json").read_text())
        a,b=set(idx["search_train_indices"]),set(idx["check_train_indices"])
        if a&b or (a|b)&set(split):
            raise ValueError("Calibration data leakage")
        base=ROOT/"v1"/directory.name/"masks.pt"
        if directory.parent.name!="v1" and base.exists():
            original=torch.load(base,map_location="cpu",weights_only=False)["l1"]
            if not all(torch.equal(original[name],mask) for name,mask in masks["l1"].items()):
                raise ValueError("Original l1 mask changed between study versions")
        audits.append({"model_directory":str(directory.relative_to(ROOT)),"quota_check":"passed",
                       "calibration_validation_disjoint":"passed","mask_sha256":common.sha(directory/"masks.pt"),
                       "actual_sparsity":json.loads((directory/"done.json").read_text())["actual_sparsity"]})
    common.write_json(ROOT/"VALIDATION.json",{"status":"passed","model_runs":audits,
                      "note":"Checks confirm reproducible mask/data constraints; they do not prove novelty or statistical significance."})
    diagnostics=[]
    pilots=[]
    for directory in ROOT.glob("v*/*_seed*"):
        if (directory/"summary.csv").exists():
            diagnostics.extend(dict(version=directory.parent.name,**r) for r in common.read_csv(directory/"summary.csv"))
        for p in sorted((directory/"pilots").glob("*.json")):
            pilots.append(dict(version=directory.parent.name,**json.loads(p.read_text())))
    common.write_csv(ROOT/"all_diagnostic_summary.csv",diagnostics)
    common.write_csv(ROOT/"all_pilot_validation.csv",pilots)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharey=True)
    variants=["l1","normalized","mean_repair","worst_repair","random"]
    colors=["#777777","#4b82a0","#ba7538","#338c72","#ad4f76"]
    for row,version in enumerate(["v1","v3_mean_safe"]):
        for column,model in enumerate(["resnet50","densenet121"]):
            ax=axes[row,column]
            samples={r["variant"]:r for r in pilots if r["version"]==version and r["model"]==model and r["seed"]==0}
            baseline=samples["l1"]["final_validation_acc"]
            gains=[(samples[v]["final_validation_acc"]-baseline)*100 for v in variants]
            ax.bar(range(5),gains,color=colors)
            for i,value in enumerate(gains):
                ax.text(i,value+(.025 if value>=0 else -.025),f"{value:+.2f}",ha="center",va="bottom" if value>=0 else "top",fontsize=9)
            ax.axhline(0,color="black",linewidth=.8)
            ax.set_ylim(-1.4,.5)
            ax.set_xticks(range(5),["L1","Norm","Mean","Tail","Random"])
            ax.set_title(f"{model}, {version}; L1={baseline*100:.2f}%")
            ax.set_ylabel("Validation gain vs L1 (pp)")
    fig.suptitle("Classification pilots: seed0 only, fixed ~90% pruning and 10 epochs")
    fig.tight_layout()
    fig.savefig(ROOT/"validation_accuracy_comparison.png",dpi=160)
    plt.close(fig)
    lines=["# 固定配額共享Mask修補：結果與問題處理", "",f"整理：{common.stamp()}","",
           "## 實驗限制", "", "固定Tm8、逐cluster kept_count、實際約90%權重剪枝、固定Dense checkpoints。每個訓練分支fine-tune10 epochs。校正資料來自train，設定調整只用校正／validation。", "",
           "v1是最多8輪的最壞誤差修補；v3_mean_safe先8輪平均、再8輪受限尾端修補，mean對照為16輪。兩版本不可混稱同一方法。", "",
           "## 五模型seed0診斷", "", "| Model | 最壞誤差改善 (%) | 平均相對誤差變化 (%) | Gate |", "|---|---:|---:|---|"]
    for row in diagnostics:
        if row["version"]=="v1" and row["seed"]=="0" and row["variant"]=="worst_repair":
            lines.append(f"| {row['model']} | {float(row['check_worst_improvement_pct']):.3f} | {float(row['check_mean_increase_pct']):.3f} | {row['passes_diagnostic_gate']} |")
    lines += ["", "## Validation消融", "", "這些結果包含開發選擇；不能把單seed validation改善當成獨立test或三seed結論。", "",
              "| Version | Model | Seed | Variant | Validation accuracy (%) | 相對同seed l1提升 (pp) |", "|---|---|---:|---|---:|---:|"]
    for r in pilots:
        baseline=next(p for p in pilots if (p["version"],p["model"],p["seed"],p["variant"])==(r["version"],r["model"],r["seed"],"l1"))
        gain=(r["final_validation_acc"]-baseline["final_validation_acc"])*100
        lines.append(f"| {r['version']} | {r['model']} | {r['seed']} | {r['variant']} | {r['final_validation_acc']*100:.3f} | {gain:+.3f} |")
    lines += ["", "![分類validation消融](validation_accuracy_comparison.png)"]
    result=json.loads((ROOT/"adjustment_result.json").read_text())
    if not result["confirmed_candidates"]:
        lines += ["", "## 本輪研究結論", "",
                  "固定配額交換可在五模型降低獨立檢查重建誤差，但兩個seed0分類pilot均未支持所提尾端修補方法優於l1。收緊平均誤差約束的mean-safe版本仍未帶來分類優勢，因此未擴大三seed，也未讀取test accuracy。",
                  "", "這是目前設定下未取得支持的初步負結果，不是證明所有mask修補都無效。兩模型、單seed的validation差距不能宣稱統計顯著或普遍劣於原方法。",
                  "", "實際完成7個模型／版本診斷及16個新fine-tune分支，另有4個逐元素相同mask的控制結果復用。隨機交換控制在DenseNet121反而較高，更不宜只憑小幅單seed差異宣稱新importance有效。"]
    if (ROOT/"confirmation_test.csv").exists():
        grouped=defaultdict(list)
        for row in common.read_csv(ROOT/"confirmation_test.csv"):
            grouped[(row["version"],row["model"],row["variant"])].append(row)
        lines += ["", "## 固定方法後的Test確認", "", "依validation預先鎖定候選，以下test不回饋參數選擇。", "",
                  "| Version | Model | Variant | Seeds | Test accuracy mean (%) | Sample std (pp) |", "|---|---|---|---:|---:|"]
        for (version,model,variant),rows in sorted(grouped.items()):
            values=[float(r["test_accuracy"])*100 for r in rows]
            lines.append(f"| {version} | {model} | {variant} | {len(values)} | {statistics.mean(values):.3f} | {statistics.stdev(values) if len(values)>1 else 0:.3f} |")
    else:
        lines += ["", "## 本輪停止條件", "", "本輪未進入test確認。若候選未通過validation對l1與mean對照的條件，就保留負結果，避免在test上繼續尋找有利設定。"]
    lines += ["", "## 已確認與尚未確認", "",
              "- 校正重建誤差與分類accuracy是不同指標，不能由前者改善直接宣稱分類有效。",
              "- 所有版本及負結果均保留；共同保留數、資料隔離與mask hash檢查見VALIDATION.json。",
              "- 本輪未更改原始硬體資料格式；相同MAC並不保證相同memory-bank行為或量測energy。",
              "- 學術新穎性仍須深入文獻比對；本實驗只提供機制及分類效果證據。", "",
              "## 問題、解法與重現", "", "詳細過程見EXECUTION_LOG.md及execution.jsonl；各cluster指標、交換位置與設定在v*/<model>_seed*/。", "",
              "ZIP包含mask、數據、圖表與程式快照，不包含大型校正patches、完整訓練權重或CIFAR資料。這些大型檔案仍在本機。", "",
              "![獨立檢查診斷](diagnostic_gates.png)"]
    (ROOT/"FINAL_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    snapshot=ROOT/"source_snapshot"
    for name in ["run_mean_safe_mask_repair.py","run_shared_mask_adjustment.py","summarize_shared_mask_study.py"]:
        shutil.copy2(PROJECT/"scripts"/name,snapshot/name)
    shutil.copy2(common.CORE,snapshot/common.CORE.name)
    shutil.copy2(PROJECT/"scripts/run_fair_criterion_r0p9.py",snapshot/"run_fair_criterion_r0p9.py")
    shutil.copy2(PROJECT/"tests/test_shared_mask_repair.py",snapshot/"test_shared_mask_repair.py")
    files=[p for p in ROOT.rglob("*") if p.is_file() and (p.suffix in {".json",".jsonl",".md",".csv",".png",".py"} or p.name=="masks.pt")]
    common.write_json(ROOT/"MANIFEST.json",{"generated_at":common.stamp(),"files":[{"path":str(p.relative_to(ROOT)),"bytes":p.stat().st_size,"sha256":common.sha(p)} for p in sorted(files) if p.name!="MANIFEST.json"]})
    archive=ROOT.with_suffix(".zip")
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as stream:
        for path in sorted(set(files+[ROOT/"MANIFEST.json"])):
            stream.write(path,path.relative_to(ROOT.parent))
    with zipfile.ZipFile(archive) as stream:
        if stream.testzip() is not None:
            raise ValueError("Archive failed integrity check")
    archive.with_suffix(".zip.sha256").write_text(common.sha(archive)+"  "+archive.name+"\n",encoding="ascii")
    print(json.dumps({"report":str(ROOT/"FINAL_REPORT.md"),"archive":str(archive),"archive_mb":archive.stat().st_size/1048576,
                      "validated_model_runs":len(audits),"test_evaluated":result["test_evaluated"]},indent=2))


if __name__=="__main__":
    main()
