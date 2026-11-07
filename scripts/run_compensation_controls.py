#!/usr/bin/env python3
"""Audit cheap controls, reuse FT10 states, and finish a bounded three-seed study."""

import fcntl
import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import torch

import compensation_controls_worker as worker

common=worker.common
core=worker.core
ROOT=worker.ROOT
OLD=worker.OLD
PROJECT=worker.PROJECT
PRIMARY=worker.PRIMARY
REFERENCES=worker.REFERENCES
VARIANTS=worker.VARIANTS
SEEDS=[0,1,2]


def event(kind,**details):
    value=dict(time=common.stamp(),kind=kind,**details)
    with (ROOT/"execution.jsonl").open("a",encoding="utf-8") as stream:
        stream.write(json.dumps(value,ensure_ascii=False)+"\n")
    print(json.dumps(value,ensure_ascii=False),flush=True)


def expected(job):
    if job["mode"]=="prepare":
        return worker.prepared(job["seed"])/"ready.json"
    return ROOT/f"seed{job['seed']}_{job['variant']}"/f"ft{job['epochs']}_result.json"


def freeze_sources():
    sources={str(p.relative_to(PROJECT)):common.sha(p) for p in worker.source_files()}
    path=ROOT/"sources.json"
    if path.exists() and json.loads(path.read_text())!=sources:
        raise ValueError("Frozen study source changed; preserve prior evidence before explicit amendment")
    if not path.exists():
        common.write_json(path,sources)
        for source in worker.source_files():
            target=ROOT/"source_snapshot"/source.relative_to(PROJECT)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target)


def run_batch(jobs):
    pending=[j for j in jobs if not expected(j).exists()]
    active={}
    try:
        while pending or active:
            for gpu,(job,process,log) in list(active.items()):
                code=process.poll()
                if code is None:
                    continue
                log.close()
                del active[gpu]
                if code or not expected(job).exists():
                    event("job_failed",job=job,gpu=gpu,exit_code=code)
                    common.write_json(ROOT/"status.json",dict(state="failed",job=job,exit_code=code))
                    raise RuntimeError(f"Worker failed: {job}, exit={code}")
                event("job_complete",job=job,gpu=gpu)
            for gpu in [1,0]:
                if gpu in active or not pending:
                    continue
                free=int(subprocess.check_output(["nvidia-smi","-i",str(gpu),"--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
                available=os.sysconf("SC_AVPHYS_PAGES")*os.sysconf("SC_PAGE_SIZE")
                if free<12000 or available<16*1024**3 or shutil.disk_usage(PROJECT).free<15*1024**3:
                    continue
                job=pending.pop(0)
                label=f"{job['mode']}_seed{job['seed']}_{job.get('variant','all')}_e{job.get('epochs',0)}"
                log=(ROOT/"logs"/(label+".log")).open("a",buffering=1)
                command=[sys.executable,"-u",str(Path(worker.__file__).resolve()),job["mode"],"--seed",str(job["seed"])]
                if job["mode"]=="train":
                    command += ["--variant",job["variant"],"--epochs",str(job["epochs"])]
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4")
                process=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[gpu]=job,process,log
                event("job_start",job=job,gpu=gpu,pid=process.pid)
            common.write_json(ROOT/"status.json",dict(state="running",updated_at=common.stamp(),pending=pending,
                active=[dict(gpu=g,pid=p.pid,job=j) for g,(j,p,_) in active.items()]))
            if active or pending:
                time.sleep(20)
    finally:
        for _,process,log in active.values():
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL)
                    process.wait()
            log.close()


def read_result(seed,variant,budget):
    if variant in worker.REUSED and budget<=10:
        directory=OLD/f"seed{seed}_{worker.REUSED[variant]}"
        result=json.loads((directory/"result.json").read_text())
        history=common.read_csv(directory/"training_history.csv")
        result["history"]=[{key:int(value) if key=="epoch" else float(value) for key,value in row.items()} for row in history]
        result["ft1_accuracy"]=result["history"][0]["validation_accuracy"]
        result["ft3_accuracy"]=result["history"][2]["validation_accuracy"]
        result["variant"]=variant
        result["reused_ft10"]=True
        return result
    return json.loads((ROOT/f"seed{seed}_{variant}"/f"ft{budget}_result.json").read_text())


def accuracy(seed,variant,budget):
    result=read_result(seed,variant,30 if budget==30 else 10)
    return result["ft0_accuracy"] if budget==0 else result[f"ft{budget}_accuracy"]


def verify_prepared():
    checks=[]
    layers_all=[]
    for seed in SEEDS:
        ready=worker.check_ready(seed)
        model,specs,payload=worker.previous.load_model(seed,torch.device("cpu"))
        grouped=worker.previous.make_masks(specs,seed)
        shared=torch.load(worker.prepared(seed)/"shared_masks.pt",map_location="cpu",weights_only=False)
        quota=torch.load(worker.prepared(seed)/"quota_masks.pt",map_location="cpu",weights_only=False)
        global_masks=torch.load(worker.prepared(seed)/"global_masks.pt",map_location="cpu",weights_only=False)
        regenerated=worker.expand_shared(grouped,specs)
        counts={"shared":0,"quota":0,"global":0}
        macs={key:0 for key in counts}
        global_kept_min,global_removed_max=math.inf,-math.inf
        for spec in specs:
            name=spec.name
            if not torch.equal(shared[name],regenerated[name]):
                raise ValueError("Shared mask differs from fresh reference/core mask")
            if not torch.equal(shared[name].flatten(1).sum(1),quota[name].flatten(1).sum(1)):
                raise ValueError("Per-filter quota mismatch")
            values=spec.module.weight.detach().abs().flatten(1)
            flat_q=quota[name].flatten(1)
            kept_min=torch.where(flat_q,values,math.inf).amin(1)
            removed_max=torch.where(~flat_q,values,-math.inf).amax(1)
            if bool((kept_min<removed_max).any()):
                raise ValueError("Quota-unstructured is not independent magnitude top-q")
            flat_g=global_masks[name].flatten(1)
            if bool(flat_g.any()):
                global_kept_min=min(global_kept_min,float(values[flat_g].min()))
            if bool((~flat_g).any()):
                global_removed_max=max(global_removed_max,float(values[~flat_g].max()))
            hw=core.output_hw(spec.input_hw,spec.module)
            for label,masks in [("shared",shared),("quota",quota),("global",global_masks)]:
                counts[label]+=int(masks[name].sum())
                macs[label]+=int(masks[name].sum())*math.prod(hw)
            layers_all.append(dict(seed=seed,layer=name,total=spec.module.weight.numel(),shared_kept=int(shared[name].sum()),
                quota_kept=int(quota[name].sum()),global_kept=int(global_masks[name].sum()),is_first=name=="conv1",is_downsample="downsample" in name))
        if global_kept_min<global_removed_max:
            raise ValueError("Global-unstructured is not global magnitude top-q")
        if len(set(counts.values()))!=1 or macs["shared"]!=macs["quota"]:
            raise ValueError("Reference budget mismatch")
        for label in counts:
            if ready["totals"][label]!={"kept":counts[label],"macs":macs[label]}:
                raise ValueError("Reported budget mismatch")
        initial_a=torch.load(OLD/f"seed{seed}_original/initialization.pt",map_location="cpu",weights_only=False)["model"]
        scalar=torch.load(worker.prepared(seed)/"scalar_initialization.pt",map_location="cpu",weights_only=False)["model"]
        bn=torch.load(worker.prepared(seed)/"bn_initialization.pt",map_location="cpu",weights_only=False)["model"]
        rows=common.read_csv(worker.prepared(seed)/"scalar_reconstruction.csv")
        for spec in specs:
            values=scalar[spec.name+".weight"].flatten(1)
            if torch.count_nonzero(values[~shared[spec.name].flatten(1)]):
                raise ValueError("Scalar revived removed weights")
            dense=payload["model_state_dict"][spec.name+".weight"].flatten(1)
            selected=[r for r in rows if r["layer"]==spec.name]
            for row in selected:
                start=int(row["cluster"])*8
                alpha=torch.tensor(json.loads(row["alpha_json"]),dtype=torch.float64)
                if row["status"]!="accepted":
                    alpha.fill_(1)
                expected=(dense[start:start+8].double()*alpha[:,None]).float()*shared[spec.name].flatten(1)[start:start+8]
                torch.testing.assert_close(values[start:start+8],expected,rtol=0,atol=0)
                if float(row["check_mse_after"])>float(row["check_mse_before"])+1e-12:
                    raise ValueError("Scalar check fallback failed")
        allowed={name for name,_ in model.named_buffers() if name.endswith(("running_mean","running_var","num_batches_tracked"))}
        conv_keys={s.name+".weight" for s in specs}
        for name in initial_a:
            if name not in allowed and not torch.equal(bn[name],initial_a[name]):
                raise ValueError("BN changed parameter/affine")
            if name not in conv_keys and not torch.equal(scalar[name],initial_a[name]):
                raise ValueError("Scalar changed BN/nonconv state")
        checks.append(dict(seed=seed,quota_masks=True,global_order=True,scalar_solution=True,bn_only_buffers=True,actual_totals=ready["totals"]))
    common.write_csv(ROOT/"layers_audit.csv",layers_all)
    common.write_json(ROOT/"PRETRAIN_VALIDATION.json",dict(passed=True,checks=checks,test_evaluated=False))


def equal_nested(a,b):
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):
        return a.keys()==b.keys() and all(equal_nested(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):
        return len(a)==len(b) and all(equal_nested(x,y) for x,y in zip(a,b))
    return a==b


def verify_final():
    checks=[]
    bn_a=[]
    for seed in SEEDS:
        model,specs,_=worker.previous.load_model(seed,torch.device("cpu"))
        for variant in VARIANTS:
            directory=ROOT/f"seed{seed}_{variant}"
            state=torch.load(directory/"training_state.pt",map_location="cpu",weights_only=False,mmap=True)
            budget=30 if variant in PRIMARY else 10
            if state["epoch"]!=budget or [r["epoch"] for r in state["history"]]!=list(range(1,budget+1)):
                raise ValueError("Epoch trajectory gap/duplication")
            masks=worker.branch_masks(seed,variant)
            if masks:
                for spec in specs:
                    if torch.count_nonzero(state["model"][spec.name+".weight"][~masks[spec.name]]):
                        raise ValueError("Final forbidden weights regrew")
            result=read_result(seed,variant,budget)
            if result["test_evaluated"] or result[f"ft{budget}_accuracy"]!=state["history"][-1]["validation_accuracy"]:
                raise ValueError("Result/history mismatch")
            if variant in worker.REUSED:
                old_state=torch.load(OLD/f"seed{seed}_{worker.REUSED[variant]}/training_state.pt",map_location="cpu",weights_only=False,mmap=True)
                if state["history"][:10]!=old_state["history"]:
                    raise ValueError("Historical FT0..10 was changed")
            checks.append(dict(seed=seed,variant=variant,epochs=budget,no_regrowth=True,continuous_history=True))
        a=torch.load(OLD/f"seed{seed}_original/training_state.pt",map_location="cpu",weights_only=False,mmap=True)
        bn=torch.load(ROOT/f"seed{seed}_BN/epoch10_state.pt",map_location="cpu",weights_only=False,mmap=True)
        keys=[k for k in a["model"] if not k.endswith("num_batches_tracked")]
        mismatched=[k for k in keys if not torch.equal(a["model"][k],bn["model"][k])]
        bn_a.append(dict(seed=seed,budget=10,functional_state_bitexact=not mismatched,mismatched_keys=mismatched,
            optimizer_bitexact=equal_nested(a["optimizer"],bn["optimizer"]),scaler_bitexact=equal_nested(a["scaler"],bn["scaler"]),
            rng_bitexact=equal_nested(a["rng"],bn["rng"])))
    common.write_json(ROOT/"BN_A_EQUIVALENCE.json",bn_a)
    common.write_json(ROOT/"VALIDATION.json",dict(passed=True,checks=checks,test_evaluated=False))


def summarize(final=False):
    rows=[]
    for seed in SEEDS:
        for variant in VARIANTS:
            available30=(ROOT/f"seed{seed}_{variant}/ft30_result.json").exists()
            if variant in worker.REUSED:
                result=read_result(seed,variant,30 if available30 else 10)
            elif not (ROOT/f"seed{seed}_{variant}/ft10_result.json").exists():
                continue
            else:
                result=read_result(seed,variant,30 if available30 else 10)
            for budget in [0,1,3,10,30]:
                if budget==30 and not available30:
                    continue
                val=result["ft0_accuracy"] if budget==0 else result.get(f"ft{budget}_accuracy")
                if val is None:
                    continue
                rows.append(dict(seed=seed,variant=variant,ft_epochs=budget,validation_pct=val*100,
                    bn_clone_ft0_pct=result["ft0_bn_clone_accuracy"]*100,
                    sparsity=result["sparsity"],test_evaluated=False))
    common.write_csv(ROOT/"validation_long.csv",rows)
    summaries=[]
    for variant in VARIANTS:
        for budget in [0,1,3,10,30]:
            values=[r["validation_pct"] for r in rows if r["variant"]==variant and r["ft_epochs"]==budget]
            if values:
                summaries.append(dict(variant=variant,ft_epochs=budget,seeds=len(values),mean_pct=statistics.mean(values),sample_std_pp=statistics.stdev(values) if len(values)>1 else ""))
    common.write_csv(ROOT/"mean_results.csv",summaries)
    paired=[]
    for variant in ["B","C"]:
        for control in ["A","BN","scalar"]:
            for budget in [0,1,3,10,30]:
                diffs=[]
                for seed in SEEDS:
                    a=next((r["validation_pct"] for r in rows if r["seed"]==seed and r["variant"]==variant and r["ft_epochs"]==budget),None)
                    b=next((r["validation_pct"] for r in rows if r["seed"]==seed and r["variant"]==control and r["ft_epochs"]==budget),None)
                    if a is not None and b is not None:
                        diffs.append(a-b)
                if diffs:
                    paired.append(dict(variant=variant,control=control,ft_epochs=budget,seeds=len(diffs),mean_gain_pp=statistics.mean(diffs),paired_sample_std_pp=statistics.stdev(diffs) if len(diffs)>1 else "",positive_seeds=sum(d>0 for d in diffs),differences_json=json.dumps(diffs)))
    common.write_csv(ROOT/"paired_differences.csv",paired)
    if not final:
        return
    candidates=[]
    for budget in [10,30]:
        advantages=[next(r for r in paired if r["variant"]=="B" and r["control"]==control and r["ft_epochs"]==budget) for control in ["BN","scalar"]]
        jointly_positive=sum(all(accuracy(seed,"B",budget)>accuracy(seed,c,budget) for c in ["BN","scalar"]) for seed in SEEDS)
        candidates.append(dict(budget=budget,mean_vs_bn_pp=advantages[0]["mean_gain_pp"],mean_vs_scalar_pp=advantages[1]["mean_gain_pp"],joint_positive_seeds=jointly_positive,
            qualifies=all(r["mean_gain_pp"]>=.2-1e-12 for r in advantages) and jointly_positive>=2))
    decision=dict(next_round_5seed_qualified=any(r["qualifies"] for r in candidates),criteria=candidates,
        caution="Effort gate, not significance. Five-seed/cross-model/sparsity experiments have not been run in this study.",test_evaluated=False)
    common.write_json(ROOT/"decision.json",decision)
    costs=[]
    for seed in SEEDS:
        preparation=json.loads((worker.prepared(seed)/"ready.json").read_text())
        old_preparation=json.loads((worker.old_cache(seed)/"ready.json").read_text())
        for variant in VARIANTS:
            result=read_result(seed,variant,30 if variant in PRIMARY else 10)
            offline=preparation["scalar_seconds"] if variant=="scalar" else preparation["bn_seconds"] if variant=="BN" else 0.
            if variant in ["B","C"]:
                offline=json.loads((OLD/f"seed{seed}_{worker.REUSED[variant]}/compensation.json").read_text())["seconds"]
            costs.append(dict(seed=seed,variant=variant,ft_epochs=30 if variant in PRIMARY else 10,
                offline_method_seconds=offline,shared_teacher_cache_seconds=old_preparation["seconds"] if variant in ["scalar","B","C"] else 0.,
                reference_mask_preparation_seconds=preparation["reference_mask_seconds"] if variant in REFERENCES else 0.,
                training_seconds=result["training_seconds"],peak_gpu_bytes=result["peak_gpu_bytes"],
                excludes_queue_and_some_diagnostics=True))
    common.write_csv(ROOT/"costs.csv",costs)
    render(rows,summaries,paired,decision)


def render(rows,summaries,paired,decision):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors={"A":"#58636b","BN":"#8b969e","scalar":"#c59443","B":"#378f88","C":"#b35f79","dense_extra":"#778c42","global_unstructured":"#7866a1","quota_unstructured":"#4c8eae"}
    fig,axes=plt.subplots(1,3,figsize=(17,4.8))
    for variant in PRIMARY:
        selected=[r for r in summaries if r["variant"]==variant]
        axes[0].plot([r["ft_epochs"] for r in selected],[r["mean_pct"] for r in selected],marker="o",label=variant,color=colors[variant])
    axes[0].set_xlabel("Fine-tuning epochs")
    axes[0].set_ylabel("Validation accuracy (%)")
    axes[0].legend(fontsize=8)
    names=["A","BN","scalar","B","C","global_unstructured","quota_unstructured","dense_extra"]
    for i,budget in enumerate([10,30]):
        selected=[next((r for r in summaries if r["variant"]==v and r["ft_epochs"]==budget),None) for v in names]
        for x,r in enumerate(selected):
            if r is not None:
                axes[1].bar(x+(i-.5)*.34,r["mean_pct"],.34,color=colors[r["variant"]],alpha=1 if i==0 else .45,
                    yerr=r["sample_std_pp"],capsize=2)
        axes[1].plot([],[],color="#555555",alpha=1 if i==0 else .45,linewidth=6,label=f"FT{budget}")
    axes[1].set_xticks(range(len(names)),["A","BN","Scalar","B","C","Global U","Quota U","Dense"],rotation=45,ha="right")
    relevant=[r["mean_pct"] for r in summaries if r["ft_epochs"] in [10,30]]
    axes[1].set_ylim(min(relevant)-1,max(relevant)+.8)
    axes[1].set_ylabel("Mean accuracy; sample std (%)")
    axes[1].legend()
    for control in ["A","BN","scalar"]:
        selected=[r for r in paired if r["variant"]=="B" and r["control"]==control and r["ft_epochs"]>0]
        axes[2].errorbar([r["ft_epochs"] for r in selected],[r["mean_gain_pp"] for r in selected],yerr=[r["paired_sample_std_pp"] for r in selected],marker="o",label="B - "+control,capsize=3)
    axes[2].axhline(0,color="#555555",linewidth=.8)
    axes[2].set_xlabel("Fine-tuning epochs")
    axes[2].set_ylabel("Paired gain / sample std (pp)")
    axes[2].legend()
    fig.suptitle("Shared-mask compensation controls / ResNet50 / CIFAR-10 / Tm8 / 3 Dense seeds")
    fig.tight_layout()
    fig.savefig(ROOT/"comparison.png",dpi=170)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.8))
    for variant in PRIMARY:
        selected=[r for r in summaries if r["variant"]==variant and r["ft_epochs"]>0]
        ax.errorbar([r["ft_epochs"] for r in selected],[r["mean_pct"] for r in selected],
            yerr=[r["sample_std_pp"] for r in selected],marker="o",label=variant,color=colors[variant],capsize=3)
    ax.set_xlabel("Fine-tuning epochs")
    ax.set_ylabel("Validation mean / sample std (%)")
    ax.set_title("Post-adaptation trajectories / 3 independent Dense seeds")
    ax.legend(ncol=3,fontsize=9)
    fig.tight_layout()
    fig.savefig(ROOT/"late_curves.png",dpi=170)
    plt.close(fig)
    def value(variant,budget):
        row=next((r for r in summaries if r["variant"]==variant and r["ft_epochs"]==budget),None)
        return "N/A" if row is None else f"{row['mean_pct']:.3f} +/- {row['sample_std_pp']:.3f}"
    lines=["# 共享Mask補償：便宜控制與FT30證據鏈","",f"完成時間：{common.stamp()}","",
        "## 評估規則","","ResNet50/CIFAR-10、Tm8、三個獨立100epoch Dense seeds0/1/2。train45000/val5000，split_seed0；原10000張test曾在早期使用，本輪沒有新test forward。校正fit512/check512為互不重疊train子集。所有數據為validation，不宣稱未見test或統計顯著。",
        "","剪53個一般Conv2d，含首層、1x1及downsample，不剪FC/BN。A/BN/scalar/B/C使用每seed完全相同global_cluster flat l1共享mask；實際sparsity與逐層/cluster q見prepared及layers_audit.csv。global U只同總weights；quota U精確同每filter q及MAC。非結構化是參考，不是精度嚴格上界。",
        "","## 方法","","A原權重；BN只預校正BN running statistics；scalar每filter一純量且check變差則整cluster fallback；B為Dense-input ridge主線；C為student-prefix逐層ridge；Dense為未剪的同預算追加訓練。全部同SGD/AMP及FT RNG，不調lambda、不挑最好epoch。",
        "","A/B/C直接重用既有FT0..10並載入完整epoch10狀態續跑11..30。其餘主要分支0..30為同一條軌跡，兩個U控制只跑FT10。長微調使用相同constant lr=.001，沒有重啟optimizer/scaler/RNG。",
        "","## 平均結果","","以下mean +/- sample std，單位%。","","| 方法 | FT0 | FT1 | FT3 | FT10 | FT30 |","|---|---:|---:|---:|---:|---:|"]
    for variant in VARIANTS:
        lines.append("| "+variant+" | "+" | ".join(value(variant,b) for b in [0,1,3,10,30])+" |")
    lines += ["","![比較圖](comparison.png)","","## 配對提升","","以下同Dense seed逐一相減後平均及sample std，單位pp。","","| 方法 | 控制 | FT | Mean gain | Paired std | 正值seeds |","|---|---|---:|---:|---:|---:|"]
    for row in paired:
        if row["ft_epochs"] in [1,3,10,30]:
            lines.append(f"| {row['variant']} | {row['control']} | {row['ft_epochs']} | {row['mean_gain_pp']:+.3f} | {row['paired_sample_std_pp']:.3f} | {row['positive_seeds']}/3 |")
    lines += ["","## 下一輪決策","",f"預先5seed投入資格：{decision['next_round_5seed_qualified']}。FT10或FT30中B對BN與scalar均平均>=0.2pp，且至少2/3seed同時對兩者為正，才建議下一輪確認。這不是顯著性檢定；本輪未跑5seed、跨模型或80/95%。詳見decision.json。",
        "","## 驗證與界線","","PRETRAIN_VALIDATION.json檢查fresh core mask、global threshold、逐filter q、scalar解與fallback、BN僅改buffers及actual budgets。VALIDATION.json檢查連續epoch與禁止位置；BN_A_EQUIVALENCE.json另對BN/A epoch10的functional state、optimizer、scaler、RNG逐bit比較，不把num_batches_tracked誤當推論差異。",
        "","53項以上既有及新增測試見UNIT_TESTS.txt。來源/資料SHA見sources.json及prepared/configuration.json。既有資料只讀，所有新checkpoint放本資料夾；source_snapshot保存本輪計畫與程式。",
        "","## 資料導覽","","- validation_long.csv、mean_results.csv、paired_differences.csv：原始軌跡及配對統計。",
        "- prepared/seed*/layers.csv、clusters.csv：所有層與cluster配額。",
        "- prepared/seed*/scalar_reconstruction.csv：alpha、fit/check誤差及fallback。",
        "- seed*/：FT10/30結果、完整resume狀態與最終模型。",
        "- BN_A_EQUIVALENCE.json：BN控制的作用是否在訓練後消失。",
        "- costs.csv、late_curves.png：離線/訓練成本及不含FT0的放大軌跡。",
        "- EXECUTION_LOG.md、execution.jsonl、logs/：執行與問題處理。",
        "","## 成本與研究界線","","曲線的epoch不是總成本；FT seconds不含全部teacher cache準備/求解/診斷/排隊。時間為共享GPU環境，不宣稱推論加速。未做新量化、RTL、分組搜尋、activation加權或補償感知mask。Ridge及二階補償有既有文獻，不把概念本身稱原創。"]
    (ROOT/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def package():
    files=[p for p in ROOT.rglob("*") if p.is_file() and p.suffix!=".pt" and p.name!="MANIFEST.json" and not p.name.endswith((".lock",".tmp"))]
    common.write_json(ROOT/"MANIFEST.json",dict(files=[dict(path=str(p.relative_to(ROOT)),sha256=common.sha(p),bytes=p.stat().st_size) for p in files],
        exclusions="All large .pt masks, initializations and model states remain locally; per-cluster q, scalar alpha and source hashes included"))
    destination=ROOT.with_suffix(".zip")
    with zipfile.ZipFile(destination,"w",zipfile.ZIP_DEFLATED) as archive:
        for p in [*files,ROOT/"MANIFEST.json"]:
            archive.write(p,str(p.relative_to(ROOT.parent)))
    destination.with_suffix(".zip.sha256").write_text(common.sha(destination)+"  "+destination.name+"\n")


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/"logs").mkdir(exist_ok=True)
    torch.set_num_threads(4)
    lock=(ROOT/"study.lock").open("w")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    freeze_sources()
    with (ROOT/"UNIT_TESTS.txt").open("w") as stream:
        result=subprocess.run([sys.executable,"-m","unittest","discover","-s","tests","-v"],cwd=PROJECT,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError("Unit tests failed")
    event("study_start_or_resume",stage="three-seed minimum evidence chain")
    permutation=torch.randperm(50000,generator=torch.Generator().manual_seed(0)).tolist()
    common.write_json(ROOT/"DATA_PROTOCOL.json",dict(train_indices=permutation[5000:],validation_indices=permutation[:5000],
        split_seed=0,split_id="42fbb7440cc27c71",test_size=10000,new_test_forward=False,historical_test_used=True,
        independent_dense_seeds=SEEDS,finetune_rng_seeds=[s+100000 for s in SEEDS],calibration_role="fit/check subsets of train only"))
    run_batch([dict(mode="prepare",seed=s) for s in SEEDS])
    verify_prepared()
    run_batch([dict(mode="train",seed=s,variant=v,epochs=10) for s in SEEDS for v in ["BN","scalar",*REFERENCES,"dense_extra"]])
    summarize()
    event("ft10_controls_complete")
    run_batch([dict(mode="train",seed=s,variant=v,epochs=30) for s in SEEDS for v in PRIMARY])
    freeze_sources()
    verify_prepared()
    verify_final()
    summarize(final=True)
    entries=[json.loads(line) for line in (ROOT/"execution.jsonl").read_text().splitlines()]
    lines=["# 執行與問題紀錄","","本輪完成三seed的24條軌跡：9個A/B/C的FT10狀態重用並續訓20epochs，9個BN/scalar/Dense軌跡連續FT30，6個非共享參考FT10。沒有新test forward。",
        "","- 啟動時兩張L40S空閒；每卡只新增一個worker，未動外部程序。",
        "- 原始cache/結果/程式只讀，新方法全部獨立資料夾與來源snapshot。",
        "- 舊epoch10狀態續訓前重新驗證validation並還原probe消耗的RNG；所有續訓保留optimizer/scaler/RNG。",
        "- 若發生錯誤保留log與checkpoint，不能修改lambda或epoch追高。","","## 事件","",
        *[f"- {e['time']} {e['kind']} {json.dumps({k:v for k,v in e.items() if k not in ['time','kind']},ensure_ascii=False)}" for e in entries]]
    (ROOT/"EXECUTION_LOG.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    common.write_json(ROOT/"status.json",dict(state="complete",completed_at=common.stamp(),dense_seeds=SEEDS,trajectories=24,test_evaluated=False))
    event("study_complete")
    package()


if __name__=="__main__":
    main()
