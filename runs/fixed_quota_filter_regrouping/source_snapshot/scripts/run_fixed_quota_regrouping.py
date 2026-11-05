#!/usr/bin/env python3
"""Run a frozen, conditional regrouping study through verification and reporting."""

import fcntl
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import run_fair_criterion_r0p9 as common
from fixed_quota_grouping import VARIANTS

PROJECT = common.PROJECT
ROOT = PROJECT/"runs/fixed_quota_filter_regrouping"
PLAN = PROJECT/"research_notes/2026-11-02_fixed_quota_regrouping_plan.md"
WORKER = PROJECT/"scripts/fixed_quota_grouping_worker.py"


def event(kind, **details):
    value = dict(time=common.stamp(),kind=kind,**details)
    with (ROOT/"execution.jsonl").open("a",encoding="utf-8") as stream:
        stream.write(json.dumps(value,ensure_ascii=False)+"\n")
    print(json.dumps(value,ensure_ascii=False),flush=True)


def output(job):
    if job["mode"] == "prepare":
        return ROOT/"prepared"/f"seed{job['seed']}"/"ready.json"
    return ROOT/f"seed{job['seed']}_{job['variant']}"/"result.json"


def run_batch(jobs):
    pending = [j for j in jobs if not output(j).exists()]
    active = {}
    try:
        while pending or active:
            for gpu,(job,process,log) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del active[gpu]
                if code or not output(job).exists():
                    event("job_failed",job=job,gpu=gpu,exit_code=code)
                    raise RuntimeError(f"Worker failed: {job}, exit={code}; inspect logs")
                event("job_complete",job=job,gpu=gpu)
            for gpu in [1,0]:
                if gpu in active or not pending:
                    continue
                free = int(subprocess.check_output(["nvidia-smi","-i",str(gpu),"--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
                if free < 12000:
                    continue
                job = pending.pop(0)
                label = f"{job['mode']}_seed{job['seed']}_{job.get('variant','all')}"
                log = (ROOT/"logs"/(label+".log")).open("a",buffering=1)
                command = [sys.executable,"-u",str(WORKER),job["mode"],"--root",str(ROOT),"--seed",str(job["seed"])]
                if job["mode"] == "train":
                    command += ["--variant",job["variant"]]
                env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4")
                process = subprocess.Popen(command,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[gpu] = job,process,log
                event("job_start",job=job,gpu=gpu,pid=process.pid)
            common.write_json(ROOT/"status.json",dict(state="running",updated_at=common.stamp(),pending=pending,
                active=[dict(gpu=g,pid=p.pid,job=j) for g,(j,p,_) in active.items()]))
            if pending or active:
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


def score(seed,variant):
    return json.loads((ROOT/f"seed{seed}_{variant}"/"result.json").read_text())["ft10_accuracy"]*100


def freeze_sources():
    sources = [Path(__file__).resolve(),WORKER,PROJECT/"scripts/fixed_quota_grouping.py",PLAN,
        PROJECT/"tests/test_fixed_quota_grouping.py",common.CORE,
        *[PROJECT/"scripts"/name for name in ["run_fair_criterion_r0p9.py","run_shared_mask_repair.py",
        "weight_compensation_worker.py","adapted_gradient_mask.py","task_loss_mask_search.py"]]]
    digests = {str(p.relative_to(PROJECT)):common.sha(p) for p in sources}
    path = ROOT/"sources.json"
    if path.exists() and json.loads(path.read_text()) != digests:
        raise ValueError("Sources changed since the frozen study; do not silently resume")
    if not path.exists():
        common.write_json(path,digests)
        for source in sources:
            target = ROOT/"source_snapshot"/source.relative_to(PROJECT)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target)


def verify(seeds):
    import torch
    from fixed_quota_grouping import discarded, grouping_loss, validate_groups
    from weight_compensation_worker import load_model
    torch.set_num_threads(4)
    checks, clusters = [], []
    for seed in seeds:
        directory = ROOT/"prepared"/f"seed{seed}"
        ready = json.loads((directory/"ready.json").read_text())
        for name,digest in ready["files_sha256"].items():
            if common.sha(directory/name) != digest:
                raise ValueError("Prepared checksum mismatch")
        model,specs,_ = load_model(seed,torch.device("cpu"))
        quotas, costs = {}, {}
        metrics = common.read_csv(directory/"layer_metrics.csv")
        for variant in VARIANTS:
            pack = torch.load(directory/f"{variant}.pt",map_location="cpu",weights_only=False)
            final = torch.load(ROOT/f"seed{seed}_{variant}/final_model.pt",map_location="cpu",weights_only=False)["model_state_dict"]
            result = json.loads((ROOT/f"seed{seed}_{variant}/result.json").read_text())
            if result["test_evaluated"] or result["epochs"] != 10 or result["grouping_sha256"] != common.sha(directory/f"{variant}.pt"):
                raise ValueError("Incorrect result metadata")
            layer_q, layer_cost = {}, {}
            for spec in specs:
                rec = pack[spec.name]
                groups,masks,q = rec["groups"],rec["masks"],rec["q"]
                validate_groups(groups,spec.module.out_channels)
                if not bool((masks.sum(1)==q).all()):
                    raise ValueError("Cluster quota mismatch")
                full = torch.zeros_like(spec.module.weight.flatten(1),dtype=torch.bool)
                # Independent row-by-row scatter check avoids a contiguous-ID assumption.
                for group_id, members in enumerate(groups.tolist()):
                    for filter_id in members:
                        full[filter_id] = masks[group_id]
                    clusters.append(dict(seed=seed,variant=variant,layer=spec.name,cluster=group_id,
                        kept_count=q,filter_indices=",".join(map(str,members))))
                if not bool((full.sum(1)==q).all()) or torch.count_nonzero(final[spec.name+".weight"].flatten(1)[~full]):
                    raise ValueError("Mask mapping or regrowth error")
                weights = spec.module.weight.detach().flatten(1)
                cost = float((weights.double().abs()*~full).sum())
                proxy = float(grouping_loss(weights,groups,q))
                if abs(cost-proxy) > max(1e-7,abs(cost)*1e-10):
                    raise ValueError("Saved mask does not minimize the group L1 objective")
                row = next(r for r in metrics if r["variant"]==variant and r["layer"]==spec.name)
                if abs(cost-float(row["discarded_l1"])) > max(1e-7,abs(cost)*1e-10):
                    raise ValueError("Saved proxy mismatch")
                layer_q[spec.name],layer_cost[spec.name] = q,cost
            quotas[variant],costs[variant] = layer_q,layer_cost
            checks.append(dict(seed=seed,variant=variant,partition=True,quotas=True,scatter=True,proxy=True,no_regrowth=True,test_evaluated=False))
        if any(quotas[v]!=quotas["contiguous"] for v in VARIANTS):
            raise ValueError("Methods have different layer quotas")
        for name in costs["jaccard"]:
            if costs["jaccard_swap"][name] > costs["jaccard"][name]+1e-7:
                raise ValueError("Swap worsened the objective")
    common.write_csv(ROOT/"cluster_metadata.csv",clusters)
    common.write_json(ROOT/"VALIDATION.json",dict(passed=True,checks=checks,source_hashes=True))


def report(seeds,decision):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    results,layer_rows,proxy = [],[],{}
    for seed in seeds:
        metrics = common.read_csv(ROOT/"prepared"/f"seed{seed}/layer_metrics.csv")
        layer_rows.extend(dict(seed=seed,**r) for r in metrics)
        for variant in VARIANTS:
            result = json.loads((ROOT/f"seed{seed}_{variant}/result.json").read_text())
            selected = [r for r in metrics if r["variant"]==variant]
            loss = sum(float(r["discarded_l1"]) for r in selected)
            lower = sum(float(r["independent_lower_l1"]) for r in selected)
            base = sum(float(r["contiguous_l1"]) for r in selected)
            from fixed_quota_grouping import closure
            proxy[seed,variant] = dict(discarded_l1=loss,gap_closure=closure(base,loss,lower))
            results.append(dict(**result,**proxy[seed,variant],gain_vs_contiguous_pp=score(seed,variant)-score(seed,"contiguous")))
    common.write_csv(ROOT/"validation_results.csv",results)
    common.write_csv(ROOT/"layer_metrics_combined.csv",layer_rows)
    summary = []
    metrics = ["ft0_accuracy","ft0_bn_clone_accuracy","ft1_accuracy","ft2_accuracy","ft10_accuracy"]
    for variant in VARIANTS:
        selected = [r for r in results if r["variant"]==variant]
        gains = [r["gain_vs_contiguous_pp"] for r in selected]
        row = dict(variant=variant,seeds=len(seeds),gain_pp=statistics.mean(gains),
            paired_std_pp=statistics.stdev(gains) if len(gains)>1 else "",
            ft10_std_pp=statistics.stdev(r["ft10_accuracy"]*100 for r in selected) if len(selected)>1 else "",
            discarded_l1_mean=statistics.mean(r["discarded_l1"] for r in selected),
            gap_closure_mean=statistics.mean(r["gap_closure"] for r in selected if r["gap_closure"] is not None),
            training_seconds_mean=statistics.mean(r["training_seconds"] for r in selected))
        row.update({m.replace("accuracy","pct"):statistics.mean(r[m]*100 for r in selected) for m in metrics})
        summary.append(row)
    common.write_csv(ROOT/"paired_summary.csv",summary)
    random_rows = []
    for seed in seeds:
        values = [score(seed,f"random{i}") for i in range(3)]
        random_rows.append(dict(seed=seed,ft10_mean_pct=statistics.mean(values),initialization_std_pp=statistics.stdev(values),
            gain_vs_contiguous_pp=statistics.mean(values)-score(seed,"contiguous")))
    common.write_csv(ROOT/"random_initialization_summary.csv",random_rows)
    colors = ["#555c63","#a0a7ae","#89949c","#71818c","#c28e42","#3b938c","#bc5f72"]
    labels = ["Contiguous","Random 0","Random 1","Random 2","L1 sorted","Jaccard","Jaccard + swap"]
    fig,axes = plt.subplots(1,3,figsize=(16,4.8))
    x = list(range(len(summary)))
    axes[0].bar(x,[r["ft10_pct"] for r in summary],color=colors,yerr=[r["ft10_std_pp"] or 0 for r in summary],capsize=3)
    axes[0].set_ylim(min(r["ft10_pct"] for r in summary)-.6,max(r["ft10_pct"] for r in summary)+.4)
    axes[0].set_ylabel("FT10 validation accuracy (%)")
    axes[1].bar(x,[r["gap_closure_mean"]*100 for r in summary],color=colors)
    axes[1].set_ylabel("L1 gap closed (%)")
    for row,color,label in zip(summary,colors,labels):
        axes[2].plot([0,1,2,10],[row[k] for k in ["ft0_pct","ft1_pct","ft2_pct","ft10_pct"]],marker="o",color=color,label=label)
    axes[2].set_xlabel("Fine-tuning epochs")
    axes[2].set_ylabel("Raw validation accuracy (%)")
    axes[2].legend(fontsize=8)
    for ax in axes[:2]:
        ax.set_xticks(x,labels,rotation=50,ha="right")
    fig.suptitle(f"ResNet50 / CIFAR-10 / fixed layer quotas / Tm8 / {len(seeds)} Dense seed(s)")
    fig.tight_layout()
    fig.savefig(ROOT/"comparison.png",dpi=170)
    plt.close(fig)
    lines = ["# 固定剪枝量的 Filter 重新分組實驗", "",f"完成時間：{common.stamp()}","",
        "## 設定","","ResNet50/CIFAR-10、Tm8、l1_sum、一般卷積含1x1。每seed由global 90% reference取得layer ratio，再固定該層所有cluster的整數q。所有方法同seed的weight/MAC budget相同；不是舊global_cluster配額，不能將舊accuracy當本輪控制。無保護、補償、量化或RTL變更。只評估5000張validation，沒有跑test。","",
        "每個方法都以原始Dense checkpoint初始化；FT10相同SGD/AMP/訓練RNG。FT0 BN欄是512張train images校正的獨立副本，不帶入主訓練。代理代價與分類loss不同；缺口閉合率的下界僅指固定每filter q的獨立top-q。","",
        "## 結果","","| 方法 | Seeds | FT0 (%) | FT0 BN副本 (%) | FT2 (%) | FT10 (%) | 對連續分組提升 (pp) | 配對std (pp) | L1缺口閉合 (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        std = f"{r['paired_std_pp']:.3f}" if r["paired_std_pp"]!="" else "N/A"
        lines.append(f"| {r['variant']} | {r['seeds']} | {r['ft0_pct']:.3f} | {r['ft0_bn_clone_pct']:.3f} | {r['ft2_pct']:.3f} | {r['ft10_pct']:.3f} | {r['gain_pp']:+.3f} | {std} | {r['gap_closure_mean']*100:.2f} |")
    lines += ["","![比較圖](comparison.png)","","## 決策與界線","",decision["reason"],"",
        f"seed0預先指定擴展門檻：swap-contiguous >=0.3pp，且swap不低於Jaccard及L1排序。實際seed0差值為{decision['seed0_gain_pp']:+.3f}pp。此為投入門檻，不是顯著性檢定。",
        "","3個random初始化不等於3個獨立Dense seeds；分開見random_initialization_summary.csv。單seed不可推論普遍優勢；即使多seed為正，也未經獨立test確認，且本validation曾用於其他方法開發。",
        "","## 資料導覽","","- validation_results.csv：各分支FT0/1/2/10、實際稀疏度、權重/MAC預算、時間與GPU記憶體。",
        "- paired_summary.csv：相同Dense seed配對的平均與sample std。",
        "- layer_metrics_combined.csv：每層q、代理代價、下界與缺口閉合率。",
        "- cluster_metadata.csv：每組filter IDs與kept_count。",
        "- prepared/：原始groups、共享masks、搜尋歷程與BN train indices。",
        "- seed*/：初始診斷、各epoch紀錄、最終模型及完整resume狀態。",
        "- source_snapshot/、sources.json、VALIDATION.json、UNIT_TESTS.txt：計畫、來源與驗證證據。",
        "","## 硬體界線","","本輪沒有cycle-accurate OS模擬，沒有宣稱非連續filter ID的查表/寫回零成本，也沒有宣稱真實speedup。不同cluster順序執行不直接等於負載不平衡浪費。",
        "","## 執行紀錄","","GPU原有負載約97-99%，每卡只新增一個worker，不動既有工作。時間包含共享GPU競爭，不能視為獨占benchmark。詳見EXECUTION_LOG.md、execution.jsonl與logs/。"]
    (ROOT/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def package():
    include = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.name.endswith((".lock",".tmp")):
            continue
        if path.suffix==".pt" and ("prepared" not in path.parts or path.name=="bn_images.pt"):
            continue
        include.append(path)
    common.write_json(ROOT/"MANIFEST.json",dict(files=[dict(path=str(p.relative_to(ROOT)),sha256=common.sha(p),bytes=p.stat().st_size) for p in include],
        exclusions="Large model/training checkpoints and BN images remain locally; grouped masks and indices included"))
    destination = ROOT.with_suffix(".zip")
    with zipfile.ZipFile(destination,"w",zipfile.ZIP_DEFLATED) as archive:
        for path in [*include,ROOT/"MANIFEST.json"]:
            archive.write(path,str(path.relative_to(ROOT.parent)))
    destination.with_suffix(".zip.sha256").write_text(common.sha(destination)+"  "+destination.name+"\n")


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/"logs").mkdir(exist_ok=True)
    lock = (ROOT/"study.lock").open("w")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    freeze_sources()
    with (ROOT/"UNIT_TESTS.txt").open("w") as stream:
        test = subprocess.run([sys.executable,"-m","unittest","discover","-s","tests","-v"],cwd=PROJECT,stdout=stream,stderr=subprocess.STDOUT)
    if test.returncode:
        raise RuntimeError("Unit tests failed")
    event("study_start_or_resume",plan_sha256=common.sha(PLAN))
    run_batch([dict(mode="prepare",seed=0)])
    run_batch([dict(mode="train",seed=0,variant=v) for v in VARIANTS])
    gain = score(0,"jaccard_swap")-score(0,"contiguous")
    promote = gain >= .3-1e-12 and score(0,"jaccard_swap") >= max(score(0,"jaccard"),score(0,"norm_sorted"))-1e-12
    decision = dict(expanded=promote,seed0_gain_pp=gain,
        reason="交換搜尋版達預先擴展條件，完成三Dense seeds確認。" if promote else "交換搜尋版未達預先擴展條件，完成seed0可行性檢查後停止，不調參追高。",test_evaluated=False)
    common.write_json(ROOT/"decision.json",decision)
    event("promotion_decision",**decision)
    seeds = [0,1,2] if promote else [0]
    if promote:
        run_batch([dict(mode="prepare",seed=s) for s in [1,2]])
        run_batch([dict(mode="train",seed=s,variant=v) for s in [1,2] for v in VARIANTS])
    freeze_sources()
    verify(seeds)
    report(seeds,decision)
    entries = [json.loads(line) for line in (ROOT/"execution.jsonl").read_text().splitlines()]
    lines = ["# 執行與問題紀錄","",f"完成{len(seeds)*len(VARIANTS)}個FT10分支，準備{len(seeds)}個Dense seeds。",
        "","- 啟動前GPU皆忙碌；限制每卡一新worker，沒有終止既有程序。",
        "- 新增獨立程式與資料夾，未修改核心剪枝或RTL。",
        "- 任意filter分組必須依原filter ID散佈full mask；以toy及真實模型獨立審核，禁止直接按cluster順序展開。",
        "- 固定layer q會改變舊global_cluster配置，因此本輪重新訓練自己的連續控制。",
        "- 若出現錯誤，保留原log與epoch resume狀態，不用validation修改搜尋規則。","",
        "## 事件","",*[f"- {e['time']} {e['kind']} {json.dumps({k:v for k,v in e.items() if k not in ['time','kind']},ensure_ascii=False)}" for e in entries]]
    (ROOT/"EXECUTION_LOG.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    common.write_json(ROOT/"status.json",dict(state="complete",completed_at=common.stamp(),seeds=seeds,branches=len(seeds)*len(VARIANTS),test_evaluated=False))
    event("study_complete",seeds=seeds)
    package()


if __name__ == "__main__":
    main()
