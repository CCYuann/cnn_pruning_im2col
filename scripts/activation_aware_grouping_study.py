#!/usr/bin/env python3
"""Two activation-aware branches, paired with frozen L1 fixed-quota controls."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import torch

import fixed_quota_grouping_worker as base
from activation_patch_scores import activation_scores, explicit_sample_moments, patch_moments, rms_from_moments
from fixed_quota_grouping import closure, discarded, exchange_search, shared_masks, validate_groups

common=base.common
core=base.core
PROJECT=base.PROJECT
ROOT=PROJECT/"runs/activation_aware_fixed_quota_grouping"
OLD=PROJECT/"runs/fixed_quota_filter_regrouping"
PLAN=PROJECT/"research_notes/2026-11-02_activation_aware_grouping_plan.md"
VARIANTS=["contiguous","jaccard_swap"]


def event(kind, **details):
    value=dict(time=common.stamp(),kind=kind,**details)
    with (ROOT/"execution.jsonl").open("a",encoding="utf-8") as stream:
        stream.write(json.dumps(value,ensure_ascii=False)+"\n")
    print(json.dumps(value,ensure_ascii=False),flush=True)


def sources():
    old_paths=[OLD/"prepared/seed0"/name for name in ["ready.json","configuration.json","contiguous.pt","jaccard.pt","jaccard_swap.pt","bn_images.pt","data_indices.json"]]
    old_paths += [OLD/f"seed0_{v}"/name for v in VARIANTS for name in ["result.json","configuration.json","training_history.csv","initial_validation.json"]]
    old_paths += [OLD/name for name in ["VALIDATION.json","BUDGET_VALIDATION.json"]]
    paths={PROJECT/name for name in base.fingerprint(0)["source_sha256"]}
    paths.update([Path(__file__).resolve(),PLAN,PROJECT/"scripts/activation_patch_scores.py",PROJECT/"tests/test_activation_patch_scores.py"])
    return paths,old_paths


def fingerprint():
    paths,old_paths=sources()
    return dict(model="resnet50",seed=0,split_id="42fbb7440cc27c71",dense_checkpoint_sha256=common.sha(common.checkpoint("resnet50",0)),
        source_sha256={str(p.relative_to(PROJECT)):common.sha(p) for p in sorted(paths)},
        reused_sha256={str(p.relative_to(PROJECT)):common.sha(p) for p in old_paths},
        score="abs(weight)*per_patch_position_RMS",calibration_images=512,calibration_source="same train-only images as original BN diagnostic",
        activation_forward="FP32 eval, no TF32",moment_accumulation="FP64, includes zero padding",search_rounds=4,
        quota="unchanged original per-layer uniform q",tm=8,ft_epochs=10,test_evaluated=False)


def freeze():
    ROOT.mkdir(parents=True,exist_ok=True)
    config=fingerprint()
    base.freeze(ROOT/"study_configuration.json",config)
    return config


def validate_old():
    ready=json.loads((OLD/"prepared/seed0/ready.json").read_text())
    if json.loads((OLD/"prepared/seed0/configuration.json").read_text()) != base.fingerprint(0):
        raise ValueError("Reused training implementation or checkpoint changed")
    for name in ["contiguous.pt","jaccard.pt","jaccard_swap.pt","bn_images.pt","data_indices.json"]:
        if common.sha(OLD/"prepared/seed0"/name)!=ready["files_sha256"][name]:
            raise ValueError("Reused prepared input changed")
    for name in ["VALIDATION.json","BUDGET_VALIDATION.json"]:
        if not json.loads((OLD/name).read_text())["passed"]:
            raise ValueError("Old controls failed verification")
    for variant in VARIANTS:
        result=json.loads((OLD/f"seed0_{variant}/result.json").read_text())
        if result["epochs"]!=10 or result["test_evaluated"] or result["split_id"]!="42fbb7440cc27c71":
            raise ValueError("Invalid reused training control")
    return ready


@torch.no_grad()
def prepare(device):
    freeze()
    old_ready=validate_old()
    cache=ROOT/"prepared/seed0"
    cache.mkdir(parents=True,exist_ok=True)
    base.freeze(cache/"configuration.json",base.fingerprint(0))
    if (cache/"ready.json").exists():
        return
    started=time.monotonic()
    model,specs,payload=base.load_model(0,device)
    images=torch.load(OLD/"prepared/seed0/bn_images.pt",map_location="cpu",weights_only=False)
    indices=json.loads((OLD/"prepared/seed0/data_indices.json").read_text())["bn_train_indices"]
    validation=set(torch.randperm(50000,generator=torch.Generator().manual_seed(0))[:5000].tolist())
    if len(images)!=512 or len(set(indices))!=512 or set(indices)&validation:
        raise ValueError("Calibration is not 512 distinct train-only images")
    moments={s.name:dict(squared_sum=None,count=0,images=0) for s in specs}
    diagnostics=[]
    chosen={specs[0].name,next(s.name for s in specs if s.module.kernel_size==(1,1)),
            next(s.name for s in specs if s.module.stride==(2,2))}
    handles=[]
    for spec in specs:
        def hook(module,inputs,output,name=spec.name):
            total,count=patch_moments(inputs[0],module)
            if count!=len(inputs[0])*output.shape[2]*output.shape[3]:
                raise ValueError("Unfold/output position mismatch")
            record=moments[name]
            if name in chosen and record["images"]==0:
                actual,ca=patch_moments(inputs[0][:1],module)
                expected,cb=explicit_sample_moments(inputs[0][:1],module)
                if ca!=cb:
                    raise ValueError("Independent patch count mismatch")
                torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-8)
                diagnostics.append(dict(layer=name,independent_patch_check=True,max_abs_difference=float((actual-expected).abs().max())))
            record["squared_sum"]=total if record["squared_sum"] is None else record["squared_sum"]+total
            record["count"]+=count
            record["images"]+=len(inputs[0])
        handles.append(spec.module.register_forward_hook(hook))
    model.eval()
    try:
        for index,batch in enumerate(images.split(32)):
            model(batch.to(device))
            if index in (0,7,15):
                event("activation_batch",images=min((index+1)*32,512))
    finally:
        for handle in handles:
            handle.remove()
    for name,value in model.state_dict().items():
        if not torch.equal(value.cpu(),payload["model_state_dict"][name]):
            raise ValueError("Collecting activations changed Dense state: "+name)
    stats_seconds=time.monotonic()-started
    rms_records={}
    for spec in specs:
        record=moments[spec.name]
        if record["images"]!=512:
            raise ValueError("Layer did not observe all calibration images")
        rms_records[spec.name]=dict(squared_sum=record["squared_sum"].cpu(),count=record["count"],images=512,
            rms=rms_from_moments(record["squared_sum"],record["count"]).cpu())
    torch.save(rms_records,cache/"activation_statistics.pt")
    contiguous=torch.load(OLD/"prepared/seed0/contiguous.pt",map_location="cpu",weights_only=False)
    initial=torch.load(OLD/"prepared/seed0/jaccard.pt",map_location="cpu",weights_only=False)
    old_optimized=torch.load(OLD/"prepared/seed0/jaccard_swap.pt",map_location="cpu",weights_only=False)
    packs={v:{} for v in VARIANTS}
    rows,searches=[],[]
    for spec in specs:
        name=spec.name
        weights=spec.module.weight.detach().flatten(1)
        scores=activation_scores(weights,rms_records[name]["rms"])
        q=contiguous[name]["q"]
        digest=int.from_bytes(hashlib.sha256(name.encode()).digest()[:4],"little")
        groups,search=exchange_search(scores,initial[name]["groups"].to(device),q,20261110+digest)
        searches.append(dict(layer=name,**search))
        generated={"contiguous":contiguous[name]["groups"].to(device),"jaccard_swap":groups}
        for variant in VARIANTS:
            mask,_=shared_masks(scores,generated[variant],q)
            packs[variant][name]=dict(contiguous[name],groups=generated[variant].cpu(),masks=mask.cpu())
        for method,pack in [("l1_contiguous",contiguous),("l1_optimized",old_optimized),
                            ("activation_contiguous",packs["contiguous"]),("activation_optimized",packs["jaccard_swap"])]:
            rec=pack[name]
            full=torch.zeros_like(weights,dtype=torch.bool)
            full[rec["groups"].to(device)]=rec["masks"].to(device)[:,None,:].expand(-1,8,-1)
            for metric,values in [("l1",weights.double().abs()),("activation",scores)]:
                cost=float((values*~full).sum())
                lower=float(discarded(values,q).sum())
                rows.append(dict(layer=name,method=method,metric=metric,discarded_cost=cost,independent_lower=lower,
                    q=q,k=weights.shape[1],filters=weights.shape[0],rms_min=float(rms_records[name]["rms"].min()),
                    rms_max=float(rms_records[name]["rms"].max()),rms_zero_positions=int((rms_records[name]["rms"]==0).sum())))
        event("layer_grouped",layer=name,q=q,swaps=search["accepted"])
    for variant in VARIANTS:
        torch.save(packs[variant],cache/f"{variant}.pt")
    shutil.copy2(OLD/"prepared/seed0/bn_images.pt",cache/"bn_images.pt")
    shutil.copy2(OLD/"prepared/seed0/data_indices.json",cache/"data_indices.json")
    common.write_csv(cache/"layer_metrics.csv",rows)
    common.write_json(cache/"search_history.json",searches)
    common.write_json(cache/"activation_audit.json",dict(dense_unchanged=True,train_images=512,test_evaluated=False,
        independent_patch_checks=diagnostics,stats_seconds=stats_seconds))
    ready={key:old_ready[key] for key in ["total_weights","kept_weights","sparsity","dense_macs","kept_macs","dense_training_config"]}
    files=["contiguous.pt","jaccard_swap.pt","activation_statistics.pt","bn_images.pt","data_indices.json","layer_metrics.csv","search_history.json","activation_audit.json"]
    ready.update(seed=0,variants=VARIANTS,seconds=time.monotonic()-started,stats_seconds=stats_seconds,
        study_configuration_sha256=common.sha(ROOT/"study_configuration.json"),test_evaluated=False,
        files_sha256={name:common.sha(cache/name) for name in files})
    common.write_json(cache/"ready.json",ready)
    event("prepare_complete",seconds=ready["seconds"],stats_seconds=stats_seconds)


def verify(final=False):
    freeze()
    old_ready=validate_old()
    cache=ROOT/"prepared/seed0"
    ready=json.loads((cache/"ready.json").read_text())
    if ready["study_configuration_sha256"]!=common.sha(ROOT/"study_configuration.json"):
        raise ValueError("Prepared study fingerprint mismatch")
    for name,digest in ready["files_sha256"].items():
        if common.sha(cache/name)!=digest:
            raise ValueError("Prepared file changed: "+name)
    model,specs,_=base.load_model(0,torch.device("cpu"))
    reference=core.global_unstructured_reference(specs,8,.9)
    rms=torch.load(cache/"activation_statistics.pt",map_location="cpu",weights_only=False)
    old_pack=torch.load(OLD/"prepared/seed0/contiguous.pt",map_location="cpu",weights_only=False)
    initial=torch.load(OLD/"prepared/seed0/jaccard.pt",map_location="cpu",weights_only=False)
    checks=[]
    for variant in VARIANTS:
        pack=torch.load(cache/f"{variant}.pt",map_location="cpu",weights_only=False)
        kept,macs=0,0
        final_state=torch.load(ROOT/f"seed0_{variant}/final_model.pt",map_location="cpu",weights_only=False)["model_state_dict"] if final else None
        for spec in specs:
            name=spec.name
            rec=pack[name]
            weights=spec.module.weight.detach().flatten(1)
            q=weights.shape[1]-round(reference[name]["layer_ratio"]*weights.shape[1])
            if q!=rec["q"] or q!=old_pack[name]["q"] or not bool((rec["masks"].sum(1)==q).all()):
                raise ValueError("Quota mismatch")
            validate_groups(rec["groups"],spec.module.out_channels)
            if variant=="contiguous" and not torch.equal(rec["groups"],old_pack[name]["groups"]):
                raise ValueError("Contiguous groups changed")
            stat=rms[name]
            expected_count=512*math.prod(core.output_hw(spec.input_hw,spec.module))
            if stat["images"]!=512 or stat["count"]!=expected_count:
                raise ValueError("Activation sample count mismatch")
            torch.testing.assert_close(stat["rms"],rms_from_moments(stat["squared_sum"],stat["count"]),rtol=1e-12,atol=1e-14)
            scores=activation_scores(weights,stat["rms"])
            expected_masks,full=shared_masks(scores,rec["groups"],q)
            if not torch.equal(expected_masks,rec["masks"]):
                raise ValueError("Mask does not match activation score")
            if variant=="jaccard_swap":
                start_masks,start_full=shared_masks(scores,initial[name]["groups"],q)
                if float((scores*~full).sum())>float((scores*~start_full).sum())+1e-7:
                    raise ValueError("Swap worsened activation objective")
            if final and torch.count_nonzero(final_state[name+".weight"].flatten(1)[~full]):
                raise ValueError("Forbidden weight regrowth")
            kept+=int(full.sum())
            macs+=int(full.sum())*math.prod(core.output_hw(spec.input_hw,spec.module))
        if kept!=old_ready["kept_weights"] or macs!=old_ready["kept_macs"]:
            raise ValueError("Actual budget differs from controls")
        if final:
            result=json.loads((ROOT/f"seed0_{variant}/result.json").read_text())
            if result["kept_weights"]!=kept or result["kept_macs"]!=macs or result["epochs"]!=10 or result["test_evaluated"]:
                raise ValueError("Training result metadata mismatch")
        checks.append(dict(variant=variant,layers=len(specs),kept_weights=kept,kept_macs=macs,mask_score_verified=True,final_no_regrowth=final))
    common.write_json(ROOT/("VALIDATION.json" if final else "PRETRAIN_VALIDATION.json"),dict(passed=True,checks=checks,test_evaluated=False))


def run_batch(jobs):
    pending=list(jobs)
    active={}
    try:
        while pending or active:
            for gpu,(job,process,log) in list(active.items()):
                code=process.poll()
                if code is None:
                    continue
                log.close()
                del active[gpu]
                if code:
                    event("job_failed",job=job,gpu=gpu,exit_code=code)
                    raise RuntimeError(f"Worker failed: {job}, exit={code}")
                event("job_complete",job=job,gpu=gpu)
            for gpu in [1,0]:
                if gpu in active or not pending:
                    continue
                free=int(subprocess.check_output(["nvidia-smi","-i",str(gpu),"--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
                if free<12000:
                    continue
                job=pending.pop(0)
                expected=ROOT/"prepared/seed0/ready.json" if job["mode"]=="prepare" else ROOT/f"seed0_{job['variant']}/result.json"
                if expected.exists():
                    event("reuse_completed",job=job)
                    continue
                label=job["mode"]+"_"+job.get("variant","all")
                log=(ROOT/"logs"/(label+".log")).open("a",buffering=1)
                command=[sys.executable,"-u",str(Path(__file__).resolve()),job["mode"]]
                if "variant" in job:
                    command += ["--variant",job["variant"]]
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4")
                process=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[gpu]=(job,process,log)
                event("job_start",job=job,gpu=gpu,pid=process.pid)
            common.write_json(ROOT/"status.json",dict(state="running",updated_at=common.stamp(),pending=pending,
                active=[dict(gpu=g,pid=p.pid,job=j) for g,(j,p,_) in active.items()]))
            if active or pending:
                time.sleep(15)
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


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cache=ROOT/"prepared/seed0"
    metrics=common.read_csv(cache/"layer_metrics.csv")
    cells=[("l1_contiguous",OLD,"contiguous"),("l1_optimized",OLD,"jaccard_swap"),
           ("activation_contiguous",ROOT,"contiguous"),("activation_optimized",ROOT,"jaccard_swap")]
    rows=[]
    for method,root,variant in cells:
        source=root/f"seed0_{variant}"
        result=json.loads((source/"result.json").read_text())
        row=dict(method=method,reused=root==OLD,source=str(source.relative_to(PROJECT)),
            **{key:result[key] for key in ["ft0_accuracy","ft0_bn_clone_accuracy","ft1_accuracy","ft2_accuracy","ft10_accuracy","training_seconds","sparsity","kept_weights","kept_macs","peak_gpu_bytes"]})
        for metric in ["l1","activation"]:
            selected=[r for r in metrics if r["method"]==method and r["metric"]==metric]
            cost=sum(float(r["discarded_cost"]) for r in selected)
            lower=sum(float(r["independent_lower"]) for r in selected)
            baseline=sum(float(r["discarded_cost"]) for r in metrics if r["method"]=="l1_contiguous" and r["metric"]==metric)
            row[metric+"_discarded_cost"]=cost
            row[metric+"_closure_vs_l1_contiguous"]=closure(baseline,cost,lower)
        rows.append(row)
        target=ROOT/"reused_controls"/method
        if root==OLD:
            target.mkdir(parents=True,exist_ok=True)
            for name in ["result.json","configuration.json","training_history.csv","initial_validation.json"]:
                shutil.copy2(source/name,target/name)
            shutil.copy2(OLD/"prepared/seed0"/f"{variant}.pt",target/"groups_masks.pt")
    common.write_csv(ROOT/"four_cell_results.csv",rows)
    acc={r["method"]:r["ft10_accuracy"]*100 for r in rows}
    effects=dict(score_effect_contiguous_pp=acc["activation_contiguous"]-acc["l1_contiguous"],
        grouping_effect_l1_pp=acc["l1_optimized"]-acc["l1_contiguous"],
        grouping_effect_activation_pp=acc["activation_optimized"]-acc["activation_contiguous"],
        optimized_pipeline_difference_pp=acc["activation_optimized"]-acc["l1_optimized"],
        interaction_pp=(acc["activation_optimized"]-acc["activation_contiguous"])-(acc["l1_optimized"]-acc["l1_contiguous"]),
        independent_dense_seeds=1,test_evaluated=False)
    common.write_json(ROOT/"effects.json",effects)
    fig,axes=plt.subplots(1,3,figsize=(15,4.6))
    colors=["#64737c","#bc6278"]
    for i,(score,methods) in enumerate([("L1",["l1_contiguous","l1_optimized"]),("Activation",["activation_contiguous","activation_optimized"])]):
        axes[0].bar([x+(i-.5)*.34 for x in [0,1]],[acc[m] for m in methods],width=.34,color=colors[i],label=score)
        for x,m in enumerate(methods):
            axes[0].text(x+(i-.5)*.34,acc[m]+.025,f"{acc[m]:.2f}",ha="center",fontsize=9)
    axes[0].set_xticks([0,1],["Contiguous","Optimized"])
    axes[0].set_ylabel("FT10 validation accuracy (%)")
    axes[0].set_ylim(min(acc.values())-.45,max(acc.values())+.2)
    axes[0].legend()
    labels=["L1 C","L1 O","Act C","Act O"]
    for i,metric in enumerate(["l1","activation"],1):
        values=[r[metric+"_closure_vs_l1_contiguous"] for r in rows]
        axes[i].bar(range(4),[v*100 if v is not None else math.nan for v in values],color=[colors[0],colors[0],colors[1],colors[1]])
        axes[i].set_xticks(range(4),labels)
        axes[i].set_ylabel(metric+" proxy gap closed (%)")
        axes[i].axhline(0,color="#555555",linewidth=.7)
    fig.suptitle("ResNet50 / CIFAR-10 / fixed quotas / Tm8 / Dense seed0")
    fig.tight_layout()
    fig.savefig(ROOT/"comparison.png",dpi=170)
    plt.close(fig)
    ready=json.loads((cache/"ready.json").read_text())
    lines=["# Activation-aware固定配額分組pilot","",f"完成：{common.stamp()}","",
        "## 設定","","ResNet50/CIFAR-10 seed0、Tm8、同Dense checkpoint、相同逐層q。實際卷積權重稀疏度89.9987%，保留2,345,024 weights及233,504,768 MAC/image。新增兩個FT10分支，重用兩個已完成L1控制。沒有保護、補償、量化或RTL變更，不跑test。",
        "","使用Dense eval FP32 forward，在與上輪BN診斷相同的512張train影像上，以每個(c,r,s)的unfold patch RMS建立S=abs(W)*RMS；納入padding/stride，FP64累計。分組與mask皆用同一S。初始groups、4輪交換上限及RNG和L1最佳化相同。",
        "","直接重用未修改的fixed_quota_grouping_worker.train，以SGD lr=.001、batch128、AMP float16、10epochs及相同training RNG訓練。BN副本不帶入主FT。固定報第10輪，不挑最好epoch。",
        "","## 四格結果","","| 方法 | 重用 | FT0 (%) | BN副本 FT0 (%) | FT2 (%) | FT10 (%) |",
        "|---|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['method']} | {'是' if r['reused'] else '否'} | {r['ft0_accuracy']*100:.2f} | {r['ft0_bn_clone_accuracy']*100:.2f} | {r['ft2_accuracy']*100:.2f} | {r['ft10_accuracy']*100:.2f} |")
    lines += ["","![比較圖](comparison.png)","","## 拆解效果","",
        f"- 固定連續groups，只換score：{effects['score_effect_contiguous_pp']:+.3f}pp。",
        f"- L1下最佳化groups：{effects['grouping_effect_l1_pp']:+.3f}pp。",
        f"- Activation score下最佳化groups：{effects['grouping_effect_activation_pp']:+.3f}pp。",
        f"- 兩個最佳化pipeline之差：{effects['optimized_pipeline_difference_pp']:+.3f}pp；此比較同時改變score及選出的groups，不是固定groups只換mask。",
        f"- 探索性interaction：{effects['interaction_pp']:+.3f}pp。",
        "","只有一個Dense seed，沒有配對std或顯著性證據。validation亦用於其他開發，不當獨立test。activation RMS不是分類loss，也不完整考慮輸入相關性；代理改善不保證accuracy提升。本pilot完成後不自動擴seed或調參。",
        "","## 成本與驗證","",f"統計收集{ready['stats_seconds']:.2f}秒，完整準備{ready['seconds']:.2f}秒；時間為當下共享GPU環境，不能作獨占speedup。FT seconds只計訓練路徑時間，不包含全部研究開銷。",
        "","PRETRAIN_VALIDATION.json/VALIDATION.json檢查source、校正樣本count、reference q、原filter ID mask scatter、score與mask相符、交換目標不增、weight/MAC budget及禁止位置無regrowth。UNIT_TESTS.txt包含padding/stride/dilation與optimizer/resume測試。",
        "","## 資料入口","","- four_cell_results.csv、effects.json：原始四格結果與效果拆解。",
        "- prepared/seed0/activation_statistics.pt：每層sum/count/RMS。",
        "- prepared/seed0/layer_metrics.csv：四格都以同一L1與activation proxy計算，不直接比較不同單位。",
        "- prepared/seed0/search_history.json、groups/masks：固定搜尋與分組資料。",
        "- reused_controls/：原L1控制的報表、設定、歷史及mask副本。",
        "- seed0_*/：新增兩分支完整訓練狀態及最終模型，僅保留本機。",
        "- study_configuration.json/source_snapshot/：凍結計畫、來源與舊結果checksum。",
        "- EXECUTION_LOG.md/execution.jsonl/logs/：執行與問題紀錄。",
        "","## 硬體界線","","本輪只評估軟體精度；無cycle-accurate模擬、INT8或實際存取/功耗收益證據。任意filter ID的硬體查表與寫回成本仍需另驗證。"]
    (ROOT/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def package():
    files=[]
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.name.endswith((".tmp",".lock")) or path.name=="MANIFEST.json":
            continue
        if path.suffix==".pt" and (path.name=="bn_images.pt" or path.name in ["training_state.pt","final_model.pt"]):
            continue
        files.append(path)
    common.write_json(ROOT/"MANIFEST.json",dict(files=[dict(path=str(p.relative_to(ROOT)),sha256=common.sha(p),bytes=p.stat().st_size) for p in files],
        exclusions="Large training/final checkpoints and BN images remain locally"))
    archive_path=ROOT.with_suffix(".zip")
    with zipfile.ZipFile(archive_path,"w",zipfile.ZIP_DEFLATED) as archive:
        for path in [*files,ROOT/"MANIFEST.json"]:
            archive.write(path,str(path.relative_to(ROOT.parent)))
    archive_path.with_suffix(".zip.sha256").write_text(common.sha(archive_path)+"  "+archive_path.name+"\n")


def run():
    freeze()
    (ROOT/"logs").mkdir(exist_ok=True)
    lock=(ROOT/"study.lock").open("w")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    paths,_=sources()
    for source in paths:
        target=ROOT/"source_snapshot"/source.relative_to(PROJECT)
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            shutil.copy2(source,target)
        elif common.sha(target)!=common.sha(source):
            raise ValueError("Source snapshot changed")
    with (ROOT/"UNIT_TESTS.txt").open("w") as stream:
        test=subprocess.run([sys.executable,"-m","unittest","discover","-s","tests","-v"],cwd=PROJECT,stdout=stream,stderr=subprocess.STDOUT)
    if test.returncode:
        raise RuntimeError("Unit tests failed")
    event("study_start_or_resume")
    run_batch([dict(mode="prepare")])
    verify()
    run_batch([dict(mode="train",variant=v) for v in VARIANTS])
    verify(final=True)
    report()
    common.write_json(ROOT/"status.json",dict(state="complete",completed_at=common.stamp(),new_branches=2,reused_branches=2,dense_seeds=[0],test_evaluated=False))
    event("study_complete")
    entries=[json.loads(line) for line in (ROOT/"execution.jsonl").read_text().splitlines()]
    lines=["# 執行與問題紀錄","","新增2個FT10分支，重用2個L1控制；不改舊實驗或RTL。",
        "","- 啟動前GPU1空閒、GPU0有既有負載；每卡最多新增一worker，未停止既有程序。",
        "- 訓練直接重用未改動的舊worker.train，新增wrapper另凍結activation方法/source及ready checksum。",
        "- 每(c,r,s) patch統計而非channel近似，納入padding、stride及dilation，以獨立切片核對真實層。",
        "- 若遇錯誤保留log與epoch resume，不能看validation改搜尋或配額。","","## 事件","",
        *[f"- {e['time']} {e['kind']} {json.dumps({k:v for k,v in e.items() if k not in ['time','kind']},ensure_ascii=False)}" for e in entries]]
    (ROOT/"EXECUTION_LOG.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    package()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["run","prepare","train","verify"])
    parser.add_argument("--variant",choices=VARIANTS,default="contiguous")
    args=parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    core.reset_training_rng(0)
    if args.mode=="run":
        run()
    elif args.mode=="verify":
        verify(final=True)
    else:
        freeze()
        if not torch.cuda.is_available():
            raise RuntimeError("GPU required")
        device=torch.device("cuda")
        if args.mode=="prepare":
            prepare(device)
        else:
            args.seed,args.root=0,ROOT
            base.train(args,device)


if __name__ == "__main__":
    main()
