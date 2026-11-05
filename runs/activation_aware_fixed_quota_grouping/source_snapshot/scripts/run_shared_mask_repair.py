#!/usr/bin/env python3
"""Diagnose shared masks and run validation-only pilots without altering the core."""

import argparse
import hashlib
import inspect
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import resnet34_pruning_experiments as core
import run_fair_criterion_r0p9 as common
from shared_mask_repair import cluster_inputs, errors, metrics, normalized_mask, project, random_swaps, repair, sample_patches


VARIANTS = ["l1", "normalized", "mean_repair", "worst_repair", "random"]
PROJECT = common.PROJECT


def event(root, kind, **data):
    record = dict(time=common.stamp(), kind=kind, **data)
    with (root / "journal.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)


def calibration_loaders(args):
    transform = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(),
                                    transforms.Normalize((.4914,.4822,.4465),(.2470,.2435,.2616))])
    dataset = datasets.CIFAR10(PROJECT / "data", train=True, download=False, transform=transform)
    split = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(0))
    available = split[5000:]
    order = torch.randperm(len(available), generator=torch.Generator().manual_seed(args.calibration_seed))
    chosen = available[order[:args.images*2]].tolist()
    if len(chosen) != args.images*2:
        raise ValueError("Insufficient disjoint training calibration samples")
    loaders = [DataLoader(Subset(dataset, indices), batch_size=32, shuffle=False, num_workers=2)
               for indices in [chosen[:args.images],chosen[args.images:]]]
    return loaders, chosen


@torch.no_grad()
def collect_patches(model, specs, loaders, args, directory, device):
    fingerprint = {"checkpoint_sha256": common.sha(common.checkpoint(args.model,args.seed)),
                   "images_per_split": args.images, "patches_per_layer": args.patches,
                   "calibration_seed": args.calibration_seed,
                   "sampler_sha256": hashlib.sha256(inspect.getsource(sample_patches).encode()).hexdigest()}
    cache = directory / "patches.pt"
    manifest = directory / "patch_manifest.json"
    if cache.exists() and manifest.exists() and json.loads(manifest.read_text()) == fingerprint:
        event(args.root,"reuse_patch_cache",model=args.model)
        return torch.load(cache,map_location="cpu",weights_only=False)
    answer = []
    for split_index, loader in enumerate(loaders):
        buffers, counts, handles, generators = {}, {}, [], {}
        for spec in specs:
            layer = spec.name
            buffers[layer], counts[layer] = [], 0
            number = int.from_bytes(hashlib.sha256((layer+str(split_index)).encode()).digest()[:4],"little")
            generators[layer] = torch.Generator().manual_seed(args.calibration_seed+number)

            def hook(module, arguments, output, name=layer):
                if counts[name] >= args.patches:
                    return
                value = arguments[0]
                length = output.shape[-2]*output.shape[-1]
                per_image = min(length,max(1,math.ceil(args.patches/args.images)))
                positions = torch.randint(length,(len(value),per_image),generator=generators[name]).to(device)
                samples = sample_patches(value,module,positions)[:,:args.patches-counts[name]].detach().cpu()
                buffers[name].append(samples)
                counts[name] += samples.shape[1]
            handles.append(spec.module.register_forward_hook(hook))
        try:
            for image,_ in loader:
                model(image.to(device))
        finally:
            for handle in handles:
                handle.remove()
        answer.append({name:torch.cat(parts,dim=1) for name,parts in buffers.items()})
        event(args.root,"patch_split_collected",model=args.model,split=split_index,min_patches=min(counts.values()),max_patches=max(counts.values()))
    torch.save(answer,cache)
    common.write_json(manifest,fingerprint)
    return answer


def model_summary(rows):
    summary=[]
    for variant in sorted({r["variant"] for r in rows}):
        samples=[r for r in rows if r["variant"]==variant]
        record={"variant":variant,"clusters":len(samples),"swaps":sum(r["swaps"] for r in samples)}
        for split in ["search","check"]:
            active=[r for r in samples if r[split+"_valid_filters"]]
            n=sum(r[split+"_valid_filters"] for r in active)
            record[split+"_mean_error"]=sum(r[split+"_mean_error"]*r[split+"_valid_filters"] for r in active)/max(n,1)
            record[split+"_mean_worst_error"]=sum(r[split+"_max_error"] for r in active)/max(len(active),1)
            record[split+"_near_total_loss_filters"]=sum(r[split+"_near_total_loss_filters"] for r in samples)
        summary.append(record)
    baseline=next(r for r in summary if r["variant"]=="l1")
    for row in summary:
        row["check_worst_improvement_pct"]=(1-row["check_mean_worst_error"]/max(baseline["check_mean_worst_error"],1e-30))*100
        row["check_mean_increase_pct"]=(row["check_mean_error"]/max(baseline["check_mean_error"],1e-30)-1)*100
        row["passes_diagnostic_gate"] = row["check_worst_improvement_pct"]>=5 and row["check_mean_increase_pct"]<=1
    return summary


@torch.no_grad()
def diagnose(model,specs,args,directory,device):
    loaders,indices=calibration_loaders(args)
    common.write_json(directory/"calibration_indices.json",{"search_train_indices":indices[:args.images],"check_train_indices":indices[args.images:],"validation_excluded":True})
    patches=collect_patches(model,specs,loaders,args,directory,device)
    reference=core.global_unstructured_reference(specs,8,.9)
    masks={variant:{} for variant in VARIANTS}
    rows,swaps_log=[],[]
    started=time.monotonic()
    for layer_index,spec in enumerate(specs):
        conv=spec.module
        weight=conv.weight.detach().flatten(1)
        xx=[part[spec.name].to(device) for part in patches]
        dense_energy=project(weight,cluster_inputs(xx[0],conv,0,conv.out_channels)).square().mean(1)
        positive=dense_energy[dense_energy>0]
        epsilon=max((positive.median().item() if len(positive) else 0)*1e-6,1e-30)
        for variant in VARIANTS:
            masks[variant][spec.name]=[]
        for cluster_id,group in enumerate(reference[spec.name]["filter_groups"]):
            start,end=group[0],group[-1]+1
            ww=weight[start:end]
            inputs=[cluster_inputs(x,conv,start,end) for x in xx]
            k=ww.shape[1]
            remove=round(reference[spec.name]["cluster_ratios"][cluster_id]*k)
            mask=torch.ones(k,dtype=torch.bool,device=device)
            mask[torch.argsort(ww.abs().sum(0))[:remove]]=False
            worst,worst_swaps=repair(ww,inputs[0],mask,epsilon,"worst",args.candidates,args.iterations)
            mean,mean_swaps=repair(ww,inputs[0],mask,epsilon,"mean",args.candidates,args.iterations)
            variants={"l1":mask,"normalized":normalized_mask(ww,inputs[0],k-remove),"mean_repair":mean,
                      "worst_repair":worst,"random":random_swaps(mask,len(worst_swaps),args.calibration_seed+layer_index*10000+cluster_id)}
            for variant,selected in variants.items():
                if int(selected.sum()) != k-remove:
                    raise ValueError("Cluster kept-count changed")
                masks[variant][spec.name].append(selected.cpu())
                row={"model":args.model,"seed":args.seed,"layer":spec.name,"cluster":cluster_id,"variant":variant,
                     "tm":8,"groups":conv.groups,"kernel":str(conv.kernel_size),"K":k,"kept_count":k-remove,
                     "epsilon":epsilon,"swaps":len(worst_swaps) if variant in {"worst_repair","random"} else len(mean_swaps) if variant=="mean_repair" else 0}
                for label,x in zip(["search","check"],inputs):
                    row.update({label+"_"+key:value for key,value in metrics(ww,x,selected,epsilon).items()})
                rows.append(row)
            for variant,history in [("worst_repair",worst_swaps),("mean_repair",mean_swaps)]:
                for step,swap in enumerate(history):
                    swaps_log.append(dict(layer=spec.name,cluster=cluster_id,variant=variant,step=step,**swap))
        full_reference=reference[spec.name]["full_mask"]
        for cluster_id,start in enumerate(range(0,conv.out_channels,16)):
            end=min(start+16,conv.out_channels)
            ww=weight[start:end]
            k=ww.shape[1]
            ratio=1-float(full_reference[start:end].sum())/full_reference[start:end].numel()
            remove=round(ratio*k)
            mask=torch.ones(k,dtype=torch.bool,device=device)
            mask[torch.argsort(ww.abs().sum(0))[:remove]]=False
            row={"model":args.model,"seed":args.seed,"layer":spec.name,"cluster":cluster_id,"variant":"l1_tm16",
                 "tm":16,"groups":conv.groups,"kernel":str(conv.kernel_size),"K":k,"kept_count":k-remove,"epsilon":epsilon,"swaps":0}
            for label,x in zip(["search","check"],xx):
                row.update({label+"_"+key:value for key,value in metrics(ww,cluster_inputs(x,conv,start,end),mask,epsilon).items()})
            rows.append(row)
        event(args.root,"layer_finished",model=args.model,layer=spec.name,index=layer_index+1,total_layers=len(specs),seconds=round(time.monotonic()-started,1))
    for variant in masks:
        masks[variant]={name:torch.stack(parts) for name,parts in masks[variant].items()}
    torch.save(masks,directory/"masks.pt")
    common.write_csv(directory/"cluster_metrics.csv",rows)
    if swaps_log:
        common.write_csv(directory/"swaps.csv",swaps_log)
    summary=model_summary(rows)
    for row in summary:
        row.update(model=args.model,seed=args.seed)
    common.write_csv(directory/"summary.csv",summary)
    total=sum(spec.module.weight.numel() for spec in specs)
    kept=sum(int(masks["l1"][spec.name].sum())*8 for spec in specs)
    baseline=common.baselines()[(args.model,args.seed)]
    if not math.isclose(1-kept/total,float(baseline["pruning_ratio"]),abs_tol=1e-6,rel_tol=0):
        raise ValueError("Regenerated mask differs from formal l1 sparsity")
    common.write_json(directory/"done.json",{"completed_at":common.stamp(),"checkpoint_id":baseline["checkpoint_id"],
                      "total_weights":total,"kept_weights":kept,"actual_sparsity":1-kept/total,
                      "seconds":time.monotonic()-started,"test_evaluated":False})
    event(args.root,"diagnostic_complete",model=args.model,summary=summary)


def pilot(model,specs,args,directory,device):
    masks=torch.load(directory/"masks.pt",map_location="cpu",weights_only=False)[args.variant]
    train_and_eval,_,metadata=core.build_cifar10(PROJECT/"data",128,32,None,None,None,2,True,0,args.seed,5000)
    train_loader,val_loader,_unused_test_loader=train_and_eval
    stats=[]
    with torch.no_grad():
        for spec in specs:
            conv=spec.module
            saved=masks[spec.name]
            full=saved.repeat_interleave(8,0)[:conv.out_channels].reshape_as(conv.weight).to(device)
            conv.weight.mul_(full)
            layer_masks=[mask.reshape(conv.weight.shape[1:]) for mask in saved]
            out_hw=core.output_hw(spec.input_hw,conv)
            original=conv.weight.numel()
            stat=core.build_stats(spec.name,args.variant,spec,layer_masks,original,int(full.sum()),
                                  original*out_hw[0]*out_hw[1],core.macs_from_masks(layer_masks,conv.out_channels,out_hw,8),8,8)
            stat.full_mask=full
            stats.append(stat)
    _,direct=core.evaluate(model,val_loader,device,use_amp="float16")
    recipe=SimpleNamespace(finetune_lr=.001,weight_decay=.0005,finetune_epochs=10,tm=8)
    core.reset_training_rng(args.seed+100000,train_loader)
    core.fine_tune(model,stats,train_loader,val_loader,recipe,device,args.model+" "+args.variant,"float16")
    loss,accuracy=core.evaluate(model,val_loader,device,use_amp="float16")
    output=directory/"pilots"
    output.mkdir(exist_ok=True)
    torch.save({"model_state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"model":args.model,
                "seed":args.seed,"variant":args.variant,"split_id":metadata["split_id"],"test_evaluated":False},output/(args.variant+".pt"))
    common.write_json(output/(args.variant+".json"),{"model":args.model,"seed":args.seed,"variant":args.variant,
                      "direct_validation_acc":direct,"final_validation_acc":accuracy,"validation_loss":loss,
                      "finetune_epochs":10,"sparsity":core.summarize(stats)["pruning_ratio"],"test_evaluated":False})
    event(args.root,"pilot_complete",model=args.model,variant=args.variant,validation_accuracy=accuracy)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["diagnose","pilot"])
    parser.add_argument("--model",choices=common.MODELS,required=True)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--root",type=Path,default=PROJECT/"runs/shared_mask_repair_study/v1")
    parser.add_argument("--images",type=int,default=512)
    parser.add_argument("--patches",type=int,default=1024)
    parser.add_argument("--calibration-seed",type=int,default=20261101)
    parser.add_argument("--candidates",type=int,default=8)
    parser.add_argument("--iterations",type=int,default=8)
    parser.add_argument("--variant",choices=VARIANTS,default="worst_repair")
    args=parser.parse_args()
    args.root=args.root.resolve()
    directory=args.root/(args.model+"_seed"+str(args.seed))
    directory.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    core.reset_training_rng(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fingerprint={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in {"mode","variant"}}
    fingerprint.update(core_sha256=common.sha(common.CORE),repair_sha256=common.sha(Path(__file__).with_name("shared_mask_repair.py")))
    path=directory/"configuration.json"
    if path.exists() and json.loads(path.read_text()) != fingerprint:
        raise ValueError("Configuration changed; use another version directory")
    common.write_json(path,fingerprint)
    checkpoint=common.checkpoint(args.model,args.seed)
    payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
    if payload["model"]!=args.model or payload["seed"]!=args.seed or payload["split_metadata"]["split_id"]!="42fbb7440cc27c71":
        raise ValueError("Wrong checkpoint")
    model=core.make_model(args.model,10,32).to(device)
    model.load_state_dict(payload["model_state_dict"])
    del payload
    model.eval()
    specs=core.collect_conv_specs(model,32)
    event(args.root,"worker_start",mode=args.mode,model=args.model,variant=args.variant,device=str(device))
    if args.mode=="diagnose":
        diagnose(model,specs,args,directory,device)
    else:
        pilot(model,specs,args,directory,device)


if __name__=="__main__":
    main()
