#!/usr/bin/env python3
"""A predeclared two-pass adjustment: mean repair then mean-constrained tail repair."""

import argparse
import json
from pathlib import Path

import run_shared_mask_repair as worker
import run_fair_criterion_r0p9 as common


def mean_safe_repair(weight,inputs,mask,epsilon,objective="worst",candidates=8,iterations=8):
    if objective=="worst":
        intermediate,first=original_repair(weight,inputs,mask,epsilon,"mean",candidates,iterations)
        result,second=original_repair(weight,inputs,intermediate,epsilon,"worst",candidates,iterations)
        return result,[dict(step,phase="mean") for step in first]+[dict(step,phase="tail") for step in second]
    result,history=original_repair(weight,inputs,mask,epsilon,objective,candidates,iterations*2)
    return result,[dict(step,phase="mean") for step in history]


original_repair=worker.repair


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument("--root",type=Path,required=True)
    known,_=parser.parse_known_args()
    known.root.mkdir(parents=True,exist_ok=True)
    definition={"method":"mean_safe_tail_repair", "worst_repair_label":"8 mean steps then 8 tail steps; mean cap reset to post-mean mask",
                "mean_repair_label":"16 mean steps (matched maximum search rounds)",
                "wrapper_sha256":common.sha(Path(__file__)),"base_worker_sha256":common.sha(Path(worker.__file__)),
                "repair_sha256":common.sha(Path(__file__).with_name("shared_mask_repair.py")),
                "test_used_for_selection":False}
    path=known.root/"method_definition.json"
    if path.exists() and json.loads(path.read_text())!=definition:
        raise ValueError("Mean-safe method changed; choose a new version")
    common.write_json(path,definition)
    worker.repair=mean_safe_repair
    worker.main()


if __name__=="__main__":
    main()
