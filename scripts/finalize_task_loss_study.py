#!/usr/bin/env python3
"""Audit completed artifacts, capture all local tests, and refresh the study package."""

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import run_fair_criterion_r0p9 as common
import run_task_loss_mask_study as study


def main():
    root = study.ROOT
    result = json.loads((root / "study_result.json").read_text())
    if result["state"] != "complete":
        raise RuntimeError("Do not finalize an active study")
    study.verify(result["jobs"])
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    test = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                          cwd=common.PROJECT, env=env, text=True, capture_output=True)
    (root / "UNIT_TESTS.txt").write_text(test.stdout + test.stderr, encoding="utf-8")
    test.check_returncode()
    for source in sorted((common.PROJECT / "tests").glob("test_*.py")) + [Path(__file__)]:
        shutil.copy2(source, root / "source_snapshot" / source.name)
    audit = []
    for job in result["jobs"]:
        path = study.directory(job)
        config = json.loads((path / "configuration.json").read_text())
        original = common.PROJECT / "runs/shared_mask_repair_study/v1" / f"resnet50_seed{job['seed']}"
        old_config = json.loads((original / "configuration.json").read_text())
        patch_config = json.loads((original / "patch_manifest.json").read_text())
        if config["core_sha256"] != old_config["core_sha256"] or config["checkpoint_sha256"] != patch_config["checkpoint_sha256"]:
            raise ValueError("Reused control source differs")
        for key, source in [("worker_sha256", common.PROJECT / "scripts/task_loss_mask_search.py"),
                            ("pilot_sha256", common.PROJECT / "scripts/run_shared_mask_repair.py"),
                            ("source_sha256", Path(config["source_masks"]))]:
            if common.sha(source) != config[key]:
                raise ValueError(f"Source SHA changed: {source}")
        audit.append(dict(job=job, core_equal_to_control=True, checkpoint_equal_to_control=True, source_hashes=True))
    common.write_json(root / "SOURCE_AUDIT.json", dict(passed=True, checks=audit, unit_tests=16))
    files = [p for p in root.rglob("*") if p.is_file() and p.name not in {"study.lock", "MANIFEST.json"}
             and (p.suffix != ".pt" or p.name == "masks.pt")]
    common.write_json(root / "MANIFEST.json", {str(p.relative_to(root)): common.sha(p) for p in files})
    archive = root.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in files + [root / "MANIFEST.json"]:
            z.write(path, path.relative_to(root.parent))
    archive.with_suffix(".zip.sha256").write_text(common.sha(archive) + "  " + archive.name + "\n", encoding="ascii")
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("Archive integrity error")
    for name, digest in json.loads((root / "MANIFEST.json").read_text()).items():
        if common.sha(root / name) != digest:
            raise ValueError(f"Manifest error: {name}")
    print(json.dumps(dict(passed=True, files=len(files), archive_mb=archive.stat().st_size / 1048576, test_evaluated=False)))


if __name__ == "__main__":
    main()
