#!/usr/bin/env python3
"""Pinned fresh binary training; no automatic test evaluation."""
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import yaml


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    repo=Path(__file__).resolve().parents[1];os.chdir(repo)
    cfg=repo/'config/labeling_binary_algae_v2.yaml';a=yaml.safe_load(cfg.read_text())
    baseline=yaml.safe_load((repo/'config/cls_aux_queue/b_aux_005.yaml').read_text())
    expected=dict(baseline,data_path='data/labeling_binary_v2_geo',
                  output_dir='outputs/labeling_binary_algae_v2',run_name='binary_aux005_seed42',
                  num_classes=2,segmentation_taxonomy='binary_algae',cls_aux_target_type='fine')
    assert a==expected,'Unexpected configuration drift'
    manifest=repo/a['data_path']/'manifest.json'
    verification=manifest.parent/'verification.json'
    checks=json.loads(verification.read_text())
    assert checks['status']=='passed' and checks['manifest_sha256']==digest(manifest)
    assert checks['raw_frames']==3986 and sum(checks['images'].values())==checks['unique_frames']
    assert checks['source_schema_mapping']=='category names, not numeric IDs'
    smoke=repo/'outputs/labeling_binary_algae_v2_checks'/f'{a["run_name"]}.json'
    s=json.loads(smoke.read_text())
    assert s['status']=='passed' and s['config_sha256']==digest(cfg)
    assert s['num_classes']==2 and s['cls_aux']['logits_shape']==[4,2]
    assert s['cls_aux']['vit_gradient_norm']>0
    out=repo/a['output_dir'];run_dir=out/a['run_name'];out.mkdir(parents=True,exist_ok=True)
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        files=[cfg,manifest,verification,smoke,repo/'train.py',Path(__file__).resolve(),
               repo/'tools/check_fusion_v2.py',repo/'docs/binary_algae_training_v2.md',
               repo/'tools/prepare_labeling_binary_v2.py']
        files+=sorted((repo/'goose_semseg').rglob('*.py'))
        files+=sorted((repo/'tests').glob('test_*.py'))
        hashes={str(p.relative_to(repo)):digest(p) for p in files}
        pin=out/'run_inputs.json'
        if pin.exists():
            assert json.loads(pin.read_text())==hashes,'Frozen run inputs changed; use a new run version'
        else:
            if run_dir.exists() and any(run_dir.iterdir()):raise RuntimeError('Unpinned output already exists')
            pin.write_text(json.dumps(hashes,indent=2))
            with tarfile.open(out/'source_snapshot.tar.gz','w:gz') as archive:
                for p in files:archive.add(p,arcname=str(p.relative_to(repo)))
        if (run_dir/'training_complete.json').exists():
            print('Already completed',flush=True);return
        if shutil.disk_usage(repo).free<15*1024**3:raise RuntimeError('Less than 15 GiB free')
        def event(status,**kwargs):
            e=dict(status=status,time=datetime.now(timezone.utc).isoformat(),**kwargs)
            (out/'state.json').write_text(json.dumps(e,indent=2))
            with (out/'status.jsonl').open('a') as f:f.write(json.dumps(e)+'\n')
            print(json.dumps(e),flush=True)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',PYTHONUNBUFFERED='1',
                 PYTHONPATH=str(repo)+':'+str(repo/'third_party'),OMP_NUM_THREADS='4',
                 HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        cmd=[sys.executable,'train.py','--config',str(cfg)]
        if (run_dir/'latest.pt').exists():cmd+=['--resume_from',str(run_dir/'latest.pt')]
        event('running',command=cmd,gpu_ids=[0,1],
              test_policy='No test evaluation during training; test was inspected in previous experiments',
              initialization='External DINOv3/Mask2Former pretraining; fresh 2-class and CLS output layers')
        try:
            with (out/'train.log').open('a') as log:
                result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f'Trainer exit {result.returncode}')
            complete=json.loads((run_dir/'training_complete.json').read_text())
            assert complete['status']=='completed' and complete['manifest_sha256']==digest(manifest)
            event('completed',best_val_miou=complete['best_val_miou'])
        except Exception as exc:
            event('failed',error=str(exc));raise


if __name__=='__main__':main()
