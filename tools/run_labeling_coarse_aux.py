#!/usr/bin/env python3
"""Pinned, resumable CLS-aux run, with the test set kept closed."""
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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    repo=Path(__file__).resolve().parents[1]
    os.chdir(repo)
    cfg=repo/'config/labeling_coarse_aux.yaml'
    args=yaml.safe_load(cfg.read_text())
    out=repo/args['output_dir']
    run_dir=out/args['run_name']
    out.mkdir(parents=True,exist_ok=True)
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        baseline=repo/'config/fusion_v3/b_random_crop.yaml'
        baseline_args=yaml.safe_load(baseline.read_text())
        differences={k for k in set(args)|set(baseline_args) if args.get(k)!=baseline_args.get(k)}
        assert differences=={'output_dir','run_name','enable_cls_aux','cls_aux_taxonomy',
                             'cls_aux_target_type','cls_aux_loss_type','cls_aux_weight'},differences
        assert args['enable_cls_aux'] and args['cls_aux_taxonomy']=='labeling_data'
        assert args['cls_aux_weight']==.1 and args['cls_aux_target_type']=='coarse'
        manifest=repo/args['data_path']/'manifest.json'
        verification=manifest.parent/'verification.json'
        checks=json.loads(verification.read_text())
        assert checks['status']=='passed' and checks['manifest_sha256']==digest(manifest)
        smoke=repo/'outputs/labeling_coarse_aux_checks/coarse7_aux.json'
        smoke_result=json.loads(smoke.read_text())
        assert smoke_result['status']=='passed' and smoke_result['config_sha256']==digest(cfg)
        assert smoke_result['cls_aux']['logits_shape']==[4,7]
        files=[cfg,baseline,manifest,verification,smoke,repo/'train.py',Path(__file__).resolve(),
               repo/'tools/check_fusion_v2.py',repo/'docs/labeling_data_taxonomy.md']
        files+=sorted((repo/'goose_semseg').rglob('*.py'))
        files+=sorted((repo/'tests').glob('test_*.py'))
        hashes={str(p.relative_to(repo)):digest(p) for p in files}
        pinned=out/'run_inputs.json'
        if pinned.exists():
            if json.loads(pinned.read_text())!=hashes:
                raise RuntimeError('Frozen input changed: use a different experiment version')
        else:
            if run_dir.exists() and any(run_dir.iterdir()):
                raise RuntimeError('Refusing to adopt an existing unpinned run')
            pinned.write_text(json.dumps(hashes,indent=2))
            with tarfile.open(out/'source_snapshot.tar.gz','w:gz') as archive:
                for p in files: archive.add(p,arcname=str(p.relative_to(repo)))
        def event(status,**extra):
            record=dict(time=datetime.now(timezone.utc).isoformat(),experiment=args['run_name'],
                        status=status,**extra)
            with (out/'status.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
        if (run_dir/'training_complete.json').exists():
            event('already_completed');return
        if shutil.disk_usage(repo).free<15*1024**3:
            raise RuntimeError('Less than 15 GiB free: cannot safely save checkpoints')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',PYTHONUNBUFFERED='1',
                 PYTHONPATH=str(repo)+':'+str(repo/'third_party'),OMP_NUM_THREADS='4',
                 HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        cmd=[sys.executable,'train.py','--config',str(cfg)]
        if (run_dir/'latest.pt').exists(): cmd+=['--resume_from',str(run_dir/'latest.pt')]
        event('running',command=cmd,gpu_ids=[0,1],test_policy='closed',
              initialization='external pretraining; resume only this run if interrupted')
        with (out/'train.log').open('a') as log:
            result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        event('completed' if result.returncode==0 else 'failed',returncode=result.returncode)
        if result.returncode: raise RuntimeError('Training failed; see outputs/labeling_coarse_aux/train.log')
        completed=json.loads((run_dir/'training_complete.json').read_text())
        old=json.loads((repo/'outputs/fusion_v3/b_random_crop/training_complete.json').read_text())
        assert completed['manifest_sha256']==old['manifest_sha256']==digest(manifest)
        best=list(run_dir.glob('best_epoch_*.pt'))
        assert len(best)==1
        (out/'summary.json').write_text(json.dumps(dict(test_opened=False,
            best_val_miou=completed['best_val_miou'],baseline_best_val_miou=old['best_val_miou'],
            delta_percentage_points=100*(completed['best_val_miou']-old['best_val_miou']),
            checkpoint=str(best[0].relative_to(repo)),
            comparison='Same frozen validation and GT-supported fine-class metric; CLS accuracy is not mIoU.',
            confound='This run also fixes DataParallel backbone no_grad. Difference from V3 is NOT an isolated CLS-aux effect.'
        ),indent=2))


if __name__=='__main__':main()
