#!/usr/bin/env python3
"""Wait for existing aux=.1, then run a frozen aux OFF/.05/.2 comparison."""
import argparse
import csv
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

REPO=Path(__file__).resolve().parents[1]
NAMES=('a_aux_off','b_aux_005','c_aux_020')
WEIGHTS=(0.,.05,.2)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_config(base, candidate, name, weight):
    expected=dict(base,output_dir='outputs/cls_aux_queue',run_name=name,
                  enable_cls_aux=weight>0,cls_aux_weight=weight)
    if candidate!=expected:
        raise ValueError(f'{name}: only output/name, aux enable and weight may differ')
    if candidate.get('init_from') or candidate.get('resume_from'):
        raise ValueError('Fresh runs must start from external pretraining')


def verify_hashes(root, hashes):
    changed=[p for p,h in hashes.items() if not (root/p).is_file() or digest(root/p)!=h]
    if changed:
        raise RuntimeError(f'Frozen inputs changed: {changed[:5]}')


def assert_completed(run_dir, manifest_hash):
    marker=run_dir/'training_complete.json'
    if not marker.is_file():
        raise RuntimeError(f'{run_dir}: process stopped without successful completion; queue will not continue')
    result=json.loads(marker.read_text())
    if result.get('status')!='completed' or result.get('manifest_sha256')!=manifest_hash:
        raise RuntimeError(f'{run_dir}: completion marker does not match this protocol')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only',action='store_true')
    options=parser.parse_args()
    os.chdir(REPO)
    base_path=REPO/'config/labeling_coarse_aux.yaml'
    base=yaml.safe_load(base_path.read_text())
    assert base['enable_cls_aux'] and base['cls_aux_weight']==.1 and not base['freeze_backbone']
    configs=[REPO/'config/cls_aux_queue'/f'{name}.yaml' for name in NAMES]
    for cfg,name,weight in zip(configs,NAMES,WEIGHTS):
        validate_config(base,yaml.safe_load(cfg.read_text()),name,weight)
    current=REPO/'outputs/labeling_coarse_aux'
    current_run=current/'coarse7_aux'
    current_pin=current/'run_inputs.json'
    # Carry forward the actual .1 experiment's source, not merely similarly named configs.
    hashes=json.loads(current_pin.read_text())
    verify_hashes(REPO,hashes)
    files=configs+[current_pin,Path(__file__).resolve(),REPO/'docs/cls_aux_queue.md',
                   REPO/'tools/cls_aux_queue_test.py']
    hashes.update({str(p.relative_to(REPO)):digest(p) for p in files})
    manifest=REPO/base['data_path']/'manifest.json'
    manifest_hash=digest(manifest)
    out=REPO/'outputs/cls_aux_queue'
    plan=dict(gpu_ids=[0,1],test_policy='closed',tta=False,manifest_sha256=manifest_hash,
        common_backbone='DataParallel no_grad bug fixed for all four experiments',
        runs=[dict(name='coarse7_aux',weight=.1,external=True,output=str(current_run.relative_to(REPO)))]+
             [dict(name=name,weight=w,external=False,output=f'outputs/cls_aux_queue/{name}')
              for name,w in zip(NAMES,WEIGHTS)],
        max_epochs=base['epochs'],validation_interval=base['val_interval'],
        early_stop_validation_checks=base['early_stopping_patience'])
    if options.check_only:
        if (out/'queue_inputs.json').exists():
            assert json.loads((out/'queue_inputs.json').read_text())==hashes
        print(json.dumps(dict(status='passed',plan=plan),indent=2));return
    out.mkdir(parents=True,exist_ok=True)
    with (out/'queue.lock').open('a') as queue_lock:
        fcntl.flock(queue_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pinned=out/'queue_inputs.json'
        if pinned.exists():
            if json.loads(pinned.read_text())!=hashes:
                raise RuntimeError('Queue inputs changed; do not mix experiment versions')
        else:
            if any((out/name).exists() for name in NAMES):
                raise RuntimeError('Refusing to adopt unpinned existing run directories')
            pinned.write_text(json.dumps(hashes,indent=2))
            with tarfile.open(out/'source_snapshot.tar.gz','w:gz') as archive:
                for p in hashes: archive.add(REPO/p,arcname=p)
        (out/'plan.json').write_text(json.dumps(plan,indent=2))
        def event(name,status,**extra):
            record=dict(time=datetime.now(timezone.utc).isoformat(),experiment=name,status=status,**extra)
            with (out/'queue_status.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
            temporary=out/'queue_state.json.tmp'
            temporary.write_text(json.dumps(record,indent=2));temporary.replace(out/'queue_state.json')
            print(json.dumps(record),flush=True)
        try:
            # The existing runner holds this lock until training and its summary finish.
            # Acquiring it consumes no GPU, and prevents duplicate ownership afterwards.
            with (current/'run.lock').open('a') as predecessor_lock:
                event('coarse7_aux','waiting',reason='Wait for existing .1 run lock to be released')
                fcntl.flock(predecessor_lock,fcntl.LOCK_EX)
                assert_completed(current_run,manifest_hash)
                verify_hashes(REPO,hashes)
                event('coarse7_aux','completed_external')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',PYTHONUNBUFFERED='1',
                    PYTHONPATH=str(REPO)+':'+str(REPO/'third_party'),OMP_NUM_THREADS='4',
                    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
                for cfg,name in zip(configs,NAMES):
                    verify_hashes(REPO,hashes)
                    run_dir=out/name
                    if (run_dir/'training_complete.json').exists():
                        assert_completed(run_dir,manifest_hash)
                        event(name,'already_completed');continue
                    if shutil.disk_usage(REPO).free<15*1024**3:
                        raise RuntimeError('Less than 15 GiB free; preserve existing checkpoints and stop')
                    cmd=[sys.executable,'train.py','--config',str(cfg)]
                    if (run_dir/'latest.pt').exists(): cmd+=['--resume_from',str(run_dir/'latest.pt')]
                    event(name,'running',command=cmd)
                    with (out/f'{name}_train.log').open('a') as log:
                        result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
                    if result.returncode:
                        raise RuntimeError(f'{name} exited with {result.returncode}; see its train log')
                    assert_completed(run_dir,manifest_hash)
                    event(name,'completed')
                summary=[]
                for entry in plan['runs']:
                    run_dir=REPO/entry['output']
                    best=list(run_dir.glob('best_epoch_*.pt'))
                    assert len(best)==1
                    epoch=int(best[0].name.split('_')[2])
                    rows=list(csv.DictReader((run_dir/'epoch_metrics.csv').open()))
                    row=next(r for r in rows if int(r['epoch'])==epoch)
                    groups=json.loads((run_dir/f'val_groups_epoch_{epoch:03d}.json').read_text())['metrics']
                    summary.append(dict(name=entry['name'],aux_weight=entry['weight'],best_epoch=epoch,
                        best_val_miou=float(row['val_miou']),checkpoint=str(best[0].relative_to(REPO)),
                        supported_class_ids=groups['overall']['supported_class_ids'],
                        per_class=groups['overall']['per_class'],
                        group_miou={k:v['miou'] for k,v in groups.items()}))
                assert all(r['supported_class_ids']==summary[0]['supported_class_ids'] for r in summary)
                baseline=next(r['best_val_miou'] for r in summary if r['aux_weight']==0)
                for r in summary:r['delta_vs_corrected_aux_off_pp']=100*(r['best_val_miou']-baseline)
                (out/'summary.json').write_text(json.dumps(dict(test_opened=False,experiments=summary,
                    selection_metric='Same GT-supported fine-class val mIoU; not coarse/CLS accuracy',
                    limitation='One seed per setting; validation-selected comparison, not a final test result.'
                ),indent=2,allow_nan=False))
                event('all','completed')
        except Exception as exc:
            event('queue','failed',error=str(exc));raise


if __name__=='__main__':main()
