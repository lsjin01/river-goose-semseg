#!/usr/bin/env python3
"""Frozen two-arm sampling ablation. Validation only; no automatic test access."""
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
    repo = Path(__file__).resolve().parents[1]
    os.chdir(repo)
    out = repo/'outputs/fusion_v3'
    out.mkdir(parents=True, exist_ok=True)
    with (out/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        configs = sorted((repo/'config/fusion_v3').glob('*.yaml'))
        assert [c.stem for c in configs] == ['a_within_image','b_random_crop']
        manifest = repo/'data/fusion_v2_geo/manifest.json'
        analysis = repo/'outputs/fusion_v3_analysis/analysis.json'
        report = json.loads(analysis.read_text())
        assert report['manifest_sha256'] == digest(manifest)
        assert report['analyzed_splits'] == ['train','val'] and not report['test_opened']
        assert not report['annotation_issues']
        data = json.loads(manifest.read_text())
        for a,b in [('train','val'),('train','test'),('val','test')]:
            for key in ('source_image','sha256','group'):
                assert not {r[key] for r in data['records'] if r['split']==a} & {
                    r[key] for r in data['records'] if r['split']==b}
        split_by_group = {r['group']:r['split'] for r in data['records']}
        for link in data['audit']['geographic_links']:
            a,b = link['groups']; assert split_by_group[a] == split_by_group[b]
        cfg_values = [yaml.safe_load(c.read_text()) for c in configs]
        compare = [{k:v for k,v in c.items() if k not in ('run_name','class_sampling_prob')} for c in cfg_values]
        assert compare[0] == compare[1], 'The two arms must differ only in crop sampling probability'
        assert [c['class_sampling_prob'] for c in cfg_values] == [.2, .0]
        assert all(c['class_sampling_mode']=='within_image' for c in cfg_values)
        files = configs+[manifest, analysis, repo/'train.py', Path(__file__).resolve(),
                         repo/'tools/analyze_fusion_data.py', repo/'tools/check_fusion_v2.py']
        files += sorted((repo/'goose_semseg').rglob('*.py'))
        files += sorted((repo/'tests').glob('test_spectral_v*.py'))
        hashes = {str(p.relative_to(repo)):digest(p) for p in files}
        pinned = out/'queue_inputs.json'
        if pinned.exists():
            if json.loads(pinned.read_text()) != hashes:
                raise RuntimeError('Frozen inputs changed. Do not mix implementations in this queue.')
        else:
            pinned.write_text(json.dumps(hashes,indent=2))
            with tarfile.open(out/'source_snapshot.tar.gz','w:gz') as archive:
                for p in files: archive.add(p, arcname=str(p.relative_to(repo)))
        (out/'queue_plan.json').write_text(json.dumps(dict(
            experiments=[c.stem for c in configs], gpu_ids=[0,1], tta=False,
            test_policy='Closed: use validation only in this diagnostic round.',
            starting_weights='External DINOv3 and Mask2Former pretraining; no v2 task checkpoints.',
            manifest_sha256=digest(manifest)),indent=2))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1', PYTHONUNBUFFERED='1',
            PYTHONPATH=str(repo)+':'+str(repo/'third_party'), OMP_NUM_THREADS='4',
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        def record(name,status,**extra):
            event = dict(time=datetime.now(timezone.utc).isoformat(), experiment=name,
                         stage='train', status=status, **extra)
            with (out/'queue_status.jsonl').open('a') as f:
                f.write(json.dumps(event)+'\n')
            print(json.dumps(event), flush=True)
        for cfg in configs:
            run_dir = out/cfg.stem
            if (run_dir/'training_complete.json').exists():
                record(cfg.stem,'already_completed'); continue
            assert all(digest(repo/p)==v for p,v in hashes.items())
            if shutil.disk_usage(repo).free < 15*1024**3:
                raise RuntimeError('Less than 15 GiB free; preserve checkpoints and stop')
            cmd = [sys.executable,'train.py','--config',str(cfg)]
            if (run_dir/'latest.pt').exists():
                cmd += ['--resume_from',str(run_dir/'latest.pt')]
            record(cfg.stem,'running', command=cmd)
            with (out/f'{cfg.stem}_train.log').open('a') as log:
                result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            record(cfg.stem,'completed' if result.returncode==0 else 'failed',returncode=result.returncode)
            if result.returncode:
                raise RuntimeError(f'{cfg.stem} failed; check its log')
            assert (run_dir/'training_complete.json').exists()
        summary = []
        for cfg in configs:
            import csv
            run_dir = out/cfg.stem
            rows = list(csv.DictReader((run_dir/'epoch_metrics.csv').open()))
            # Match the actual retained checkpoint rather than ignoring min_delta.
            best = next(run_dir.glob('best_epoch_*.pt'))
            epoch = int(best.name.split('_')[2])
            row = next(r for r in rows if int(r['epoch'])==epoch)
            detail = json.loads((run_dir/f'val_groups_epoch_{epoch:03d}.json').read_text())
            summary.append(dict(experiment=cfg.stem, best_epoch=epoch,
                val_miou=float(row['val_miou']), checkpoint=str(best.relative_to(repo)),
                validation_groups={k:dict(miou=v['miou'],supported_class_ids=v['supported_class_ids'])
                                   for k,v in detail['metrics'].items()}))
        (out/'summary.json').write_text(json.dumps(dict(test_opened=False,experiments=summary),indent=2))
        record('all','completed')


if __name__ == '__main__':
    main()
