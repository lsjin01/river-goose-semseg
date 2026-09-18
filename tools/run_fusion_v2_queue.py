#!/usr/bin/env python3
"""Sequential, locked queue with pinned split/config/source hashes and final test evaluation."""
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    repo = Path(__file__).resolve().parents[1]
    os.chdir(repo)
    out = repo/'outputs/fusion_v2'
    out.mkdir(parents=True, exist_ok=True)
    with (out/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        configs = sorted((repo/'config/fusion_v2').glob('*.yaml'))
        assert len(configs)==5
        manifest=repo/'data/fusion_v2_geo/manifest.json'
        verification=json.loads((manifest.parent/'verification.json').read_text())
        assert verification['status']=='passed' and verification['manifest_sha256']==digest(manifest)
        files = configs + [manifest,repo/'train.py']
        files += sorted((repo/'goose_semseg').rglob('*.py'))
        files += [repo/'tools/check_fusion_v2.py',Path(__file__).resolve()]
        hashes = {str(p.relative_to(repo)):digest(p) for p in files}
        pinned = out/'queue_inputs.json'
        if pinned.exists():
            if json.loads(pinned.read_text()) != hashes:
                raise RuntimeError('Queue source/config/manifest changed; use a new experiment version.')
        else:
            pinned.write_text(json.dumps(hashes,indent=2))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1',PYTHONUNBUFFERED='1',
                   PYTHONPATH=str(repo)+':'+str(repo/'third_party'),OMP_NUM_THREADS='4',
                   HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        def record(name,stage,status):
            with (out/'queue_status.jsonl').open('a') as f:
                f.write(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(),
                                       experiment=name,stage=stage,status=status))+'\n')
        def run(name,stage,command):
            # Verify every experiment consumes the same frozen implementation.
            assert all(digest(repo/p)==v for p,v in hashes.items())
            if shutil.disk_usage(repo).free < 15*1024**3:
                raise RuntimeError('Insufficient disk space for atomic checkpoint saving')
            record(name,stage,'running')
            with (out/(name+'_'+stage+'.log')).open('a') as log:
                result = subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            record(name,stage,'completed' if result.returncode==0 else 'failed')
            if result.returncode:
                raise RuntimeError(f'{name} {stage} failed; see {log.name}')
        for cfg in configs:
            name = cfg.stem
            run_dir = out/name
            if (run_dir/'training_complete.json').exists():
                record(name,'train','already_completed')
                continue
            cmd = [sys.executable,'train.py','--config',str(cfg)]
            if (run_dir/'latest.pt').exists():
                cmd += ['--resume_from',str(run_dir/'latest.pt')]
            run(name,'train',cmd)
            assert (run_dir/'training_complete.json').exists()
        # Test is opened only after all predeclared models finish; no test-driven training decisions.
        for cfg in configs:
            if not (out/cfg.stem/'test_no_tta.json').exists():
                run(cfg.stem,'test',[sys.executable,'tools/check_fusion_v2.py','--config',str(cfg),'--evaluate'])
        summary=[]
        for cfg in configs:
            result=json.loads((out/cfg.stem/'test_no_tta.json').read_text())
            trained=json.loads((out/cfg.stem/'training_complete.json').read_text())
            summary.append(dict(experiment=cfg.stem,best_val_miou=trained['best_val_miou'],
                                test_miou={s:r['miou'] for s,r in result['metrics'].items()}))
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
        record('all','queue','completed')


if __name__=='__main__':
    main()
