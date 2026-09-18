#!/usr/bin/env python3
"""Independent assertions on the frozen source split and spatial grouping."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='data/fusion_v2_geo')
    a=p.parse_args()
    folder=Path(a.root)
    manifest=folder/'manifest.json'
    m=json.loads(manifest.read_text())
    bysplit={s:[r for r in m['records'] if r['split']==s] for s in ('train','val','test')}
    assert sum(map(len,bysplit.values()))==795
    task_split=defaultdict(set)
    for r in m['records']:
        task_split[r['group']].add(r['split'])
    assert all(len(s)==1 for s in task_split.values())
    for x,y in [('train','val'),('train','test'),('val','test')]:
        for key in ('source_image','sha256','group'):
            assert not {r[key] for r in bysplit[x]} & {r[key] for r in bysplit[y]},(x,y,key)
    for link in m['audit']['geographic_links']:
        x,y=link['groups']
        assert task_split[x]==task_split[y],link
    support={s:sorted({c for r in rs for c in r['classes_present']}) for s,rs in bysplit.items()}
    assert support['train']==list(range(11))
    for s,rs in bysplit.items():
        assert {r['sensor'] for r in rs}=={'Altum','P1'}
    # Count valid raster pixels for the rarest class to guard a misleading presence-only split.
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from goose_semseg.data.spectral_tiles import SpectralTileDataset
    rare_pixels={}
    tile_counts={}
    for s in bysplit:
        ds=SpectralTileDataset(folder,s)
        rare_pixels[s]=sum(int((ds._mask(i)==8).sum()) for i,r in enumerate(ds.records) if 8 in r['classes_present'])
        tile_counts[s]=len(ds)
    assert all(v>0 for v in rare_pixels.values())
    result=dict(status='passed',manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                images={s:len(rs) for s,rs in bysplit.items()},supported_class_ids=support,
                unsupported_classes={s:[m['class_names'][i] for i in range(11) if i not in cs] for s,cs in support.items()},
                algae4_raster_pixels=rare_pixels,tiles_or_train_samples=tile_counts,
                task_overlap=0,exact_hash_overlap=0,p1_geographic_group_overlap=0,
                limitation='Altum site/date and cross-sensor scene independence cannot be certified without missing metadata')
    (folder/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
