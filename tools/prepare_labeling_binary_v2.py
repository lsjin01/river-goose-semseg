#!/usr/bin/env python3
"""Name-based binary labels, spatial/task split and train-only normalization."""
import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

POSITIVE={'algae','algae0','algae1','algae2','algae3','algae4','nps_algae'}
NEGATIVE={'river','land','bridge','other','nps','turbid'}
NAMES=['non_algae','algae_including_nps_algae']


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='Labeling_Data_v2')
    p.add_argument('--output',default='data/labeling_binary_v2_geo')
    args=p.parse_args();source=Path(args.source).resolve();out=Path(args.output)
    assert (source/'extraction_complete.json').is_file()
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    def progress(stage,**kw):
        e=dict(stage=stage,elapsed_seconds=time.monotonic()-start,**kw)
        (out/'progress.json').write_text(json.dumps(e,indent=2));print(json.dumps(e),flush=True)
    records=[];schemas={};annotation_names=Counter();json_hashes={};excluded=[];raw_frames=0
    for sensor in ('Altum','P1'):
        for path in sorted((source/sensor/'Annotations').glob('*.json')):
            d=json.loads(path.read_text());task=path.stem.removeprefix('instances_')
            names={c['id']:c['name'].strip().lower() for c in d['categories']}
            assert len(names)==len(d['categories'])
            assert not set(names.values())-(POSITIVE|NEGATIVE|{'ambiguous'})
            schemas[f'{sensor}/{task}']=names;json_hashes[str(path.relative_to(source))]=digest(path)
            anns=defaultdict(list);image_ids={i['id'] for i in d['images']}
            assert len(image_ids)==len(d['images'])
            for a in d['annotations']:
                assert a['image_id'] in image_ids
                name=names[a['category_id']];annotation_names[name]+=1
                label=1 if name in POSITIVE else 0 if name in NEGATIVE else 255
                assert isinstance(a['segmentation'],list),'RLE requires explicit support'
                for polygon in a['segmentation']:
                    assert len(polygon)>=6 and len(polygon)%2==0 and np.isfinite(polygon).all()
                anns[a['image_id']].append(dict(a,label_id=label,source_category_name=name))
            for item in sorted(d['images'],key=lambda i:i['file_name']):
                raw_frames+=1
                path_image=source/sensor/'Images'/task/item['file_name']
                assert path_image.resolve().is_relative_to(source) and path_image.is_file()
                a=anns[item['id']]
                classes=sorted({x['label_id'] for x in a if x['label_id']!=255 and x['segmentation']})
                if not classes:
                    excluded.append(dict(source_image=str(path_image.relative_to(source)),
                                         reason='No non-ignored polygon labels; not a negative example'))
                    continue
                records.append(dict(sensor=sensor,task=task,group=f'{sensor}/{task}',
                    source_image=str(path_image.relative_to(source)),width=item['width'],height=item['height'],
                    classes_present=classes,annotations=a))
    assert len({r['source_image'] for r in records})==len(records)
    progress('annotation_audit',raw_frames=raw_frames,labeled_frames=len(records),tasks=len(schemas),
             excluded=excluded,annotation_names=dict(annotation_names))
    def audit_image(record):
        r=dict(record);image=source/r['source_image'];r['sha256']=digest(image)
        with Image.open(image) as im:
            r['source_size']=list(im.size);r['source_pages']=getattr(im,'n_frames',1);r['source_mode']=im.mode
            if r['sensor']=='Altum':
                assert r['source_pages']==7
                for page in range(7):
                    im.seek(page);assert im.size==tuple(r['source_size'])
                im.seek(0)
            exif=im.getexif();gps=exif.get_ifd(34853)
            if 2 in gps and 4 in gps:
                def degrees(v):return float(v[0])+float(v[1])/60+float(v[2])/3600
                r['gps']=[degrees(gps[2])*(-1 if gps.get(1)=='S' else 1),
                          degrees(gps[4])*(-1 if gps.get(3)=='W' else 1)]
            r['capture_time']=str(exif.get(306,''))
        return r
    audited=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,r in enumerate(pool.map(audit_image,records),1):
            audited.append(r)
            if i%200==0:progress('image_audit',done=i,total=len(records))
    records=audited
    (out/'audited_records.json').write_text(json.dumps(records))
    parents={r['group']:r['group'] for r in records}
    def root(x):
        while parents[x]!=x:x=parents[x]
        return x
    def join(a,b):parents[root(a)]=root(b)
    hashes=defaultdict(list)
    for r in records:hashes[r['sha256']].append(r)
    for rs in hashes.values():
        for r in rs[1:]:join(r['group'],rs[0]['group'])
    gps=defaultdict(list)
    for r in records:
        if 'gps' in r:gps[r['group']].append(r['gps'])
    links=[];keys=sorted(gps)
    for i,g in enumerate(keys):
        a=np.radians(gps[g])
        for h in keys[:i]:
            b=np.radians(gps[h]);delta=a[:,None,:]-b[None,:,:]
            hav=np.sin(delta[:,:,0]/2)**2+np.cos(a[:,None,0])*np.cos(b[None,:,0])*np.sin(delta[:,:,1]/2)**2
            dist=float((6371000*2*np.arcsin(np.sqrt(hav.clip(0,1)))).min())
            if dist<1000:join(g,h);links.append(dict(groups=[g,h],min_gps_distance_m=dist))
    # Deduplicate only if the final binary annotation geometry agrees exactly.
    unique=[];duplicates=[]
    def annotation_signature(r):
        return json.dumps(sorted([(a['label_id'],a['area'],a['segmentation']) for a in r['annotations']],
                                 key=lambda a:json.dumps(a)),sort_keys=True)
    for rs in hashes.values():
        if len(rs)>1:
            assert len({annotation_signature(r) for r in rs})==1,'Duplicate images with conflicting labels: review required'
            duplicates.extend(r['source_image'] for r in rs[1:])
        unique.append(rs[0])
    records=unique
    groups=sorted({root(r['group']) for r in records});group_index={g:i for i,g in enumerate(groups)}
    totals=np.zeros((len(groups),5))
    for r in records:
        i=group_index[root(r['group'])];totals[i,0]+=1;totals[i,1+(r['sensor']=='P1')]+=1
        for c in r['classes_present']:totals[i,3+c]+=1
    target=np.array([.7,.15,.15]);rng=np.random.default_rng(42);best=float('inf');choice=None
    for _ in range(120000):
        assign=rng.choice(3,len(groups),p=target)
        count=np.stack([totals[assign==s].sum(0) for s in range(3)])
        if np.any(count[:,1:]==0):continue
        ratio=count/totals.sum(0).clip(1)
        score=12*np.square(ratio[:,:3]-target[:,None]).sum()+np.square(ratio[:,3:]-target[:,None]).mean()
        if score<best:best=score;choice=assign.copy()
    if choice is None:raise RuntimeError('No spatial split with both sensors and both classes')
    for r in records:
        r['group_id']=root(r['group']);r['split']=('train','val','test')[choice[group_index[r['group_id']]]]
    split_counts={s:dict(images=sum(r['split']==s for r in records),
        sensors=dict(Counter(r['sensor'] for r in records if r['split']==s)),
        tasks=len({r['group'] for r in records if r['split']==s}),
        groups=len({r['group_id'] for r in records if r['split']==s})) for s in ('train','val','test')}
    progress('split_frozen',splits=split_counts,independent_groups=len(groups),duplicates_removed=len(duplicates))
    statistics={}
    for sensor in ('Altum','P1'):
        train=[r for r in records if r['split']=='train' and r['sensor']==sensor]
        def samples(r):
            with Image.open(source/r['source_image']) as im:
                if sensor=='Altum':
                    bands=[]
                    for page in range(7):
                        im.seek(page);bands.append(np.asarray(im)[::24,::24].astype(np.float32).ravel())
                    return np.stack(bands)
                a=np.asarray(im.convert('RGB'))[::48,::48,::-1].reshape(-1,3).T.astype(np.float32)*257
                return np.concatenate([a,np.zeros((4,a.shape[1]),dtype=np.float32)])
        values=[]
        with ThreadPoolExecutor(max_workers=3) as pool:
            for i,a in enumerate(pool.map(samples,train),1):
                values.append(a)
                if i%200==0:progress('train_normalization',sensor=sensor,done=i,total=len(train))
        a=np.concatenate(values,axis=1);low,high=np.percentile(a,[1,99],axis=1)
        scale=np.maximum(high-low,np.maximum(a.std(1),1))
        statistics[sensor]=dict(low=low.tolist(),scale=scale.tolist(),mean=a.mean(1).tolist(),
                                std=a.std(1).tolist(),sample_count=int(a.shape[1]))
    manifest=dict(source=str(source),version='binary_labeling_v2',seed=42,
        class_names=NAMES,annotation_label_key='label_id',normalization=statistics,records=records,
        taxonomy=dict(positive_names=sorted(POSITIVE),negative_names=sorted(NEGATIVE),ignore_names=['ambiguous']),
        audit=dict(splits=split_counts,geographic_links=links,geographic_merge_distance_m=1000,
            schemas=schemas,annotation_json_sha256=json_hashes,annotation_names=dict(annotation_names),
            duplicates_removed=duplicates,excluded_unlabeled=excluded,normalization_split='train',
            limitations=['Tasks without GPS cannot certify site independence; cross-sensor matching is incomplete.',
                'All source TIFF pages retained; physical band identities unverified.',
                'Fresh expanded-data split; not directly comparable to the old 795-frame test.']))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    # Re-read the artifact and independently verify disjoint identities and grouping.
    m=json.loads((out/'manifest.json').read_text());splits=('train','val','test')
    for i,s in enumerate(splits):
        rs=[r for r in m['records'] if r['split']==s]
        assert {r['sensor'] for r in rs}=={'Altum','P1'}
        assert {c for r in rs for c in r['classes_present']}=={0,1}
        for t in splits[:i]:
            ts=[r for r in m['records'] if r['split']==t]
            for key in ('source_image','sha256','group','group_id'):
                assert not {r[key] for r in rs}&{r[key] for r in ts},(s,t,key)
    task_split={r['group']:r['split'] for r in records}
    assert all(task_split[a['groups'][0]]==task_split[a['groups'][1]] for a in links)
    verification=dict(status='passed',manifest_sha256=digest(out/'manifest.json'),
        images={s:split_counts[s]['images'] for s in splits},splits=split_counts,
        task_overlap=0,exact_hash_overlap=0,geographic_group_overlap=0,
        normalization_split='train',source_schema_mapping='category names, not numeric IDs',
        raw_frames=raw_frames,labeled_frames=len(audited),unique_frames=len(records),raw_tasks=len(schemas),
        excluded_unlabeled=excluded,
        gps_frames=dict(Counter(r['sensor'] for r in records if 'gps' in r)))
    (out/'verification.json').write_text(json.dumps(verification,indent=2))
    progress('completed',verification=verification)


if __name__=='__main__':main()
