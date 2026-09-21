#!/usr/bin/env python3
"""Audit originals and freeze task-disjoint splits and train-only normalization."""
import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='Labeling_Data')
    p.add_argument('--output', default='data/fusion_v2')
    p.add_argument('--reuse-manifest', default=None,
                   help='Reuse source hashes from an audit performed in this session.')
    args = p.parse_args()
    source, out = Path(args.source).resolve(), Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    records, hashes, headers = [], defaultdict(list), Counter()
    if args.reuse_manifest:
        cached = json.loads(Path(args.reuse_manifest).read_text())
        assert cached['source'] == str(source)
        records = cached['records']
        for i,r in enumerate(records):
            hashes[r['sha256']].append(i)
    for sensor in (() if records else ('Altum', 'P1')):
        for path in sorted((source / sensor / 'Annotations').glob('*.json')):
            coco = json.loads(path.read_text())
            task = path.stem.removeprefix('instances_')
            annotations = defaultdict(list)
            for ann in coco['annotations']:
                if ann['category_id'] != 11:
                    annotations[ann['image_id']].append(ann)
            for item in sorted(coco['images'], key=lambda x: x['file_name']):
                image = source / sensor / 'Images' / task / item['file_name']
                h = hashlib.sha256()
                with image.open('rb') as f:
                    for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
                        h.update(block)
                digest = h.hexdigest()
                with Image.open(image) as im:
                    headers[(sensor, im.mode, getattr(im, 'n_frames', 1))] += 1
                    size = im.size
                anns = annotations[item['id']]
                classes = sorted({a['category_id'] - 1 - int(a['category_id'] > 11) for a in anns})
                records.append(dict(sensor=sensor, task=task, group=f'{sensor}/{task}',
                                    source_image=str(image.relative_to(source)), sha256=digest,
                                    width=item['width'], height=item['height'], source_size=list(size),
                                    classes_present=classes, annotations=anns))
                hashes[digest].append(len(records)-1)
    # Duplicate images couple tasks, even if their filenames differ.
    parents = {r['group']: r['group'] for r in records}
    def root(x):
        while parents[x] != x:
            x = parents[x]
        return x
    for indices in hashes.values():
        for i in indices[1:]:
            parents[root(records[i]['group'])] = root(records[indices[0]]['group'])
    gps_points = defaultdict(list)
    gps_times = defaultdict(list)
    for r in records:
        if r['sensor'] != 'P1':
            continue
        with Image.open(source/r['source_image']) as im:
            exif=im.getexif(); gps=exif.get_ifd(34853)
            if 2 not in gps or 4 not in gps:
                continue
            def degrees(v):
                return float(v[0])+float(v[1])/60+float(v[2])/3600
            lat=degrees(gps[2])*(-1 if gps.get(1)=='S' else 1)
            lon=degrees(gps[4])*(-1 if gps.get(3)=='W' else 1)
            gps_points[r['group']].append((lat,lon))
            gps_times[r['group']].append(str(exif.get(306,'')))
    geographic_links=[]
    keys=sorted(gps_points)
    for i,g in enumerate(keys):
        a=np.radians(gps_points[g])
        for h in keys[:i]:
            b=np.radians(gps_points[h])
            d=a[:,None,:]-b[None,:,:]
            hav=np.sin(d[:,:,0]/2)**2 + np.cos(a[:,None,0])*np.cos(b[None,:,0])*np.sin(d[:,:,1]/2)**2
            distance=float((6371000*2*np.arcsin(np.sqrt(hav.clip(0,1)))).min())
            if distance < 1000:
                parents[root(g)] = root(h)
                geographic_links.append(dict(groups=[g,h],min_gps_distance_m=distance))
    groups = sorted({root(r['group']) for r in records})
    totals = np.zeros((len(groups), 14))  # frames, sensor counts, per-class image presence
    for r in records:
        g = groups.index(root(r['group']))
        totals[g, 0] += 1
        totals[g, 1 + int(r['sensor'] == 'P1')] += 1
        totals[g, 3 + np.array(r['classes_present'], dtype=int)] += 1
    rare_area = np.zeros(len(groups))
    for r in records:
        rare_area[groups.index(root(r['group']))] += sum(
            ann.get('area',0)/(r['width']*r['height'])
            for ann in r['annotations'] if ann['category_id']==9)
    # algae4 exists in only three tasks: reserve the richest task for learning,
    # with the other two providing independent validation and test coverage.
    rare_train_group = int(rare_area.argmax())
    rng = np.random.default_rng(42)
    best, choice = float('inf'), None
    target = np.array([.7, .15, .15])
    available_groups=(totals[:,3:]>0).sum(0)
    # Optimize only source identities and label coverage, never model metrics.
    for _ in range(120000):
        assign = rng.choice(3, len(groups), p=target)
        if assign[rare_train_group] != 0:
            continue
        count = np.stack([totals[assign == s].sum(0) for s in range(3)])
        if (count[:, 1:3] == 0).any() or (count[0,3:]==0).any():
            continue
        if (count[:,3:][:,available_groups>=3]==0).any():
            continue
        ratio = count / totals.sum(0).clip(1)
        score = 12 * ((ratio[:, :3] - target[:, None]) ** 2).sum()
        score += ((ratio[:, 3:] - target[:, None]) ** 2).mean()
        if score < best:
            best, choice = score, assign.copy()
    if choice is None:
        raise RuntimeError('No geographically grouped split satisfying feasible class coverage; review grouping.')
    splits = ['train', 'val', 'test']
    for r in records:
        r['split'] = splits[choice[groups.index(root(r['group']))]]
        r['group_id'] = root(r['group'])
    # Exact duplicates must not be counted multiple times; retain one deterministic representative.
    unique, seen = [], set()
    for r in records:
        if r['sha256'] not in seen:
            seen.add(r['sha256']); unique.append(r)
    records = unique
    statistics = {}
    for sensor in ('Altum', 'P1'):
        values = []
        for r in records:
            if r['split'] != 'train' or r['sensor'] != sensor:
                continue
            with Image.open(source / r['source_image']) as im:
                if sensor == 'Altum':
                    channels = []
                    for band in range(7):
                        im.seek(band)
                        channels.append(np.asarray(im)[::24, ::24].astype(np.float32).ravel())
                    a = np.stack(channels)
                else:
                    a = np.asarray(im.convert('RGB'))[::48, ::48, ::-1].reshape(-1, 3).T.astype(np.float32) * 257
                    a = np.concatenate((a, np.zeros((4, a.shape[1]), dtype=np.float32)))
                values.append(a)
        a = np.concatenate(values, axis=1)
        low, high = np.percentile(a, [1, 99], axis=1)
        scale = np.maximum(high - low, np.maximum(a.std(1), 1))
        statistics[sensor] = dict(low=low.tolist(), scale=scale.tolist(),
                                  mean=a.mean(1).tolist(), std=a.std(1).tolist(),
                                  sample_count=int(a.shape[1]))
    old = Path('data/labeling_semseg_11cls_multispectral/manifest.json')
    old_records = json.loads(old.read_text())['records'] if old.exists() else []
    old_groups = defaultdict(set)
    for r in old_records:
        old_groups[r['sensor']+'/'+r['task']].add(r['split'])
    summary = {}
    for split in splits:
        rs = [r for r in records if r['split'] == split]
        summary[split] = dict(images=len(rs), sensors=dict(Counter(r['sensor'] for r in rs)),
                             groups=sorted({r['group'] for r in rs}),
                             class_images={str(i): sum(i in r['classes_present'] for r in rs) for i in range(11)})
    report = dict(old_cross_split_tasks={g: sorted(s) for g, s in old_groups.items() if len(s)>1},
                  duplicates_removed=sum(len(v)-1 for v in hashes.values()),
                  source_headers=(cached['audit']['source_headers'] if args.reuse_manifest else {str(k): v for k,v in headers.items()}), splits=summary,
                  rare_training_group=groups[rare_train_group],
                  geographic_merge_distance_m=1000, geographic_links=geographic_links,
                  class_independent_group_counts=available_groups.tolist(),
                  gps_time_ranges={g:[min(t),max(t)] for g,t in gps_times.items()},
                  checks=dict(task_overlap=0, exact_sha256_overlap=0, geographic_group_overlap=0,
                              normalization_split='train'),
                  limitations=['P1 groups merged within 1 km across dates. Altum has no GPS/time metadata: only task independence verified for Altum.',
                               'Classes confined to one independent group are train-only; do not hide unsupported validation/test classes.',
                               'Band identities are unverified; all 7 TIFF pages preserved.',
                               'ND indices are raw-page ratio proxies, not calibrated physical NDVI/NDRE.',
                               'New test contains some images previously used in old training; new runs start from external pretraining.'])
    for x in range(3):
        for y in range(x+1,3):
            a = [r for r in records if r['split']==splits[x]]
            b = [r for r in records if r['split']==splits[y]]
            assert not {r['group'] for r in a} & {r['group'] for r in b}
            assert not {root(r['group']) for r in a} & {root(r['group']) for r in b}
            assert not {r['sha256'] for r in a} & {r['sha256'] for r in b}
    out.mkdir(parents=True)
    portable_source = os.path.relpath(source, start=out.resolve())
    manifest = dict(source=portable_source, version=2, seed=42, normalization=statistics,
                    class_names=['land','bridge','other','nps','algae0','algae1','algae2','algae3','algae4','turbid','nps_algae'],
                    records=records, audit=report)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    (out/'split_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
