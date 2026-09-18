#!/usr/bin/env python3
"""Train/validation raster audit; deliberately never opens test images or masks."""
import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from goose_semseg.data.spectral_tiles import SpectralTileDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/fusion_v2_geo')
    parser.add_argument('--output', default='outputs/fusion_v3_analysis')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.data)/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    names = manifest['class_names']
    parents = {r['group']:r['group'] for r in manifest['records']}
    def root(group):
        while parents[group] != group:
            group = parents[group]
        return group
    for link in manifest['audit']['geographic_links']:
        a,b = link['groups']
        parents[root(a)] = root(b)
    rows, groups, selected = [], {}, {}
    palette = np.array([[130,100,60],[240,180,40],[150,150,150],[240,80,80],
                        [50,170,240],[40,230,230],[40,180,90],[110,220,30],
                        [210,240,20],[140,70,200],[230,90,200]], dtype=np.uint8)
    issues = []
    for split in ('train', 'val'):
        ds = SpectralTileDataset(args.data, split, flip_prob=0, rare_prob=0)
        for index, record in enumerate(ds.records):
            mask = ds._mask(index)
            counts = np.bincount(mask.ravel(), minlength=256)
            valid = int(counts[:11].sum())
            present = np.flatnonzero(counts[:11]).tolist()
            key = f"{split}/{record['group']}"
            g = groups.setdefault(key, dict(split=split, group=record['group'],
                sensor=record['sensor'], group_id=root(record['group']), images=0,
                pixels=[0]*11, class_images=[0]*11, ignored_pixels=0, total_pixels=0))
            g['images'] += 1
            g['pixels'] = (np.array(g['pixels'])+counts[:11]).tolist()
            g['class_images'] = (np.array(g['class_images'])+(counts[:11]>0)).tolist()
            g['ignored_pixels'] += int(counts[255])
            g['total_pixels'] += mask.size
            row = dict(split=split, group=record['group'], sensor=record['sensor'],
                source_image=record['source_image'], width=record['width'], height=record['height'],
                valid_fraction=valid/mask.size,
                **{name:int(counts[c]) for c,name in enumerate(names)})
            rows.append(row)
            missing = sorted(set(record['classes_present'])-set(present))
            if missing:
                issues.append(dict(image=record['source_image'], vanished_classes=missing))
            for ann in record['annotations']:
                for polygon in ann.get('segmentation', []):
                    a = np.asarray(polygon).reshape(-1, 2)
                    if len(a) and ((a[:,0]<0).any() or (a[:,1]<0).any() or
                                  (a[:,0]>record['width']).any() or (a[:,1]>record['height']).any()):
                        issues.append(dict(image=record['source_image'], out_of_bounds_annotation=ann.get('id')))
            # Select a large-area representative for each class and sensor, in each split.
            for c in present:
                sample_key = (split, record['sensor'], c)
                fraction = counts[c]/mask.size
                if sample_key not in selected or fraction > selected[sample_key][0]:
                    small = Image.fromarray(mask).resize((384,256), Image.Resampling.NEAREST)
                    selected[sample_key] = (fraction, record, np.array(small))
            if index % 40 == 0:
                print(split, index, '/', len(ds.records), flush=True)
    with (out/'image_pixels.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (out/'group_pixels.csv').open('w') as f:
        fieldnames = ['split','group','sensor','images','ignored_fraction']+names
        writer = csv.DictWriter(f, fieldnames=fieldnames); writer.writeheader()
        for g in groups.values():
            writer.writerow(dict(split=g['split'],group=g['group'],sensor=g['sensor'],images=g['images'],
                ignored_fraction=g['ignored_pixels']/g['total_pixels'],**dict(zip(names,g['pixels']))))
    summary = {}
    for split in ('train','val'):
        gs = [g for g in groups.values() if g['split']==split]
        counts = np.sum([g['pixels'] for g in gs], axis=0)
        images = np.sum([g['class_images'] for g in gs], axis=0)
        summary[split] = dict(images=sum(g['images'] for g in gs),
            ignored_fraction=sum(g['ignored_pixels'] for g in gs)/sum(g['total_pixels'] for g in gs),
            classes={name:dict(pixels=int(counts[c]), pixel_fraction=float(counts[c]/counts.sum()),
                images=int(images[c]), independent_groups=len({g['group_id'] for g in gs if g['pixels'][c]>0}))
                for c,name in enumerate(names)})
    # Visual diagnostic panels use only train/val, never the reserved test set.
    for split in ('train','val'):
        for sensor in ('Altum','P1'):
            samples = [(k,v) for k,v in selected.items() if k[:2]==(split,sensor)]
            canvas = Image.new('RGB',(1152,280*len(samples)), 'white')
            draw = ImageDraw.Draw(canvas)
            for row_index, (key, (_,r,m)) in enumerate(sorted(samples)):
                with Image.open(Path(manifest['source'])/r['source_image']) as im:
                    if sensor=='P1':
                        rgb = np.array(im.convert('RGB').resize((384,256)))
                    else:
                        rgb = []
                        for band in (2,1,0):
                            im.seek(band)
                            page = np.asarray(im.resize((384,256))).astype(float)
                            lo,hi = np.percentile(page,[1,99])
                            rgb.append(np.clip((page-lo)/max(hi-lo,1)*255,0,255))
                        rgb = np.stack(rgb,axis=-1).astype(np.uint8)
                color = np.zeros_like(rgb)
                for c in range(11): color[m==c] = palette[c]
                overlay = rgb.copy(); v=m!=255
                overlay[v] = (.55*rgb[v]+.45*color[v]).astype(np.uint8)
                y=row_index*280
                draw.text((4,y+3), f"{names[key[2]]} | {r['group']} | {Path(r['source_image']).name}", fill='black')
                for x,a in zip((0,384,768),(rgb,color,overlay)):
                    canvas.paste(Image.fromarray(a),(x,y+24))
            canvas.save(out/f'{split}_{sensor}_overlays.jpg', quality=88)
    report = dict(manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        analyzed_splits=['train','val'], test_opened=False, summary=summary, groups=groups,
        annotation_issues=issues,
        caveat='Altum previews use independently stretched pages 3/2/1; physical band identities are unverified.')
    (out/'analysis.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(summary,indent=2), flush=True)
    print('annotation issues',len(issues),flush=True)


if __name__ == '__main__':
    main()
