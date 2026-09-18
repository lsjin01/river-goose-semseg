#!/usr/bin/env python3
"""Rank complete test frames using fine or coarse predictions, without TTA."""
import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from evaluate_grouped_test import (
    ROOT, torch, DataLoader, SpectralTileDataset, fine_to_coarse_mapping,
    load_dinov3_backbone, load_pretrained_mask2former,
    infer_pretrained_feature_channels, build_segmentation_decoder,
    seed_everything, mask2former_semantic_scores, update_confusion_matrix,
    aggregate_confusion, metric, COARSE_NAMES, atomic_json,
)
from goose_semseg.data.labeling_taxonomy import FINE_NAMES

COLORS = np.array([[150,115,75], [135,90,210], [150,155,165],
                   [245,145,35], [40,190,85], [40,175,230], [220,55,135]], dtype=np.uint8)
LOOKUP = np.array([0,1,2,3,4,4,4,4,4,5,6], dtype=np.uint8)
FINE_COLORS = np.array([[150,115,75], [135,90,210], [150,155,165],
    [245,145,35], [190,230,180], [230,215,60], [40,190,85],
    [0,110,105], [45,65,155], [40,175,230], [220,55,135]], dtype=np.uint8)


def thumbnail(mask):
    im = Image.fromarray(mask)
    im.thumbnail((1280,1280), Image.Resampling.NEAREST)
    return im


def heatmap(cm, names, path, title):
    cm = np.asarray(cm, dtype=np.float64)
    percentages = np.divide(cm * 100, cm.sum(1, keepdims=True),
                            out=np.zeros_like(cm), where=cm.sum(1, keepdims=True)>0)
    n = len(names)
    fig, ax = plt.subplots(figsize=(max(9,n),max(7,n*.8)))
    im = ax.imshow(percentages, vmin=0, vmax=100, cmap='Blues')
    ax.set(xticks=range(n), yticks=range(n), xticklabels=names, yticklabels=names,
           xlabel='Prediction', ylabel='Ground truth', title=title)
    plt.setp(ax.get_xticklabels(), rotation=40, ha='right')
    for i in range(n):
        for j in range(n):
            ax.text(j,i,f'{percentages[i,j]:.1f}',ha='center',va='center',
                    color='white' if percentages[i,j]>50 else 'black', fontsize=9)
    fig.colorbar(im,ax=ax,label='% of each ground-truth class')
    fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)


def source_preview(ds, record, size):
    with Image.open(ds.source/record['source_image']) as im:
        if record['sensor']=='P1':
            return np.array(im.convert('RGB').resize(size,Image.Resampling.LANCZOS)), 'Source RGB'
        bands=[]
        for page in range(3):
            im.seek(page)
            band=np.array(im,dtype=np.float32)
            lo,hi=np.percentile(band,[1,99])
            bands.append(np.uint8(np.clip((band-lo)/max(hi-lo,1),0,1)*255))
        rgb=Image.fromarray(np.stack(bands,-1)).resize(size,Image.Resampling.LANCZOS)
        return np.array(rgb),'Altum pages 1/2/3 (display composite; not verified RGB)'


def draw_case(ds, entry, out, label, names=COARSE_NAMES, colors=COLORS):
    idx=entry['frame_index']; r=ds.records[idx]
    pred=np.array(Image.open(out/'previews'/f'{idx:03d}_pred.png'))
    gt=np.array(Image.open(out/'previews'/f'{idx:03d}_gt.png'))
    valid=gt!=255
    rgb, source_title=source_preview(ds,r,(gt.shape[1],gt.shape[0]))
    palette=np.zeros((256,3),dtype=np.uint8);palette[:len(names)]=colors
    gcolor=palette[gt];pcolor=palette[pred];pcolor[~valid]=0
    errors=(rgb.astype(np.float32)*.35).astype(np.uint8)
    errors[valid & (pred!=gt)]=[255,35,40];errors[~valid]=0
    fig,axes=plt.subplots(1,4,figsize=(22,7.0 if len(names)>7 else 6.3))
    for ax,img,title in zip(axes,[rgb,gcolor,pcolor,errors],
                            [source_title,f'Ground truth ({len(names)} classes)',f'Prediction ({len(names)} classes)',
                             'Errors: red | Ignored: black']):
        ax.imshow(img);ax.set_title(title,fontsize=10);ax.axis('off')
    fig.suptitle(f'{label} | {r["sensor"]}/{r["task"]}/{Path(r["source_image"]).name}\n'
                 f'Full-frame mIoU: {entry["miou"]*100:.2f}% | '
                 f'GT-present classes: {len(entry["supported_class_ids"])} | No TTA',fontsize=14)
    fig.legend(handles=[Patch(color=c/255,label=n) for c,n in zip(colors,names)],
               loc='lower center',ncol=6 if len(names)>7 else 7,frameon=False)
    fig.subplots_adjust(left=.01,right=.99,top=.82,bottom=.10,wspace=.035)
    name=f'{label.lower().replace(" ","_")}_frame_{idx:03d}.png'
    fig.savefig(out/name,dpi=140);plt.close(fig)
    return name


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--taxonomy',choices=['coarse7','fine11'],default='coarse7')
    p.add_argument('--worst-only',action='store_true')
    args=p.parse_args();os.chdir(ROOT)
    is_fine=args.taxonomy=='fine11'
    names=FINE_NAMES if is_fine else COARSE_NAMES
    colors=FINE_COLORS if is_fine else COLORS
    previous=json.loads(Path(args.evaluation).read_text())
    out=Path(args.output_dir).resolve()
    if (out/'report.json').exists():raise FileExistsError('Completed report already exists')
    (out/'previews').mkdir(parents=True,exist_ok=True)
    a=argparse.Namespace(**previous['train_args'])
    manifest=ROOT/a.data_path/'manifest.json'
    assert hashlib.sha256(manifest.read_bytes()).hexdigest()==previous['manifest_sha256']
    ds=SpectralTileDataset(ROOT/a.data_path,'test',a.tile_size)
    fine_to_coarse_mapping(ds.manifest['class_names'])
    assert len(ds.records)==previous['frames'] and len(ds)==previous['tiles']
    seed_everything(a.seed)
    backbone=load_dinov3_backbone(a)
    pretrained=load_pretrained_mask2former(a.mask2former_pretrained_model_name_or_path)
    model=build_segmentation_decoder(backbone,decoder_type='m2f',hidden_dim=a.hidden_dim,
        num_classes=a.num_classes,autocast_dtype=torch.bfloat16,freeze_backbone=a.freeze_backbone,
        feature_channels=infer_pretrained_feature_channels(pretrained),input_channels=a.input_channels,
        fusion_type=a.fusion_type,precomputed_indices=True,
        cls_aux_num_classes=a.cls_aux_num_classes if a.enable_cls_aux else 0)
    del pretrained
    payload=torch.load(ROOT/previous['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    model.load_state_dict(payload['model_state_dict'],strict=True);del payload
    model=model.cuda().eval()
    loader=DataLoader(ds,batch_size=args.batch_size,num_workers=4,pin_memory=True,prefetch_factor=1)
    matrices=torch.zeros((len(ds.records),11,11),dtype=torch.int64,device='cuda')
    offset=0; current=None; pred_frame=None; gt_frame=None;start=time.monotonic()
    def save_preview(idx, prediction, target):
        thumbnail(prediction).save(out/'previews'/f'{idx:03d}_pred.png')
        thumbnail(target).save(out/'previews'/f'{idx:03d}_gt.png')
    with torch.inference_mode():
        for x,y in loader:
            x,y=x.cuda(non_blocking=True),y.cuda(non_blocking=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred=mask2former_semantic_scores(model(x),target_size=y.shape[-2:]).argmax(1)
            pn=pred.byte().cpu().numpy();yn=y.byte().cpu().numpy()
            for j in range(len(y)):
                idx,left,top=ds.tiles[offset+j];r=ds.records[idx]
                update_confusion_matrix(matrices[idx],pred[j],y[j],11,255)
                if current!=idx:
                    if current is not None:save_preview(current,pred_frame,gt_frame)
                    current=idx
                    pred_frame=np.full((r['height'],r['width']),255,dtype=np.uint8)
                    gt_frame=np.full_like(pred_frame,255)
                h=min(a.tile_size,r['height']-top);w=min(a.tile_size,r['width']-left)
                pred_frame[top:top+h,left:left+w]=pn[j,:h,:w] if is_fine else LOOKUP[pn[j,:h,:w]]
                tile=yn[j,:h,:w];mapped=np.full_like(tile,255);valid=tile!=255
                mapped[valid]=tile[valid] if is_fine else LOOKUP[tile[valid]]
                gt_frame[top:top+h,left:left+w]=mapped
            offset+=len(y)
            if offset%256==0 or offset==len(ds):
                progress=dict(tiles_done=offset,total_tiles=len(ds),elapsed_seconds=time.monotonic()-start)
                atomic_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    save_preview(current,pred_frame,gt_frame)
    total=matrices.sum(0).cpu()
    old=torch.tensor(previous['fine_metrics']['overall']['confusion_matrix'])
    assert torch.equal(total,old),'Rerun predictions differ from recorded evaluation; inspect before publishing'
    entries=[]
    for idx,cm in enumerate(matrices.cpu()):
        r=ds.records[idx];m=metric(cm if is_fine else aggregate_confusion(cm),names)
        entries.append(dict(frame_index=idx,source_image=r['source_image'],sensor=r['sensor'],
                            task=r['task'],**m))
    ranked=sorted(entries,key=lambda e:(-e['miou'],e['source_image']))
    selected={}
    case_groups=[('worst',list(reversed(ranked[-3:])))]
    if not args.worst_only:case_groups.insert(0,('best',ranked[:3]))
    for category,items in case_groups:
        selected[category]=[]
        for rank,entry in enumerate(items,1):
            name=draw_case(ds,entry,out,f'{category.title()} {rank}',names,colors)
            selected[category].append(dict(frame_index=entry['frame_index'],source_image=entry['source_image'],
                miou=entry['miou'],supported_class_ids=entry['supported_class_ids'],image=name))
    cm=previous['fine_metrics' if is_fine else 'coarse_metrics']['overall']['confusion_matrix']
    if not is_fine:
        heatmap(cm,COARSE_NAMES,out/'confusion_coarse7.png','Test confusion | algae0-4 merged | No TTA')
    fine_names=[r['class_name'] for r in previous['fine_metrics']['overall']['per_class']]
    heatmap(old.numpy(),fine_names,out/'confusion_fine11.png','Test confusion | fine classes | No TTA (algae0 has no GT)')
    confusions=[]
    for i,row in enumerate(cm):
        for j,count in enumerate(row):
            if i!=j and count:
                confusions.append(dict(ground_truth=names[i],prediction=names[j],
                    pixels=count,percent_of_gt=100*count/sum(row)))
    confusions.sort(key=lambda e:-e['percent_of_gt'])
    with (out/'frame_metrics.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['rank','source_image','sensor','task','miou','gt_present_classes'])
        for rank,e in enumerate(ranked,1):writer.writerow([rank,e['source_image'],e['sensor'],e['task'],
            e['miou'],';'.join(names[i] for i in e['supported_class_ids'])])
    report=dict(status='completed',evaluation=str(Path(args.evaluation)),checkpoint=previous['checkpoint'],
        manifest_sha256=previous['manifest_sha256'],tta=False,frames=len(entries),tiles=offset,
        taxonomy=args.taxonomy,class_names=list(names),
        ranking='Full-frame pixel confusion; mIoU averages classes with GT support in that frame. '
                'Different frame class composition can affect rankings; not mean tile IoU.',
        visualization='Nearest-neighbor reduced masks (max 1280 px); original-resolution scoring. '
                      'Altum display uses pages 1/2/3 with per-page percentile stretch, not verified RGB.',
        exact_match_previous_confusion=True,selected=selected,confusions=confusions,frame_metrics=entries)
    atomic_json(out/'report.json',report)
    lines=[f'# Test {args.taxonomy} error analysis','',
           f'- Checkpoint: `{previous["checkpoint"]}`',
           '- No TTA. '+('algae0–4 remain separate.' if is_fine else 'algae0–4 merged; nps_algae remains separate.'),
           '- 114 full frames / 6,670 tiles. Rerun confusion exactly matches prior evaluation.',
           f'- Cases ranked by full-frame GT-supported {len(names)}-class mIoU, not tile average.',
           '- Present classes differ between frames; best/worst cases are descriptive, not representative estimates.',
           '- Previews downsampled; metrics use full resolution. Altum page composite is not verified RGB.','',
           '## Confusion (row-normalized)','',
           '![Confusion]('+('confusion_fine11.png' if is_fine else 'confusion_coarse7.png')+')','',
           '| Ground truth | Prediction | % of true-class pixels |','|---|---|---:|']
    for e in confusions[:12]:lines.append(f'| {e["ground_truth"]} | {e["prediction"]} | {e["percent_of_gt"]:.2f} |')
    for category in selected:
        lines.extend(['',f'## {category.title()} 3',''])
        for e in selected[category]:lines.extend([f'- `{e["source_image"]}`: {e["miou"]*100:.2f}%',
                                                f'![{category}]({e["image"]})',''])
    (out/'README.md').write_text('\n'.join(lines))
    atomic_json(out/'progress.json',dict(status='completed',frames=len(entries),tiles=offset))
    print(json.dumps(dict(status='completed',selected=selected)),flush=True)


if __name__=='__main__':main()
