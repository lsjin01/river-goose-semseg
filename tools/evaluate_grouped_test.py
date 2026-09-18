#!/usr/bin/env python3
"""Evaluate a fixed 11-class checkpoint, then merge algae0..4 in its predictions.

The reported coarse metric remaps fine argmax predictions and ground truth;
it is not probability-sum inference, a binary algae task, or a retrained model.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'third_party')]
import torch
from torch.utils.data import DataLoader
from goose_semseg.data.spectral_tiles import SpectralTileDataset
from goose_semseg.data.labeling_taxonomy import (
    FINE_NAMES,COARSE_NAMES,FINE_TO_COARSE,fine_to_coarse_mapping,aggregate_confusion,remap_mask,
)
from goose_semseg.models.backbone.loader import load_dinov3_backbone
from goose_semseg.models.builder import build_segmentation_decoder
from goose_semseg.pretrained.mask2former import load_pretrained_mask2former,infer_pretrained_feature_channels
from goose_semseg.models.head_utils import mask2former_semantic_scores
from goose_semseg.utils.metrics import update_confusion_matrix,compute_mean_iou,per_class_metric_rows
from goose_semseg.utils.seed import seed_everything


def atomic_json(path,data):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False));temporary.replace(path)


def metric(cm,names):
    rows=per_class_metric_rows(cm,0)
    for row in rows:
        row['class_name']=names[row['class_id']]
        row.pop('epoch',None)
        for k,v in row.items():
            if isinstance(v,float) and not math.isfinite(v):row[k]=None
    return dict(miou=compute_mean_iou(cm,gt_present_only=True),
                supported_class_ids=torch.where(cm.sum(1)>0)[0].tolist(),
                per_class=rows,confusion_matrix=cm.tolist(),valid_pixels=int(cm.sum()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--num-workers',type=int,default=4)
    args=p.parse_args()
    os.chdir(ROOT)
    checkpoint=Path(args.checkpoint).resolve();output=Path(args.output).resolve()
    if output.exists():raise FileExistsError(f'Refusing to overwrite completed evaluation: {output}')
    output.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();started_utc=datetime.now(timezone.utc).isoformat()
    assert torch.cuda.is_available() and args.batch_size>0
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    training=argparse.Namespace(**payload['args'])
    assert training.num_classes==11 and training.source_tiles
    manifest=ROOT/training.data_path/'manifest.json'
    manifest_hash=hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert (checkpoint.parent/'dataset_manifest_sha256.txt').read_text().strip()==manifest_hash
    ds=SpectralTileDataset(ROOT/training.data_path,'test',training.tile_size)
    fine_to_coarse_mapping(ds.manifest['class_names'])
    seed_everything(training.seed)
    backbone=load_dinov3_backbone(training)
    pretrained=load_pretrained_mask2former(training.mask2former_pretrained_model_name_or_path)
    model=build_segmentation_decoder(backbone,decoder_type='m2f',hidden_dim=training.hidden_dim,
        num_classes=training.num_classes,autocast_dtype=torch.bfloat16,
        freeze_backbone=training.freeze_backbone,
        feature_channels=infer_pretrained_feature_channels(pretrained),
        input_channels=training.input_channels,fusion_type=training.fusion_type,
        precomputed_indices=True,cls_aux_num_classes=training.cls_aux_num_classes if training.enable_cls_aux else 0)
    del pretrained
    model.load_state_dict(payload['model_state_dict'],strict=True)
    selected_epoch=int(payload['epoch'])+1
    del payload
    model=model.cuda().eval()
    loader=DataLoader(ds,batch_size=args.batch_size,num_workers=args.num_workers,
                      pin_memory=True,**({'prefetch_factor':1} if args.num_workers else {}))
    matrices={s:torch.zeros(11,11,dtype=torch.int64,device='cuda') for s in ('overall','Altum','P1')}
    direct_coarse=torch.zeros(7,7,dtype=torch.int64,device='cuda')
    offset=0
    progress_path=output.with_name(output.stem+'_progress.json')
    print('TEST START',len(ds.records),'frames',len(ds),'tiles',flush=True)
    with torch.inference_mode():
        for x,y in loader:
            x,y=x.cuda(non_blocking=True),y.cuda(non_blocking=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                predictions=mask2former_semantic_scores(model(x),target_size=y.shape[-2:]).argmax(1)
            update_confusion_matrix(matrices['overall'],predictions,y,11,255)
            # Independent pixel remapping verifies the confusion-matrix aggregation.
            update_confusion_matrix(direct_coarse,remap_mask(predictions),remap_mask(y),7,255)
            for j in range(len(y)):
                record=ds.records[ds.tiles[offset+j][0]]
                update_confusion_matrix(matrices[record['sensor']],predictions[j],y[j],11,255)
            offset+=len(y)
            if offset%128==0 or offset==len(ds):
                elapsed=time.monotonic()-started
                progress=dict(status='running',tiles_done=offset,total_tiles=len(ds),
                    elapsed_seconds=elapsed,estimated_remaining_seconds=elapsed/offset*(len(ds)-offset))
                atomic_json(progress_path,progress)
                print(json.dumps(progress),flush=True)
    assert offset==len(ds)
    assert torch.equal(matrices['overall'],matrices['Altum']+matrices['P1'])
    assert torch.equal(direct_coarse,aggregate_confusion(matrices['overall']))
    fine={s:metric(cm.cpu(),FINE_NAMES) for s,cm in matrices.items()}
    coarse={s:metric(aggregate_confusion(cm).cpu(),COARSE_NAMES) for s,cm in matrices.items()}
    assert coarse['overall']['supported_class_ids']==list(range(7))
    result=dict(status='completed',started_utc=started_utc,completed_utc=datetime.now(timezone.utc).isoformat(),
        checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_epoch=selected_epoch,
        checkpoint_selection='Selected by best fine-class validation mIoU before test evaluation',
        manifest_sha256=manifest_hash,split='test',test_opened=True,frames=len(ds.records),tiles=len(ds),
        tta=False,train_args=vars(training),evaluation_batch_size=args.batch_size,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        fine_to_coarse=FINE_TO_COARSE,
        definition='Remap 11-class argmax predictions and GT to 7 groups; algae0..4 merge, nps_algae stays separate. '
                   'No probability summation before argmax. mIoU averages GT-supported groups.',
        verification=dict(pixel_remap_matches_aggregated_confusion=True,sensor_totals_match=True),
        fine_metrics=fine,coarse_metrics=coarse,elapsed_seconds=time.monotonic()-started,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    atomic_json(output,result)
    atomic_json(progress_path,dict(status='completed',tiles_done=offset,total_tiles=len(ds),output=str(output)))
    print('TEST COMPLETE coarse mIoU',coarse['overall']['miou'],
          'algae IoU',coarse['overall']['per_class'][4]['iou'],flush=True)


if __name__=='__main__':main()
