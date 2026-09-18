#!/usr/bin/env python3
"""Full-model smoke tests and frozen-test evaluation for fusion v2."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'third_party')]
import torch
from torch.utils.data import DataLoader
from goose_semseg.data.spectral_tiles import SpectralTileDataset
from goose_semseg.models.backbone.loader import load_dinov3_backbone
from goose_semseg.models.builder import build_segmentation_decoder
from goose_semseg.optim.optimizer import build_optimizer
from goose_semseg.pretrained.mask2former import (
    load_pretrained_mask2former, infer_pretrained_feature_channels, initialize_head_from_pretrained,
)
from goose_semseg.losses.m2f_criterion import Mask2FormerSetCriterion
from goose_semseg.models.head_utils import mask2former_semantic_scores
from goose_semseg.utils.metrics import update_confusion_matrix, compute_mean_iou, per_class_metric_rows
from goose_semseg.utils.seed import seed_everything
from train import parse_args
from goose_semseg.utils.samples import _resolve_cls_aux_output_dim
from goose_semseg.data.coarse_labels import build_batch_cls_aux_targets


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--evaluate', action='store_true')
    options = p.parse_args()
    sys.argv = ['train.py', '--config', options.config]
    args = parse_args()
    dataset_class = SpectralTileDataset
    if args.segmentation_taxonomy == 'binary_algae':
        from goose_semseg.data.binary_algae import BinaryAlgaeTileDataset
        assert args.num_classes == 2 and args.cls_aux_target_type == 'fine'
        dataset_class = BinaryAlgaeTileDataset
    args.cls_aux_num_classes = _resolve_cls_aux_output_dim(target_type=args.cls_aux_target_type,
        num_classes=args.num_classes, taxonomy=args.cls_aux_taxonomy)
    mapping = None
    if args.enable_cls_aux:
        assert args.cls_aux_loss_type == 'bce', 'Smoke test supports unweighted BCE auxiliary'
        if args.cls_aux_target_type == 'coarse':
            assert args.cls_aux_taxonomy == 'labeling_data'
            from goose_semseg.data.labeling_taxonomy import fine_to_coarse_mapping
            manifest = json.loads((Path(args.data_path)/'manifest.json').read_text())
            mapping = fine_to_coarse_mapping(manifest['class_names'])
    seed_everything(args.seed)
    backbone = load_dinov3_backbone(args)
    pretrained = load_pretrained_mask2former(args.mask2former_pretrained_model_name_or_path)
    model = build_segmentation_decoder(backbone, decoder_type='m2f', hidden_dim=args.hidden_dim,
        num_classes=args.num_classes, autocast_dtype=torch.bfloat16, freeze_backbone=False,
        feature_channels=infer_pretrained_feature_channels(pretrained), input_channels=args.input_channels,
        fusion_type=args.fusion_type, precomputed_indices=True,
        cls_aux_num_classes=args.cls_aux_num_classes if args.enable_cls_aux else 0).cuda()
    initialize_head_from_pretrained(model, pretrained)
    del pretrained
    output = Path(args.output_dir)/args.run_name
    if options.evaluate:
        checkpoint = next(output.glob('best_epoch_*.pt'))
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(payload['model_state_dict'], strict=True)
        del payload
    optimizer = build_optimizer(model, args.lr, args.encoder_lr, args.weight_decay)
    # Assert that wrapping a DINO adapter never changes the optimizer group.
    from goose_semseg.models.backbone.adapter import DINOv3_Adapter
    vit = {id(p) for m in model.modules() if isinstance(m,DINOv3_Adapter) for p in m.backbone.parameters()}
    for group in optimizer.param_groups:
        assert all(group['lr'] == (args.encoder_lr if id(p) in vit else args.lr) for p in group['params'])
    parameter_counts = [sum(p.numel() for p in g['params']) for g in optimizer.param_groups]
    model = torch.nn.DataParallel(model)
    if not options.evaluate:
        ds = dataset_class(args.data_path,'train',args.tile_size,0,args.class_sampling_prob,
                                 args.class_sampling_mode)
        indices = [next(i for i,r in enumerate(ds.records) if r['sensor']==s) for s in ('Altum','P1')]
        batch = [ds[indices[i % len(indices)]] for i in range(args.batch_size)]
        x,y = (torch.stack([b[k] for b in batch]).cuda() for k in (0,1))
        assert torch.isfinite(x).all()
        assert set(torch.unique(y).tolist()) <= set(range(args.num_classes)) | {255}
        criterion = Mask2FormerSetCriterion(num_classes=args.num_classes).cuda()
        model.train()
        aux_report = None
        with torch.autocast('cuda',dtype=torch.bfloat16):
            pred = model(x)
            loss,_ = criterion(pred,y)
            if args.enable_cls_aux:
                targets = build_batch_cls_aux_targets(y,target_type=args.cls_aux_target_type,
                    num_classes=args.num_classes,num_coarse=args.cls_aux_num_classes,
                    fine_to_coarse=mapping)
                assert pred['cls_logits'].shape == targets.shape
                aux_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred['cls_logits'].float(), targets)
                loss = loss + args.cls_aux_weight * aux_loss
                aux_report = dict(logits_shape=list(pred['cls_logits'].shape), targets=targets.tolist(),
                    raw_bce=float(aux_loss.detach()),weight=args.cls_aux_weight)
        if args.enable_cls_aux:
            from goose_semseg.models.builder import SpectralInputAdapter
            input_weight = next(m.rgb_projection.weight for m in model.modules() if isinstance(m,SpectralInputAdapter))
            # Exercise production backward, including DataParallel's replica reduction.
            aux_loss.backward(retain_graph=True)
            gradients = (model.module.cls_head.fc.weight.grad, input_weight.grad)
            aux_report['head_and_input_adapter_gradient_norms'] = [float(g.norm()) for g in gradients]
            vit_grad_norm = sum(float(p.grad.detach().float().square().sum())
                                for m in model.modules() if isinstance(m,DINOv3_Adapter)
                                for p in m.backbone.parameters() if p.grad is not None)**.5
            aux_report['vit_gradient_norm'] = vit_grad_norm
            print('CLS AUX CHECK',json.dumps(aux_report),flush=True)
            assert all(torch.isfinite(g).all() and g.abs().sum()>0 for g in gradients),aux_report
            assert vit_grad_norm > 0, 'CLS auxiliary must reach the unfrozen ViT'
            optimizer.zero_grad(set_to_none=True)
        assert torch.isfinite(loss), loss
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters()
                   if 'spectral' in n or 'rgb_projection' in n or 'projections' in n or 'cross_scale' in n)
        optimizer.step()
        diagnostics = None
        if args.val_group_metrics:
            from goose_semseg.engine.trainer import run_epoch
            val_ds = dataset_class(args.data_path,'val',args.tile_size)
            val_ds.tiles = [next(t for t in val_ds.tiles if val_ds.records[t[0]]['sensor']==s)
                            for s in ('Altum','P1')]
            groups = {}
            result = run_epoch(model=model, loader=DataLoader(val_ds,batch_size=2),
                criterion=criterion, optimizer=None, scaler=None, device=torch.device('cuda'),
                amp=True, grad_clip_norm=35, grad_accum_steps=1, num_classes=args.num_classes,
                ignore_index=255, epoch=0, epochs=1, enable_cls_aux=args.enable_cls_aux,
                cls_aux_target_type=args.cls_aux_target_type, cls_aux_loss_type='bce', cls_aux_weight=args.cls_aux_weight,
                cls_aux_num_classes=args.cls_aux_num_classes, cls_aux_pos_weight=None, gt_present_only=True,
                cls_aux_fine_to_coarse=mapping,
                group_confusion_matrices=groups)
            assert torch.equal(groups['sensor/Altum']+groups['sensor/P1'],result[-1])
            diagnostics = {k:int(v.sum()) for k,v in groups.items()}
            if args.enable_cls_aux:
                assert result[2]['loss_cls_aux'] > 0
                aux_report['validation_weighted_aux_loss'] = result[2]['loss_cls_aux']
                aux_report['validation_binary_accuracy'] = result[3]['cls_aux_accuracy']
        dest = Path(args.output_dir).with_name(Path(args.output_dir).name+'_checks')
        dest.mkdir(parents=True,exist_ok=True)
        (dest/(args.run_name+'.json')).write_text(json.dumps(dict(
            status='passed', loss=float(loss.detach()), batch_shape=list(x.shape),
            num_classes=args.num_classes,segmentation_taxonomy=args.segmentation_taxonomy,
            optimizer_group_parameters=parameter_counts, input_channels=9,
            validation_group_pixel_counts=diagnostics,
            cls_aux=aux_report,config_sha256=hashlib.sha256(Path(options.config).read_bytes()).hexdigest(),
            peak_memory_bytes=[torch.cuda.max_memory_allocated(i)
                               for i in range(torch.cuda.device_count())]),indent=2))
        print('SMOKE PASSED',args.run_name,float(loss.detach()),flush=True)
        return
    del optimizer
    ds = dataset_class(args.data_path,'test',args.tile_size)
    loader = DataLoader(ds,batch_size=args.val_batch_size,num_workers=args.num_workers,pin_memory=True)
    matrices = {s:torch.zeros(args.num_classes,args.num_classes,dtype=torch.int64,device='cuda') for s in ('overall','Altum','P1')}
    model.eval()
    offset = 0
    with torch.inference_mode():
        for x,y in loader:
            x,y = x.cuda(),y.cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                outputs = model(x)
                predicted = mask2former_semantic_scores(outputs,target_size=y.shape[-2:]).argmax(1)
            update_confusion_matrix(matrices['overall'],predicted,y,args.num_classes,255)
            for j in range(len(y)):
                rec = ds.records[ds.tiles[offset+j][0]]
                update_confusion_matrix(matrices[rec['sensor']],predicted[j],y[j],args.num_classes,255)
            offset += len(y)
            if offset % 100 == 0:
                print('test tiles',offset,'/',len(ds),flush=True)
    result = dict(checkpoint=str(checkpoint),tta=False, tiles=len(ds),
                  metric_definition='mIoU over GT-supported classes; same class set for all models on each split',
                  metrics={s:dict(miou=compute_mean_iou(cm,gt_present_only=True),
                     union_class_miou=compute_mean_iou(cm),
                     supported_classes=torch.where(cm.sum(1)>0)[0].cpu().tolist(),
                     per_class=per_class_metric_rows(cm,0),confusion_matrix=cm.cpu().tolist())
                     for s,cm in matrices.items()})
    for metrics in result['metrics'].values():
        for row in metrics['per_class']:
            row['class_name']=ds.manifest['class_names'][row['class_id']]
    (output/'test_no_tta.json').write_text(json.dumps(result,indent=2))
    print('TEST COMPLETE',args.run_name,result['metrics']['overall']['miou'],flush=True)


if __name__=='__main__':
    main()
