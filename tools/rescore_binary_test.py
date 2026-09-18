#!/usr/bin/env python3
"""Rescore fixed fine-argmax test predictions as binary; no training or inference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from visualize_grouped_test import ROOT, FINE_NAMES, heatmap

POSITIVE = ('algae0','algae1','algae2','algae3','algae4','nps_algae')
NAMES = ('non_algae', 'algae_including_nps_algae')


def binary_metrics(matrix, mapping):
    fine=np.asarray(matrix,dtype=np.int64)
    assert fine.shape==(11,11) and np.all(fine>=0)
    cm=np.zeros((2,2),dtype=np.int64)
    np.add.at(cm,(mapping[:,None],mapping[None,:]),fine)
    assert cm.sum()==fine.sum()
    rows=[]
    for i,name in enumerate(NAMES):
        tp=int(cm[i,i]);gt=int(cm[i].sum());pred=int(cm[:,i].sum());union=gt+pred-tp
        rows.append(dict(class_id=i,class_name=name,gt_pixels=gt,pred_pixels=pred,tp=tp,
            fp=pred-tp,fn=gt-tp,iou=tp/union if union else None,
            precision=tp/pred if pred else None,recall=tp/gt if gt else None,
            dice=2*tp/(gt+pred) if gt+pred else None))
    supported=[r['iou'] for r in rows if r['gt_pixels']]
    return dict(confusion_matrix=cm.tolist(),per_class=rows,
        miou=float(np.mean(supported)) if supported else None,
        pixel_accuracy=float(np.trace(cm)/cm.sum()) if cm.sum() else None,
        valid_pixels=int(cm.sum()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--frame-report')
    args=p.parse_args()
    source=Path(args.evaluation).resolve();out=Path(args.output_dir).resolve()
    if (out/'report.json').exists():raise FileExistsError('Completed report already exists')
    d=json.loads(source.read_text())
    assert d['status']=='completed' and d['split']=='test' and not d['tta']
    assert tuple(r['class_name'] for r in d['fine_metrics']['overall']['per_class'])==FINE_NAMES
    mapping=np.array([int(name in POSITIVE) for name in FINE_NAMES])
    metrics={key:binary_metrics(value['confusion_matrix'],mapping) for key,value in d['fine_metrics'].items()}
    assert np.array_equal(np.asarray(metrics['overall']['confusion_matrix']),
        np.asarray(metrics['Altum']['confusion_matrix'])+np.asarray(metrics['P1']['confusion_matrix']))
    result=dict(status='completed',source_evaluation=str(source),
        source_evaluation_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        checkpoint=d['checkpoint'],checkpoint_epoch=d['checkpoint_epoch'],
        manifest_sha256=d['manifest_sha256'],frames=d['frames'],tiles=d['tiles'],split='test',
        tta=False,retrained=False,inference_rerun=False,positive_classes=list(POSITIVE),
        negative_classes=[n for n in FINE_NAMES if n not in POSITIVE],
        fine_to_binary=mapping.tolist(),ignore_index=255,
        definition='Remap fixed 11-class argmax predictions and GT. No probability-sum inference, '
            'threshold tuning or retraining. Ignore pixels stay excluded. mIoU uses GT-supported classes.',
        verification=dict(pixel_totals_preserved=True,sensor_totals_match=True),metrics=metrics)
    out.mkdir(parents=True,exist_ok=True)
    if args.frame_report:
        frames=json.loads(Path(args.frame_report).read_text())
        assert frames['taxonomy']=='fine11' and frames['checkpoint']==d['checkpoint']
        assert frames['manifest_sha256']==d['manifest_sha256'] and len(frames['frame_metrics'])==d['frames']
        accumulated=np.zeros((11,11),dtype=np.int64)
        entries=[]
        for e in frames['frame_metrics']:
            accumulated+=np.asarray(e['confusion_matrix'],dtype=np.int64)
            entries.append(dict(frame_index=e['frame_index'],source_image=e['source_image'],
                sensor=e['sensor'],task=e['task'],**binary_metrics(e['confusion_matrix'],mapping)))
        assert np.array_equal(accumulated,np.asarray(d['fine_metrics']['overall']['confusion_matrix']))
        result['frame_metrics']=sorted(entries,key=lambda e:(e['miou'],e['source_image']))
        result['verification']['frame_totals_match']=True
        with (out/'frame_metrics.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['rank_worst_first','source_image','miou','non_algae_iou','algae_iou'])
            for rank,e in enumerate(result['frame_metrics'],1):
                w.writerow([rank,e['source_image'],e['miou'],*[r['iou'] for r in e['per_class']]])
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    overall=metrics['overall']
    with (out/'class_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(overall['per_class'][0]));w.writeheader();w.writerows(overall['per_class'])
    heatmap(overall['confusion_matrix'],['non-algae','algae + nps_algae'],
            out/'confusion_binary.png','Binary test | algae0-4 + nps_algae | No TTA')
    lines=['# Binary test: algae0~4 + nps_algae','',
        '- 양성: algae0, algae1, algae2, algae3, algae4, nps_algae.',
        '- 음성: land, bridge, other, nps, turbid. 평가 제외(255) 픽셀은 음성에 넣지 않음.',
        f'- 체크포인트: `{d["checkpoint"]}`',
        f'- 고정 test {d["frames"]}프레임 / {d["tiles"]:,}타일. TTA 없음.',
        '- 기존 11-class argmax 예측과 정답을 이진화. 재학습·재추론·확률 합산·임계값 조정 없음.',
        '- 서로 다른 세부 클래스 사이의 오류가 통합으로 사라지므로, mIoU 상승은 모델 개선량이 아님.',
        '- algae0도 요청한 그룹에 포함했으며 이것이 생물학적 녹조 존재 판정 기준이라는 의미는 아님.','',
        '| 클래스 | IoU | Precision | Recall |','|---|---:|---:|---:|']
    for r in overall['per_class']:
        lines.append(f'| {r["class_name"]} | {r["iou"]*100:.2f}% | {r["precision"]*100:.2f}% | {r["recall"]*100:.2f}% |')
    lines.extend(['',f'**2-class mIoU: {overall["miou"]*100:.2f}%**','',
        '![Binary confusion](confusion_binary.png)','',
        '[원시 결과 및 클래스 매핑](report.json) · [클래스별 CSV](class_metrics.csv)'])
    if args.frame_report:lines.extend(['','[이미지별 순위 CSV](frame_metrics.csv)'])
    (out/'README.md').write_text('\n'.join(lines))
    print(json.dumps(overall),flush=True)


if __name__=='__main__':main()
