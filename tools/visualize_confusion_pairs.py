#!/usr/bin/env python3
"""Visualize directed mistakes from saved test masks; no new model inference."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

from visualize_grouped_test import ROOT, COLORS, COARSE_NAMES, source_preview

PAIRS = [(6,3),(2,3),(1,0),(4,5),(5,3),(3,0)]


def dense_crop(mask):
    """Find the square window with most directed-error pixels in the preview."""
    h,w=mask.shape
    side=min(h,w,max(160,round(min(h,w)*.42)))
    integral=np.pad(mask.astype(np.int64),((1,0),(1,0))).cumsum(0).cumsum(1)
    counts=integral[side:,side:]-integral[:-side,side:]-integral[side:,:-side]+integral[:-side,:-side]
    y,x=np.unravel_index(counts.argmax(),counts.shape)
    assert counts[y,x]>0
    return int(x),int(y),int(x+side),int(y+side)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report-dir',required=True)
    p.add_argument('--output-dir',required=True)
    args=p.parse_args()
    root=Path(args.report_dir).resolve();out=Path(args.output_dir).resolve()
    if (out/'report.json').exists():raise FileExistsError('Completed visualization exists')
    out.mkdir(parents=True,exist_ok=True)
    report=json.loads((root/'report.json').read_text())
    evaluation=json.loads((ROOT/report['evaluation']).read_text())
    path=ROOT/evaluation['train_args']['data_path']/'manifest.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==report['manifest_sha256']
    manifest=json.loads(path.read_text())
    records=[r for r in manifest['records'] if r['split']=='test']
    ds=SimpleNamespace(source=Path(manifest['source']))
    palette=np.zeros((256,3),dtype=np.uint8);palette[:7]=COLORS
    selected=[]
    for truth,prediction in PAIRS:
        candidates=sorted(report['frame_metrics'],key=lambda e:-e['confusion_matrix'][truth][prediction])
        entry=candidates[0];idx=entry['frame_index'];record=records[idx]
        assert record['source_image']==entry['source_image']
        gt=np.array(Image.open(root/'previews'/f'{idx:03d}_gt.png'))
        pred=np.array(Image.open(root/'previews'/f'{idx:03d}_pred.png'))
        assert gt.shape==pred.shape
        error=(gt==truth)&(pred==prediction)
        box=dense_crop(error);x0,y0,x1,y1=box
        rgb,source_title=source_preview(ds,record,(gt.shape[1],gt.shape[0]))
        gc=palette[gt];pc=palette[pred].copy();pc[gt==255]=0
        highlight=(rgb.astype(np.float32)*.4).astype(np.uint8)
        highlight[error]=[255,30,40];highlight[gt==255]=0
        count=entry['confusion_matrix'][truth][prediction]
        rate=100*count/sum(entry['confusion_matrix'][truth])
        names=f'{COARSE_NAMES[truth]} -> {COARSE_NAMES[prediction]}'
        fig,axes=plt.subplots(2,4,figsize=(18,9))
        arrays=[rgb,gc,pc,highlight]
        titles=[source_title,'Ground truth','Prediction',f'Only {names}: RED']
        for j,(arr,title) in enumerate(zip(arrays,titles)):
            axes[0,j].imshow(arr,interpolation='nearest');axes[0,j].set_title(title,fontsize=10)
            axes[0,j].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='yellow',linewidth=1.8))
            axes[1,j].imshow(arr[y0:y1,x0:x1],interpolation='nearest')
            axes[1,j].set_title('Selected-region zoom',fontsize=10)
        for ax in axes.flat:ax.axis('off')
        fig.suptitle(f'{names} | {record["sensor"]}/{record["task"]}/{Path(record["source_image"]).name}\n'
                     f'{count:,} full-resolution error pixels | {rate:.2f}% of this frame\'s true '
                     f'{COARSE_NAMES[truth]} pixels | No TTA',fontsize=14)
        fig.legend(handles=[Patch(color=c/255,label=n) for c,n in zip(COLORS,COARSE_NAMES)],
                   loc='lower center',ncol=7,frameon=False)
        fig.text(.5,.043,'Yellow box: zoom region | Red: this directed error only | '
                 'Saved reduced masks; zoom does not restore original mask resolution',ha='center',fontsize=9)
        fig.subplots_adjust(left=.01,right=.99,bottom=.09,top=.87,wspace=.04,hspace=.14)
        name=f'{COARSE_NAMES[truth]}_to_{COARSE_NAMES[prediction]}_frame_{idx:03d}.png'
        fig.savefig(out/name,dpi=150);plt.close(fig)
        selected.append(dict(ground_truth=COARSE_NAMES[truth],prediction=COARSE_NAMES[prediction],
            source_image=record['source_image'],frame_index=idx,error_pixels_full_resolution=count,
            percent_of_frame_gt=rate,zoom_box_preview_xyxy=box,
            preview_size_wh=[gt.shape[1],gt.shape[0]],image=name))
        print(json.dumps(selected[-1]),flush=True)
    result=dict(status='completed',checkpoint=report['checkpoint'],manifest_sha256=report['manifest_sha256'],
        tta=False,inference_rerun=False,source_report=str(root/'report.json'),
        selection='For each directed pair, select the test frame with most full-resolution error pixels; '
                  'zoom into the preview square containing most such errors.',
        visualization='Saved nearest-neighbor reduced masks; no original-resolution mask reconstruction. '
                      'NPS labels kept verbatim; their annotation semantics are not verified.',cases=selected)
    (out/'report.json').write_text(json.dumps(result,indent=2))
    lines=['# 오분류 유형별 확대 시각화','',
        '- 기존 test 예측 사용. 재학습·재추론·TTA 없음. algae0~4 통합, nps_algae 별도.',
        '- 각 정답 → 예측 조합에서 원본 해상도 오분류 픽셀 수가 가장 많은 프레임을 선택.',
        '- 프레임 안에서는 해당 오류가 가장 많이 포함되는 사각형을 확대.',
        '- 위: 전체 장면 / 아래: 노란 상자 확대. 열: 원본 / 정답 / 예측 / 해당 오류만 빨강.',
        '- 확대는 저장된 축소 마스크 기준이며 원본 해상도 복원 결과가 아님. 수치는 원본 해상도 혼동행렬 기준.',
        '- 이미지의 비율은 해당 프레임 정답 클래스 픽셀 기준이며, 전체 test의 혼동 비율과 다름.',
        '- nps와 nps_algae는 원본 라벨명을 그대로 표기하며 실제 라벨링 의미는 미확인.','']
    for e in selected:
        lines.extend([f'## {e["ground_truth"]} → {e["prediction"]}','',
            f'`{e["source_image"]}` — {e["error_pixels_full_resolution"]:,}픽셀, '
            f'해당 프레임 정답 클래스의 {e["percent_of_frame_gt"]:.2f}%.','',f'![오분류 확대]({e["image"]})',''])
    (out/'README.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
