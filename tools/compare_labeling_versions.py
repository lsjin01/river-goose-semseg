#!/usr/bin/env python3
"""Compare actual extracted dataset contents, including shared-file SHA256."""
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    old=Path('Labeling_Data').resolve();new=Path('Labeling_Data_v2').resolve()
    a={str(p.relative_to(old)):p for p in old.rglob('*') if p.is_file()}
    b={str(p.relative_to(new)):p for p in new.rglob('*') if p.is_file() and p.name!='extraction_complete.json'}
    shared=sorted(a.keys()&b.keys());results=[]
    def compare(name):
        x=sha(a[name]);y=sha(b[name]);return dict(path=name,old_sha256=x,new_sha256=y,identical=x==y)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,r in enumerate(pool.map(compare,shared),1):
            results.append(r)
            if i%100==0:print(json.dumps(dict(checked=i,total=len(shared))),flush=True)
    def summary(paths):return dict(Counter(Path(p).suffix.lower() for p in paths))
    missing=sorted(a.keys()-b.keys());added=sorted(b.keys()-a.keys())
    changed=[r['path'] for r in results if not r['identical']]
    result=dict(status='completed',old_source=str(old),new_source=str(new),
        old_files=len(a),new_files=len(b),old_extensions=summary(a),new_extensions=summary(b),
        missing=missing,added=added,added_extensions=summary(added),changed=changed,
        shared_files_sha256=results,
        all_original_images_and_annotations_identical=not any(
            Path(p).suffix.lower() in ('.jpg','.tif','.json') for p in missing+changed))
    out=Path('outputs/dataset_comparison_labeling_v2');out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(result,indent=2))
    lines=['# Labeling_Data (1) 기존 데이터 포함 여부','',
        '- 기존 실제 폴더 `Labeling_Data`와 새 ZIP을 별도로 푼 `Labeling_Data_v2` 비교.',
        '- 공통 파일 모두 SHA256 대조. 파일명만 비교한 결과가 아님.',
        f'- 기존 파일 {len(a)}개, 새 파일 {len(b)}개, 누락 {len(missing)}개.',
        f'- 내용 변경: {", ".join(changed) if changed else "없음"}.',
        f'- 기존 이미지·라벨 모두 동일: {result["all_original_images_and_annotations_identical"]}.','',
        '| 항목 | 기존 | 새 데이터 | 추가 |','|---|---:|---:|---:|']
    for ext,label in [('.tif','Altum 프레임'),('.jpg','P1 프레임'),('.json','라벨 JSON/task')]:
        lines.append(f'| {label} | {summary(a).get(ext,0)} | {summary(b).get(ext,0)} | {summary(added).get(ext,0)} |')
    lines.extend(['','새 데이터만 사용하며, 기존 폴더를 다시 합쳐 중복 프레임을 만들지 않는다.',
                  '58개 task 중 3개는 category ID 체계가 다르므로 이름 기준 라벨 매핑이 필요하다.'])
    (out/'README.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in result.items() if k not in ('shared_files_sha256','added')},indent=2),flush=True)


if __name__=='__main__':main()
