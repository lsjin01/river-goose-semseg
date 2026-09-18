#!/usr/bin/env python3
"""Derive a frozen seven-class manifest from the verified expanded-data split."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from goose_semseg.data.merged_algae import CLASS_NAMES, source_name_to_label, taxonomy_metadata


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='data/labeling_binary_v2_geo')
    parser.add_argument('--output', default='data/labeling_merged_algae_7class_v2_geo')
    args = parser.parse_args()
    source = Path(args.source)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)

    source_manifest_path = source / 'manifest.json'
    source_verification_path = source / 'verification.json'
    source_manifest = json.loads(source_manifest_path.read_text())
    source_verification = json.loads(source_verification_path.read_text())
    if source_verification['status'] != 'passed':
        raise ValueError('Parent split verification did not pass')
    if source_verification['manifest_sha256'] != digest(source_manifest_path):
        raise ValueError('Parent manifest hash changed')

    distribution = {}
    for split in ('train', 'val', 'test'):
        distribution[split] = {
            name: {'annotations': 0, 'frames': 0, 'polygon_area': 0.0}
            for name in CLASS_NAMES
        }
    frame_presence = defaultdict(set)

    records = source_manifest['records']
    for frame_index, record in enumerate(records):
        present = set()
        for annotation in record['annotations']:
            source_name = annotation['source_category_name']
            label_id = source_name_to_label(source_name)
            annotation['label_id'] = label_id
            if label_id == 255 or not annotation.get('segmentation'):
                continue
            present.add(label_id)
            name = CLASS_NAMES[label_id]
            stats = distribution[record['split']][name]
            stats['annotations'] += 1
            stats['polygon_area'] += float(annotation.get('area', 0.0))
            frame_presence[(record['split'], label_id)].add(frame_index)
        record['classes_present'] = sorted(present)
        if not present:
            raise ValueError(f"Frame lost all supervised classes: {record['source_image']}")

    for split in distribution:
        for label_id, name in enumerate(CLASS_NAMES):
            distribution[split][name]['frames'] = len(frame_presence[(split, label_id)])
            distribution[split][name]['polygon_area'] = round(
                distribution[split][name]['polygon_area']
            )

    manifest = {
        'source': source_manifest['source'],
        'version': 'merged_algae_7class_labeling_v2',
        'seed': source_manifest['seed'],
        'class_names': list(CLASS_NAMES),
        'annotation_label_key': 'label_id',
        'normalization': source_manifest['normalization'],
        'records': records,
        'taxonomy': taxonomy_metadata(),
        'audit': {
            'parent_manifest': str(source_manifest_path),
            'parent_manifest_sha256': digest(source_manifest_path),
            'split_policy': 'Exact reuse of the verified binary v2 group split; labels only were remapped.',
            'splits': source_manifest['audit']['splits'],
            'class_distribution': distribution,
            'limitations': [
                'river has one annotation in one train frame and no validation/test support.',
                *source_manifest['audit']['limitations'],
            ],
        },
    }
    manifest_path = output / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Independently verify that identity and group assignments exactly match the parent split.
    parent_assignments = {
        record['source_image']: (record['split'], record['group'], record['group_id'], record['sha256'])
        for record in source_manifest['records']
    }
    derived = json.loads(manifest_path.read_text())
    assert len(derived['records']) == len(parent_assignments)
    for record in derived['records']:
        assert parent_assignments[record['source_image']] == (
            record['split'], record['group'], record['group_id'], record['sha256']
        )
        assert set(record['classes_present']) <= set(range(len(CLASS_NAMES)))
        assert all(annotation['label_id'] in (*range(len(CLASS_NAMES)), 255)
                   for annotation in record['annotations'])
    assert distribution['train']['river']['frames'] == 1
    assert distribution['val']['river']['frames'] == 0
    assert distribution['test']['river']['frames'] == 0
    for split in ('train', 'val', 'test'):
        required = CLASS_NAMES if split == 'train' else CLASS_NAMES[1:]
        assert all(distribution[split][name]['frames'] > 0 for name in required)

    verification = {
        'status': 'passed',
        'manifest_sha256': digest(manifest_path),
        'parent_manifest_sha256': digest(source_manifest_path),
        'records': len(derived['records']),
        'images': source_verification['images'],
        'task_overlap': source_verification['task_overlap'],
        'exact_hash_overlap': source_verification['exact_hash_overlap'],
        'geographic_group_overlap': source_verification['geographic_group_overlap'],
        'normalization_split': source_verification['normalization_split'],
        'class_distribution': distribution,
        'unsupported_eval_classes': ['river'],
    }
    (output / 'verification.json').write_text(json.dumps(verification, indent=2))
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
