"""Binary training view of the frozen 11-class source masks (no source edits)."""
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw

from goose_semseg.data.labeling_taxonomy import FINE_NAMES, fine_to_coarse_mapping
from goose_semseg.data.spectral_tiles import SpectralTileDataset

BINARY_NAMES = ('non_algae', 'algae_including_nps_algae')
POSITIVE_NAMES = ('algae0', 'algae1', 'algae2', 'algae3', 'algae4', 'nps_algae')
FINE_TO_BINARY = tuple(int(name in POSITIVE_NAMES) for name in FINE_NAMES)


def remap_binary(mask):
    """Preserve ignored pixels and never mutate the original fine mask."""
    if not np.issubdtype(mask.dtype, np.integer):
        raise ValueError('Mask must have integer class IDs')
    valid = mask != 255
    if np.any(valid & ((mask < 0) | (mask >= len(FINE_NAMES)))):
        raise ValueError('Unknown fine class ID')
    out = np.full(mask.shape, 255, dtype=np.uint8)
    out[valid] = np.asarray(FINE_TO_BINARY, dtype=np.uint8)[mask[valid]]
    return out


class BinaryAlgaeTileDataset(SpectralTileDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.native_binary = self.manifest.get('annotation_label_key') == 'label_id'
        if self.native_binary:
            if self.manifest['class_names'] != list(BINARY_NAMES):
                raise ValueError('Native binary manifest class names do not match')
            for record in self.records:
                if not set(record['classes_present']) <= {0,1}:
                    raise ValueError('Unknown binary record class')
                if any(a.get('label_id') not in (0,1,255) for a in record['annotations']):
                    raise ValueError('Missing/invalid binary annotation label')
            self.class_members = {i: [j for j,r in enumerate(self.records) if i in r['classes_present']]
                                  for i in range(2)}
            self.active_classes = [i for i,indices in self.class_members.items() if indices]
            return
        fine_to_coarse_mapping(self.manifest['class_names'])  # Exact source order.
        self.manifest['source_class_names'] = list(FINE_NAMES)
        self.manifest['class_names'] = list(BINARY_NAMES)
        self.manifest['segmentation_taxonomy'] = 'binary_algae'
        # Only the in-memory view changes. COCO annotations retain source IDs.
        for record in self.manifest['records']:
            record['source_classes_present'] = list(record['classes_present'])
            record['classes_present'] = sorted({FINE_TO_BINARY[i] for i in record['classes_present']})
        self.class_members = {i: [j for j,r in enumerate(self.records) if i in r['classes_present']]
                              for i in range(2)}
        self.active_classes = [i for i,indices in self.class_members.items() if indices]

    @lru_cache(maxsize=2)
    def _mask(self, index):
        if self.native_binary:
            r = self.records[index]
            mask = Image.new('L', (r['width'],r['height']), 255)
            draw = ImageDraw.Draw(mask)
            for ann in sorted(r['annotations'], key=lambda a:a.get('area',0), reverse=True):
                for polygon in ann.get('segmentation', []):
                    if len(polygon) >= 6:
                        draw.polygon(list(zip(polygon[::2],polygon[1::2])),fill=ann['label_id'])
            return np.array(mask)
        return remap_binary(super()._mask(index))


def binary_taxonomy_metadata():
    return dict(name='binary_algae', class_names=list(BINARY_NAMES),
                source_class_names=list(FINE_NAMES), fine_to_binary=list(FINE_TO_BINARY),
                positive_names=list(POSITIVE_NAMES), ignore_index=255,
                source_annotations_unchanged=True)
