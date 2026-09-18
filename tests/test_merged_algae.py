import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from goose_semseg.data.merged_algae import CLASS_NAMES, source_name_to_label
from goose_semseg.data.spectral_tiles import SpectralTileDataset


class MergedAlgaeTests(unittest.TestCase):
    def test_name_mapping(self):
        self.assertEqual(
            [source_name_to_label(name) for name in CLASS_NAMES[:-1]],
            list(range(6)),
        )
        for name in ('algae', 'algae0', 'algae1', 'algae2', 'algae3', 'algae4', 'nps_algae'):
            self.assertEqual(source_name_to_label(name), 6)
        self.assertEqual(source_name_to_label('ambiguous'), 255)
        with self.assertRaises(ValueError):
            source_name_to_label('unknown')

    def test_generic_native_manifest_rasterizes_seven_classes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotations = [
                dict(category_id=100 + label_id, label_id=label_id, area=2,
                     segmentation=[[label_id * 2, 0, label_id * 2 + 1, 0,
                                    label_id * 2 + 1, 1, label_id * 2, 1]])
                for label_id in range(7)
            ]
            annotations.append(dict(category_id=999, label_id=255, area=1,
                                    segmentation=[[14, 0, 15, 0, 15, 1, 14, 1]]))
            record = dict(sensor='P1', width=16, height=3, source_image='fake.jpg',
                          split='train', classes_present=list(range(7)), annotations=annotations)
            manifest = dict(
                source=temporary,
                class_names=list(CLASS_NAMES),
                annotation_label_key='label_id',
                records=[record],
                normalization={'P1': dict(low=[0] * 7, scale=[65535] * 7)},
            )
            (root / 'manifest.json').write_text(json.dumps(manifest))
            dataset = SpectralTileDataset(root, 'train', 32, 0, 0)
            mask = dataset._mask(0)
            self.assertEqual(dataset.num_classes, 7)
            self.assertEqual(set(dataset.class_members), set(range(7)))
            self.assertEqual([int(mask[0, i * 2]) for i in range(7)], list(range(7)))
            self.assertEqual(int(mask[0, 14]), 255)
            dataset._source_pixels = Mock(return_value=np.zeros((3, 16, 3), dtype=np.uint8))
            _, target = dataset[0]
            self.assertEqual(set(target.unique().tolist()), set(range(7)) | {255})


if __name__ == '__main__':
    unittest.main()
