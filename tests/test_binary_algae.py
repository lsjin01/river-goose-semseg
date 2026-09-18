import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import torch

from goose_semseg.data.binary_algae import BinaryAlgaeTileDataset, FINE_TO_BINARY, remap_binary
from goose_semseg.data.spectral_tiles import SpectralTileDataset
from goose_semseg.data.labeling_taxonomy import FINE_NAMES
from goose_semseg.data.coarse_labels import build_batch_cls_aux_targets


class BinaryAlgaeTests(unittest.TestCase):
    def test_mapping_ignore_and_no_mutation(self):
        mask=np.array([list(range(11))+[255]],dtype=np.uint8)
        original=mask.copy()
        self.assertEqual(remap_binary(mask).tolist(),[[0,0,0,0,1,1,1,1,1,0,1,255]])
        np.testing.assert_array_equal(mask,original)
        for bad in [np.array([11]),np.array([-1]),np.array([1.0])]:
            with self.assertRaises(ValueError):remap_binary(bad)

    def test_source_raster_and_metadata_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            anns=[dict(category_id=i+1+int(i>=10),area=2,
                       segmentation=[[i*2,0,i*2+1,0,i*2+1,1,i*2,1]]) for i in range(11)]
            record=dict(sensor='P1',width=22,height=3,source_image='fake.jpg',split='train',
                        classes_present=list(range(11)),annotations=anns,group='P1/test',task='test')
            manifest=dict(source=tmp,class_names=list(FINE_NAMES),records=[record],
                          normalization={'P1':dict(low=[0]*7,scale=[65535]*7)})
            path=root/'manifest.json';path.write_text(json.dumps(manifest));before=path.read_bytes()
            original=SpectralTileDataset(root,'train',32,0,0)
            binary=BinaryAlgaeTileDataset(root,'train',32,0,0)
            np.testing.assert_array_equal(binary._mask(0),remap_binary(original._mask(0)))
            self.assertEqual(original.manifest['class_names'],list(FINE_NAMES))
            self.assertEqual(binary.records[0]['classes_present'],[0,1])
            self.assertEqual(set(binary.class_members),{0,1})
            self.assertEqual(path.read_bytes(),before)
            binary._source_pixels=Mock(return_value=np.zeros((3,22,3),dtype=np.uint8))
            x,y=binary[0]
            self.assertEqual(tuple(x.shape),(9,32,32))
            self.assertEqual(set(y.unique().tolist()),{0,1,255})
            self.assertTrue(torch.all(y[3:]==255))
            self.assertTrue(torch.all(x[3:]==0))

    def test_two_class_presence_includes_zero(self):
        masks=torch.tensor([[[0,255]],[[1,255]],[[0,1]]])
        targets=build_batch_cls_aux_targets(masks,target_type='fine',num_classes=2)
        self.assertEqual(targets.tolist(),[[1,0],[0,1],[1,1]])

    def test_native_labels_ignore_raw_category_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            anns=[dict(category_id=99,label_id=1,area=16,segmentation=[[0,0,3,0,3,3,0,3]]),
                  dict(category_id=5,label_id=0,area=4,segmentation=[[0,0,1,0,1,1,0,1]]),
                  dict(category_id=13,label_id=255,area=1,segmentation=[[3,3,4,3,4,4,3,4]])]
            r=dict(sensor='P1',width=5,height=5,source_image='fake.jpg',split='val',
                   classes_present=[0,1],annotations=anns)
            m=dict(source=tmp,class_names=['non_algae','algae_including_nps_algae'],
                   annotation_label_key='label_id',records=[r],normalization={})
            p=root/'manifest.json';p.write_text(json.dumps(m));before=p.read_bytes()
            ds=BinaryAlgaeTileDataset(root,'val',8)
            mask=ds._mask(0)
            self.assertEqual(int(mask[0,0]),0)
            self.assertEqual(int(mask[2,2]),1)
            self.assertEqual(int(mask[3,3]),255)
            self.assertEqual(int(mask[4,0]),255)
            self.assertEqual(ds.class_members,{0:[0],1:[0]})
            self.assertEqual(p.read_bytes(),before)


if __name__=='__main__':unittest.main()
