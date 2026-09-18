import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'third_party')]
from goose_semseg.data.spectral_tiles import SpectralTileDataset
from goose_semseg.models.builder import SpectralInputAdapter, SpectralFeatureFusion
from goose_semseg.models.backbone.adapter import DINOv3_Adapter
from goose_semseg.optim.optimizer import build_optimizer
from goose_semseg.utils.metrics import compute_mean_iou


class SpectralTests(unittest.TestCase):
    def test_unsupported_class_does_not_change_denominator(self):
        cm=torch.tensor([[9,1,0],[1,9,0],[0,0,0]])
        self.assertAlmostEqual(compute_mean_iou(cm,True),9/11,places=6)
        cm[0,0]-=1; cm[0,2]+=1
        self.assertAlmostEqual(compute_mean_iou(cm,True),(8/11+9/11)/2,places=6)
    def test_indices_have_learnable_effect(self):
        a=SpectralInputAdapter(9,precomputed_indices=True)
        b=SpectralInputAdapter(9,add_indices=True,precomputed_indices=True)
        self.assertEqual(a.rgb_projection.in_channels,7)
        self.assertEqual(b.rgb_projection.in_channels,9)
        x=torch.rand(2,9,8,8)
        with torch.no_grad(): b.rgb_projection.weight[:,7:]=1
        y=x.clone(); y[:,7:]=0
        self.assertTrue(torch.equal(a(x),a(y)))
        self.assertFalse(torch.equal(b(x),b(y)))
        b(x).sum().backward()
        self.assertGreater(float(b.rgb_projection.weight.grad[:,7:].abs().sum()),0)

    def test_optimizer_wrapping(self):
        # Minimal adapter object: exercises exact production type and nested parameter ownership.
        d=DINOv3_Adapter.__new__(DINOv3_Adapter)
        torch.nn.Module.__init__(d)
        d.backbone=torch.nn.Linear(2,2)
        d.spatial_prior=torch.nn.Linear(2,2)
        wrapper=torch.nn.Module(); wrapper.backbone=d
        for model in (d,wrapper):
            opt=build_optimizer(model,4e-5,1e-6,.01)
            lrs={id(p):g['lr'] for g in opt.param_groups for p in g['params']}
            self.assertEqual(lrs[id(d.backbone.weight)],1e-6)
            self.assertEqual(lrs[id(d.spatial_prior.weight)],4e-5)

    def test_tiles_cover_valid_pixels_once_and_keep_pages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            pages=[Image.fromarray(np.full((11,13),1000*(i+1),np.uint16)) for i in range(7)]
            pages[0].save(root/'x.tif',save_all=True,append_images=pages[1:])
            record=dict(sensor='Altum',width=13,height=11,source_image='x.tif',
                split='val',classes_present=[8],annotations=[dict(category_id=9,
                    area=143,segmentation=[[0,0,12,0,12,10,0,10]])])
            (root/'manifest.json').write_text(json.dumps(dict(source=str(root),
                normalization={'Altum':dict(low=[0]*7,scale=[10000]*7)},records=[record])))
            ds=SpectralTileDataset(root,'val',8)
            pixels=0
            for i in range(len(ds)):
                x,y=ds[i]; pixels+=int((y!=255).sum())
                self.assertTrue(torch.isfinite(x).all())
                self.assertAlmostEqual(float(x[6,0,0]),float(np.arcsinh(.7)),places=5)
                self.assertAlmostEqual(float(x[7,0,0]),1/7,places=5)
                self.assertAlmostEqual(float(x[8,0,0]),-1/9,places=5)
                self.assertTrue(torch.equal(ds[i][0],x))
            self.assertEqual(pixels,143)


if __name__=='__main__':
    unittest.main()
