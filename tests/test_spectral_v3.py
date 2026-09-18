import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'third_party')]
from goose_semseg.data.spectral_tiles import SpectralTileDataset
from goose_semseg.engine.trainer import run_epoch


class FakeTiles(Dataset):
    records = [dict(sensor='Altum',group='Altum/a'), dict(sensor='P1',group='P1/b')]
    tiles = [(0,0,0),(1,0,0),(1,2,0)]
    def __len__(self): return len(self.tiles)
    def __getitem__(self, i):
        label = torch.full((2,2),int(i>0),dtype=torch.long)
        return label[None].float(),label


class FakeModel(torch.nn.Module):
    def forward(self, images):
        return torch.nn.functional.one_hot(images[:,0].long(),2).permute(0,3,1,2).float()


class FakeCriterion(torch.nn.Module):
    def forward(self, outputs, labels): return outputs.sum()*0+1, {}


class SpectralV3Tests(unittest.TestCase):
    def test_within_image_does_not_replace_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            records=[dict(sensor='P1',width=8,height=8,source_image=f'{i}.jpg',
                          split='train',classes_present=[i],annotations=[]) for i in (0,1)]
            (root/'manifest.json').write_text(json.dumps(dict(source=tmp,records=records,
                normalization={'P1':dict(low=[0]*7,scale=[65535]*7)})))
            ds=SpectralTileDataset(root,'train',8,0,1,'within_image')
            ds._source_pixels=Mock(return_value=np.full((8,8,3),128,np.uint8))
            ds._mask=Mock(side_effect=lambda index:np.full((8,8),index,np.uint8))
            with patch('goose_semseg.data.spectral_tiles.random.choice',side_effect=lambda x:x[0]):
                x,y=ds[1]
            ds._source_pixels.assert_called_once_with(1)
            self.assertTrue((y==1).all())
            self.assertEqual(tuple(x.shape),(9,8,8))
            self.assertTrue((x[3:]==0).all())
            with self.assertRaises(ValueError): SpectralTileDataset(root,'train',rare_prob=1.1)
            with self.assertRaises(ValueError): SpectralTileDataset(root,'train',class_sampling_mode='bad')

    def test_group_counts_match_global_and_reject_shuffle(self):
        groups={}
        kwargs=dict(model=FakeModel(),criterion=FakeCriterion(),optimizer=None,
            scaler=None,device=torch.device('cpu'),amp=False,grad_clip_norm=1,
            grad_accum_steps=1,num_classes=2,ignore_index=255,epoch=0,epochs=1,
            enable_cls_aux=False,cls_aux_target_type='presence',cls_aux_loss_type='bce',
            cls_aux_weight=0,cls_aux_num_classes=0,cls_aux_pos_weight=None,
            gt_present_only=True,group_confusion_matrices=groups)
        with patch('goose_semseg.engine.trainer.mask2former_semantic_scores',side_effect=lambda x,**kw:x):
            result=run_epoch(loader=DataLoader(FakeTiles(),batch_size=2),**kwargs)
        self.assertEqual(result[1],1.0)
        self.assertTrue(torch.equal(groups['sensor/Altum']+groups['sensor/P1'],result[-1]))
        self.assertEqual(int(groups['task/P1/b'].sum()),8)
        with self.assertRaises(ValueError):
            run_epoch(loader=DataLoader(FakeTiles(),batch_size=2,shuffle=True),**kwargs)


if __name__ == '__main__': unittest.main()
