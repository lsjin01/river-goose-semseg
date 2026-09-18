import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'third_party')]
from goose_semseg.data.labeling_taxonomy import (
    FINE_NAMES, COARSE_NAMES, FINE_TO_COARSE, fine_to_coarse_mapping, remap_mask, aggregate_confusion,
)
from goose_semseg.data.coarse_labels import build_batch_cls_aux_targets
from goose_semseg.utils.samples import _resolve_cls_aux_output_dim
from goose_semseg.models.builder import CLSMultiLabelHead


class TaxonomyTests(unittest.TestCase):
    def test_backbone_grad_gate_preserves_training_and_input_adapters(self):
        from goose_semseg.models.backbone.adapter import backbone_grad_enabled
        x=torch.ones(2)
        self.assertTrue(backbone_grad_enabled(False,x))
        self.assertFalse(backbone_grad_enabled(True,x))
        x.requires_grad_(True)
        self.assertTrue(backbone_grad_enabled(True,x))
        with torch.no_grad():
            self.assertFalse(backbone_grad_enabled(False,x))
            self.assertFalse(backbone_grad_enabled(True,x))

    def test_complete_partition_and_ignore(self):
        x=torch.tensor([[0,1,2,3,4,5,6,7,8,9,10,255]])
        original=x.clone()
        expected=torch.tensor([[0,1,2,3,4,4,4,4,4,5,6,255]])
        self.assertTrue(torch.equal(remap_mask(x),expected))
        self.assertTrue(torch.equal(x,original))
        self.assertEqual(set(FINE_TO_COARSE),set(range(11)))
        self.assertEqual(set(FINE_TO_COARSE.values()),set(range(7)))

    def test_reject_wrong_taxonomy(self):
        with self.assertRaises(ValueError): fine_to_coarse_mapping(tuple(reversed(FINE_NAMES)))
        with self.assertRaises(ValueError): remap_mask(torch.tensor([11]))
        with self.assertRaises(ValueError): remap_mask(torch.tensor([1.5]))
        with self.assertRaises(ValueError): aggregate_confusion(torch.zeros(12,12))

    def test_new_aux_and_legacy_unchanged(self):
        labels=torch.tensor([[[0,5,10,255]], [[3,9,255,255]]])
        targets=build_batch_cls_aux_targets(labels,target_type='coarse',num_classes=11,
            num_coarse=7,fine_to_coarse=fine_to_coarse_mapping())
        self.assertTrue(torch.equal(targets,torch.tensor([[1.,0,0,0,1,0,1],[0.,0,0,1,0,1,0]])))
        self.assertEqual(_resolve_cls_aux_output_dim(target_type='coarse',num_classes=11,
                         taxonomy='labeling_data'),7)
        self.assertEqual(_resolve_cls_aux_output_dim(target_type='coarse',num_classes=12),3)
        legacy=build_batch_cls_aux_targets(torch.tensor([[[0,1,6,10]]]),target_type='coarse')
        self.assertTrue(torch.equal(legacy,torch.ones(1,3)))
        # Current land ID 0 is a real class; legacy background ID 0 was excluded.
        self.assertEqual(float(build_batch_cls_aux_targets(torch.zeros(1,1,1,dtype=torch.long)).sum()),0)

    def test_coarse_head_backward(self):
        head=CLSMultiLabelHead(in_dim=8,num_labels=7)
        x=torch.randn(2,8,requires_grad=True)
        y=build_batch_cls_aux_targets(torch.tensor([[[4,5]],[[0,10]]]),num_classes=11,
                                     num_coarse=7,fine_to_coarse=fine_to_coarse_mapping())
        logits=head(x)
        self.assertEqual(logits.shape,y.shape)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y)
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(x.grad.abs().sum()),0)

    def test_confusion_conservation(self):
        cm=torch.arange(121,dtype=torch.int64).reshape(11,11)
        coarse=aggregate_confusion(cm)
        self.assertEqual(int(coarse.sum()),int(cm.sum()))
        self.assertEqual(int(coarse[4,4]),int(cm[4:9,4:9].sum()))
        cm.zero_();cm[5,6]=100;cm[10,6]=20
        coarse=aggregate_confusion(cm)
        self.assertEqual(int(coarse[4,4]),100)
        self.assertEqual(int(coarse[6,4]),20)


if __name__ == '__main__': unittest.main()
