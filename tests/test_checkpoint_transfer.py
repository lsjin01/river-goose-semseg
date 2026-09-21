import unittest

import torch
from torch import nn

from goose_semseg.utils.checkpoint import load_model_state_allowing_token_specialization


class _StateOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.segmentation_model = nn.ModuleList(
            [nn.Conv2d(7, 3, 1, bias=False), nn.Linear(3, 4), nn.Linear(4, 5)]
        )


class CheckpointTransferTests(unittest.TestCase):
    def test_rgb_layout_is_shifted_after_spectral_adapter(self):
        model = _StateOnlyModel()
        # The real adapter detection key is a state-dict contract, so use a tiny
        # synthetic mapping with the same key layout.
        target = model.state_dict()
        target["segmentation_model.0.rgb_projection.weight"] = target.pop(
            "segmentation_model.0.weight"
        )

        class _Synthetic(nn.Module):
            def state_dict(self, *args, **kwargs):
                return target

            def load_state_dict(self, state_dict, strict=False):
                self.loaded = state_dict
                missing = [key for key in target if key not in state_dict]
                return torch.nn.modules.module._IncompatibleKeys(missing, [])

        synthetic = _Synthetic()
        source = {
            "segmentation_model.0.weight": torch.full_like(target["segmentation_model.1.weight"], 2),
            "segmentation_model.0.bias": torch.full_like(target["segmentation_model.1.bias"], 3),
            "segmentation_model.1.weight": torch.full_like(target["segmentation_model.2.weight"], 4),
            "segmentation_model.1.bias": torch.full_like(target["segmentation_model.2.bias"], 5),
        }
        missing, unexpected, remapped = load_model_state_allowing_token_specialization(
            synthetic, source
        )
        self.assertEqual(remapped, 4)
        self.assertIn("segmentation_model.0.rgb_projection.weight", missing)
        self.assertFalse(unexpected)
        self.assertTrue(torch.equal(synthetic.loaded["segmentation_model.1.weight"], source["segmentation_model.0.weight"]))


if __name__ == "__main__":
    unittest.main()
