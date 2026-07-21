import torch

from goose_semseg.models.head_utils import mask2former_semantic_scores


def test_mask2former_semantic_scores_shape() -> None:
    outputs = {
        "pred_logits": torch.randn(2, 5, 4),
        "pred_masks": torch.randn(2, 5, 8, 8),
    }
    scores = mask2former_semantic_scores(outputs, target_size=(16, 12))
    assert scores.shape == (2, 3, 16, 12)
    assert torch.isfinite(scores).all()
