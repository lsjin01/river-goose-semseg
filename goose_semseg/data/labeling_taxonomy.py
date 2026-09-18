"""Conservative hierarchy for Labeling_Data, separate from the legacy AIHub taxonomy.

Algae stages share a parent, including stage 0 (not a claim of algae presence).
Mixed nps_algae remains exclusive: its annotation does not identify which pixels
belong to each constituent. Unknown `other` is not assumed to be land.
"""
from typing import Dict, Sequence

import torch

FINE_NAMES = ('land', 'bridge', 'other', 'nps', 'algae0', 'algae1',
              'algae2', 'algae3', 'algae4', 'turbid', 'nps_algae')
COARSE_NAMES = ('land', 'bridge', 'other', 'nps', 'algae', 'turbid', 'nps_algae')
COARSE_DISPLAY_NAMES = ('육지', '교량', '기타', 'NPS', '녹조 단계', '탁수', 'NPS·녹조 복합')
FINE_TO_COARSE = {i:COARSE_NAMES.index('algae' if name.startswith('algae') else name)
                  for i,name in enumerate(FINE_NAMES)}


def fine_to_coarse_mapping(fine_names: Sequence[str] = FINE_NAMES) -> Dict[int, int]:
    """Require the current 11-class ordering; do not silently remap old IDs."""
    if tuple(fine_names) != FINE_NAMES:
        raise ValueError('Labeling_Data hierarchy requires the explicit 11-class manifest order; '
                         '12-class/legacy labels must be converted first.')
    return dict(FINE_TO_COARSE)


def remap_mask(mask: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    """Create a coarse mask without modifying the fine mask or ignored pixels."""
    if mask.is_floating_point() or mask.is_complex():
        raise ValueError('Mask must contain integer class IDs')
    valid = mask != ignore_index
    if torch.any(valid & ((mask < 0) | (mask >= len(FINE_NAMES)))):
        raise ValueError('Mask contains unknown fine class IDs')
    out = torch.full_like(mask, ignore_index, dtype=torch.long)
    lookup = torch.tensor(list(FINE_TO_COARSE.values()), device=mask.device)
    out[valid] = lookup[mask[valid].long()]
    return out


def aggregate_confusion(matrix: torch.Tensor) -> torch.Tensor:
    """Re-score fixed fine argmax predictions at the coarse level (not new inference)."""
    if matrix.shape != (len(FINE_NAMES), len(FINE_NAMES)):
        raise ValueError('Expected an 11 x 11 fine confusion matrix')
    mapping = torch.tensor(list(FINE_TO_COARSE.values()), device=matrix.device)
    indices = (mapping[:,None]*len(COARSE_NAMES)+mapping[None,:]).flatten()
    out = matrix.new_zeros(len(COARSE_NAMES)**2)
    out.index_add_(0, indices, matrix.flatten())
    return out.reshape(len(COARSE_NAMES),len(COARSE_NAMES))
