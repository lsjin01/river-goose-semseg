from __future__ import annotations

from goose_semseg.data.coarse_labels import GOOSE_NUM_COARSE_CATEGORIES

# CLS auxiliary head outputs one slot per 대분류 coarse category.
# River taxonomy has no Void/background coarse class, so this is exactly the
# number of 대분류 (= 3); background (fine id 0) maps to no coarse slot.
CLS_AUX_COARSE_NUM_CLASSES = GOOSE_NUM_COARSE_CATEGORIES  # 3

__all__ = ["CLS_AUX_COARSE_NUM_CLASSES"]
