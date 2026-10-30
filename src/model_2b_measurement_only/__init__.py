"""Model 2B: tabular ECG measurements only (not the 1D CNN, not fusion).

Predicts the same five weak labels from timing/axis features. Report text is
never a feature. Reuse Model 2 labels and split when comparing modalities.
"""

from model_2_text_only.utils import LABEL_ORDER

__all__ = ["LABEL_ORDER"]
