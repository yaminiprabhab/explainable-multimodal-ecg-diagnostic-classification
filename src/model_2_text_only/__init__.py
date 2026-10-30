"""Model 2: leakage-controlled residual-text LSTM (unimodal text baseline).

Labels are weak rules from report_* fields. The classifier sees residual
text after trigger/summary phrases are removed. Not a clinical diagnostic
system. Mapping v1.0.0 must stay frozen after test evaluation.
"""

from .utils import LABEL_ORDER

__all__ = ["LABEL_ORDER"]
