"""Write test_predictions.csv with study_id plus true/prob/pred per label.

compare_models.py inner-joins these tables. study_id stays in outputs/ (gitignored);
public error reports still use example_id only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import LABEL_ORDER


def write_test_predictions(
    path: Path,
    study_ids: np.ndarray | list,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame({"study_id": np.asarray(study_ids).astype(int)})
    for i, name in enumerate(LABEL_ORDER):
        frame[f"true_{name}"] = y_true[:, i]
        frame[f"prob_{name}"] = y_prob[:, i]
        frame[f"pred_{name}"] = y_pred[:, i]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame
