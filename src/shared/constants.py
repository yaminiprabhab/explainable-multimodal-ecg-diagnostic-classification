"""Frozen five-label order and 12-lead ECG conventions.

LABEL_ORDER is the only output layout used by classifiers, metrics, and
prediction tables. Do not reorder it. LEAD_ORDER is the channel axis after
name normalization (aVR → AVR). TARGET_LENGTH_SAMPLES is 10 s at 500 Hz.
"""

LABEL_ORDER = ["NORM", "HYP", "STTC", "MI", "CD"]

# MIMIC-IV-ECG / standard 12-lead order after name normalization.
LEAD_ORDER = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]

TARGET_LENGTH_SAMPLES = 5000  # 10 s at 500 Hz
TARGET_FS = 500
N_LEADS = 12
