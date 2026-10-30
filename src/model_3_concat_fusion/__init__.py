"""Model 3: concat(z_ecg, z_text) then a linear head.

Tests whether combining independently encoded residual text and the 12-lead
CNN vector beats the unimodal models under the same labels and split.
"""
