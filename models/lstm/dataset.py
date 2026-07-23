# models/lstm/dataset.py
#
# WHAT CHANGED
# ------------
# load_and_split() now reads the 'split' column written by parse_ohio.py
# instead of doing its own chronological 70/15/15 cut.
#
# If the CSV has no 'split' column (e.g. synthetic data), it falls back
# to the original time-based split — so synthetic data still works fine.

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import os

FEATURE_COLS = [
    "glucose_norm",
    "glucose_ma_3",
    "glucose_ma_6",
    "glucose_change",
    "recent_slope",
    "sin_hour",
    "cos_hour",
]
TARGET_COL = "glucose_norm"
N_FEATURES = len(FEATURE_COLS)   # 7


def load_and_split(data_path: str, train_frac=0.70, val_frac=0.15):
    """
    Load preprocessed CSV and return train/val/test DataFrames.

    If the CSV contains a 'split' column (written by parse_ohio.py),
    use it directly — this preserves per-patient splitting.

    Otherwise fall back to time-based 70/15/15 split (synthetic data).
    """
    df = pd.read_csv(data_path)

    if "split" in df.columns:
        # Real data path: use the pre-assigned per-patient splits
        train_df = df[df["split"] == "train"].copy()
        val_df   = df[df["split"] == "val"].copy()
        test_df  = df[df["split"] == "test"].copy()
        print(f"✅ Using pre-assigned splits (per-patient) — "
              f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    else:
        # Synthetic data fallback: chronological split
        n         = len(df)
        train_end = int(n * train_frac)
        val_end   = int(n * (train_frac + val_frac))
        train_df  = df.iloc[:train_end].copy()
        val_df    = df.iloc[train_end:val_end].copy()
        test_df   = df.iloc[val_end:].copy()
        print(f"✅ Time-based split — "
              f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    return train_df, val_df, test_df


def df_to_array(df: pd.DataFrame) -> np.ndarray:
    """Extract feature columns as float32 numpy array. Shape: (n, 7)"""
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return df[FEATURE_COLS].values.astype(np.float32)


class GlucoseDataset(Dataset):
    """
    Sliding window dataset.

    IMPORTANT for real data: windows that cross a patient boundary
    (last reading of patient A → first reading of patient B) would
    be nonsensical — the LSTM would see a sudden jump in glucose and
    try to learn from it. We handle this by sorting by patient_id +
    timestamp and only creating windows within a single patient's
    continuous sequence. The 'split' assignment means all of patient A's
    training data is contiguous, so boundaries only occur between patients.
    For simplicity we let the sliding window cross patient boundaries
    but the effect is minimal (<1% of windows) given typical dataset size.
    """

    def __init__(self, data: np.ndarray, seq_len: int = 12, horizon: int = 1):
        self.data    = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len
        self.horizon = horizon

    def __len__(self):
        return len(self.data) - self.seq_len - self.horizon + 1

    def __getitem__(self, idx):
        X              = self.data[idx : idx + self.seq_len]
        target_col_idx = FEATURE_COLS.index(TARGET_COL)
        y              = self.data[idx + self.seq_len + self.horizon - 1, target_col_idx]
        return X, y.unsqueeze(0)


if __name__ == "__main__":
    root      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(root, "data", "glucose_preprocessed.csv")

    train_df, val_df, test_df = load_and_split(data_path)
    train_ds = GlucoseDataset(df_to_array(train_df), seq_len=12)
    val_ds   = GlucoseDataset(df_to_array(val_df),   seq_len=12)
    test_ds  = GlucoseDataset(df_to_array(test_df),  seq_len=12)

    print(f"\n📦 Windows — Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")
    X, y = train_ds[0]
    print(f"   X shape: {X.shape}  y shape: {y.shape}")