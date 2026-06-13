# models/lstm/dataset.py
#
# WHAT THIS FILE DOES
# -------------------
# Converts your flat CSV of glucose readings into overlapping
# sequences that the LSTM can learn from.
#
# The core idea: instead of feeding the model one row at a time,
# we feed it a "window" of the last N readings, and ask it to
# predict the next one. This is called sequence modeling.
#
# Example with seq_len=12 (1 hour of 5-min readings):
#   Input  X: [t0, t1, t2, ..., t11]  → 12 readings (1 hour)
#   Target y: t12                      → what comes next
#   Then slide the window by 1 and repeat.

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import os


# ──────────────────────────────────────────────
# The features we'll feed to the LSTM
# These are the columns from preprocess_data.py
# ──────────────────────────────────────────────
FEATURE_COLS = [
    "glucose_norm",     # normalized glucose (our main signal)
    "glucose_ma_3",     # 15-min moving average (short-term trend)
    "glucose_ma_6",     # 30-min moving average (longer trend)
    "glucose_change",   # rate of change (direction signal)
    "sin_hour",         # time of day — cyclical encoding
    "cos_hour",         # time of day — cyclical encoding
]

TARGET_COL = "glucose_norm"   # what we're predicting


# ──────────────────────────────────────────────
# load_and_split()
# Loads the preprocessed CSV and splits it
# into train / validation / test by TIME.
#
# WHY time-based split?
# If we split randomly, a test row at 2pm might have its
# neighbouring rows (1:55pm, 2:05pm) in the training set —
# the model would be "cheating" by seeing adjacent context.
# Splitting by time ensures the model is tested on data it
# has truly never seen.
# ──────────────────────────────────────────────
def load_and_split(data_path: str, train_frac=0.7, val_frac=0.15):
    """
    Load preprocessed CSV and split into train/val/test.

    Returns three DataFrames in chronological order:
      train_df, val_df, test_df
    """
    df = pd.read_csv(data_path)

    n = len(df)
    train_end = int(n * train_frac)
    val_end   = int(n * (train_frac + val_frac))

    train_df = df.iloc[:train_end].copy()
    val_df   = df.iloc[train_end:val_end].copy()
    test_df  = df.iloc[val_end:].copy()

    print(f"✅ Split sizes → Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df


# ──────────────────────────────────────────────
# fit_scalers()
# Fit normalization ONLY on training data.
#
# WHY only training data?
# If we fit the scaler on all data, it learns the min/max
# of future (test) data — that's "data leakage". The model
# would be cheating at evaluation time.
# We fit once on train, then use the same scaler to
# transform val and test.
# ──────────────────────────────────────────────
def fit_scalers(train_df: pd.DataFrame):
    """
    Fit a MinMaxScaler on training data only.
    Returns the fitted scaler (to reuse on val/test).
    """
    scaler = MinMaxScaler()

    # We only need to rescale glucose_change since the other
    # features were already normalized in preprocess_data.py.
    # But to be safe and correct, we refit on all numeric features.
    scaler.fit(train_df[FEATURE_COLS])

    print("✅ Scaler fitted on training data only")
    return scaler


# ──────────────────────────────────────────────
# apply_scalers()
# Apply the already-fitted scaler to a DataFrame.
# ──────────────────────────────────────────────
def apply_scalers(df: pd.DataFrame, scaler: MinMaxScaler) -> np.ndarray:
    """
    Transform features using the pre-fitted scaler.
    Returns a numpy array of shape (n_rows, n_features).
    """
    return scaler.transform(df[FEATURE_COLS])


# ──────────────────────────────────────────────
# GlucoseDataset
# A PyTorch Dataset that creates sliding windows.
#
# WHY inherit from torch.utils.data.Dataset?
# PyTorch's DataLoader (which handles batching and shuffling)
# requires this interface. You just implement __len__ and
# __getitem__ and PyTorch handles the rest.
# ──────────────────────────────────────────────
class GlucoseDataset(Dataset):
    """
    Sliding window dataset for glucose sequence prediction.

    Args:
        data     : numpy array of shape (n_rows, n_features)
        seq_len  : how many past readings to use as input
                   (default=12 → 1 hour of 5-min readings)
        horizon  : how many steps ahead to predict
                   (default=1 → predict the very next reading)

    Each item returns:
        X : tensor of shape (seq_len, n_features)  ← the input window
        y : tensor of shape (1,)                   ← the target value
    """

    def __init__(self, data: np.ndarray, seq_len: int = 12, horizon: int = 1):
        self.data    = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len
        self.horizon = horizon

    def __len__(self):
        # Total windows we can create:
        # We need seq_len rows for input + horizon rows for target
        return len(self.data) - self.seq_len - self.horizon + 1

    def __getitem__(self, idx):
        # Input window: rows from idx to idx+seq_len
        X = self.data[idx : idx + self.seq_len]

        # Target: the glucose_norm value horizon steps after the window
        # FEATURE_COLS.index(TARGET_COL) gives us the column index of
        # glucose_norm so we grab only that column as our target.
        target_col_idx = FEATURE_COLS.index(TARGET_COL)
        y = self.data[idx + self.seq_len + self.horizon - 1, target_col_idx]

        return X, y.unsqueeze(0)   # y shape: (1,) for MSELoss compatibility


# ──────────────────────────────────────────────
# Quick test — run this file directly to verify
# ──────────────────────────────────────────────
if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(root, "data", "glucose_preprocessed.csv")

    train_df, val_df, test_df = load_and_split(data_path)
    scaler = fit_scalers(train_df)

    train_data = apply_scalers(train_df, scaler)
    val_data   = apply_scalers(val_df,   scaler)
    test_data  = apply_scalers(test_df,  scaler)

    train_ds = GlucoseDataset(train_data, seq_len=12)
    val_ds   = GlucoseDataset(val_data,   seq_len=12)
    test_ds  = GlucoseDataset(test_data,  seq_len=12)

    print(f"\n📦 Dataset sizes:")
    print(f"   Train: {len(train_ds)} windows")
    print(f"   Val:   {len(val_ds)} windows")
    print(f"   Test:  {len(test_ds)} windows")

    # Show one example window
    X, y = train_ds[0]
    print(f"\n🔍 Example window:")
    print(f"   X shape: {X.shape}  ← (seq_len=12, n_features={X.shape[1]})")
    print(f"   y shape: {y.shape}  ← target glucose_norm value")
    print(f"   X[0] (first timestep): {X[0].numpy().round(4)}")
    print(f"   y (target):            {y.item():.4f}")