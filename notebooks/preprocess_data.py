# notebooks/preprocess_data.py

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")

# ── Normalization constants ───────────────────────────────────────────
# CHANGED: 70–250 → 40–400
# Why: OhioT1DM patients regularly hit 300–400 mg/dL.
# The old range caused values above 250 to normalize to >1.0,
# which the LSTM was never trained to handle — every high glucose
# prediction was systematically wrong.
# 40 is the CGM hardware floor, 400 is the CGM hardware ceiling.
# Using hardware limits means NO real reading will ever fall outside [0,1].
GLUCOSE_MIN  = 40.0
GLUCOSE_MAX  = 400.0
CHANGE_SCALE = 30.0   # max realistic 5-min change, unchanged


def normalize_glucose(series):
    """Linear [0,1] normalization using CGM hardware bounds."""
    return (series - GLUCOSE_MIN) / (GLUCOSE_MAX - GLUCOSE_MIN)


def normalize_change(series):
    """Signed [-1,+1] normalization. Negative=falling, positive=rising."""
    return np.clip(series / CHANGE_SCALE, -1.0, 1.0)


# ── Load dataset ──────────────────────────────────────────────────────
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(root_dir, "data")

for filename in ["glucose_ohio.csv", "glucose_365days.csv", "glucose_7days.csv"]:
    data_path = os.path.join(data_dir, filename)
    if os.path.exists(data_path):
        print(f"📂 Loading: {filename}")
        break
else:
    raise FileNotFoundError("No glucose CSV found. Run parse_ohio.py or synthetic.py first.")

df = pd.read_csv(data_path)
print(f"✅ Loaded: {len(df):,} rows")
print(f"   Glucose range in data: {df['glucose_mg_dl'].min():.0f} – {df['glucose_mg_dl'].max():.0f} mg/dL")

# ── Time features ─────────────────────────────────────────────────────
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"]      = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
df["dayofweek"] = df["timestamp"].dt.dayofweek

# ── Feature engineering ───────────────────────────────────────────────
df["glucose_norm"] = normalize_glucose(df["glucose_mg_dl"])
df["glucose_ma_3"] = normalize_glucose(
    df["glucose_mg_dl"].rolling(window=3, min_periods=1).mean()
)
df["glucose_ma_6"] = normalize_glucose(
    df["glucose_mg_dl"].rolling(window=6, min_periods=1).mean()
)

raw_change           = df["glucose_mg_dl"].diff().fillna(0)
df["glucose_change"] = normalize_change(raw_change)
df["recent_slope"]   = normalize_change(
    raw_change.rolling(window=3, min_periods=1).mean()
)

df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)

# ── Sanity check: all glucose_norm values should be in [0, 1] ─────────
out_of_range = ((df["glucose_norm"] < 0) | (df["glucose_norm"] > 1)).sum()
if out_of_range > 0:
    print(f"⚠️  {out_of_range} glucose_norm values outside [0,1] — check GLUCOSE_MIN/MAX")
else:
    print(f"✅ All glucose_norm values within [0, 1]")

# ── Feature summary ───────────────────────────────────────────────────
FEATURE_COLS = [
    "glucose_norm", "glucose_ma_3", "glucose_ma_6",
    "glucose_change", "recent_slope", "sin_hour", "cos_hour",
]
print(f"\n✅ Features ({len(FEATURE_COLS)}):")
for col in FEATURE_COLS:
    print(f"   {col:20s}: [{df[col].min():+.3f}, {df[col].max():+.3f}]  mean={df[col].mean():+.3f}")

pos = (df["glucose_change"] > 0).mean()
neg = (df["glucose_change"] < 0).mean()
print(f"\n   glucose_change: {pos*100:.1f}% rising  {neg*100:.1f}% falling")

# ── Save ──────────────────────────────────────────────────────────────
out_path = os.path.join(data_dir, "glucose_preprocessed.csv")
df.to_csv(out_path, index=False)
print(f"\n✅ Saved → {out_path}")