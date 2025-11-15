# notebooks/preprocess_data.py

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import seaborn as sns

# ------------------------------------------------------------
# 1️⃣ Load the dataset
# ------------------------------------------------------------
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(root_dir, "data", "glucose_7days.csv")

df = pd.read_csv(data_path)
print("✅ Loaded dataset:", df.shape)
print(df.head())

# ------------------------------------------------------------
# 2️⃣ Convert timestamp & extract useful time features
# ------------------------------------------------------------
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
df["dayofweek"] = df["timestamp"].dt.dayofweek  # 0=Mon, 6=Sun

# ------------------------------------------------------------
# 3️⃣ Add engineered features
# ------------------------------------------------------------
# Rolling mean (smooth average glucose trend)
df["glucose_ma_3"] = df["glucose_mg_dl"].rolling(window=3, min_periods=1).mean()
df["glucose_ma_6"] = df["glucose_mg_dl"].rolling(window=6, min_periods=1).mean()

# Rate of change (how fast glucose is increasing/decreasing)
df["glucose_change"] = df["glucose_mg_dl"].diff().fillna(0)

# Time-of-day cyclical encoding (for model awareness)
df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)

# ------------------------------------------------------------
# 4️⃣ Normalize glucose values (0-1 scaling)
# ------------------------------------------------------------
scaler = MinMaxScaler()
df["glucose_norm"] = scaler.fit_transform(df[["glucose_mg_dl"]])

print("\n✅ Feature engineering done. Columns now:")
print(df.columns.tolist())

# ------------------------------------------------------------
# 5️⃣ Save preprocessed dataset
# ------------------------------------------------------------
out_path = os.path.join(root_dir, "data", "glucose_preprocessed.csv")
df.to_csv(out_path, index=False)
print(f"✅ Saved preprocessed data → {out_path}")

# ------------------------------------------------------------
# 6️⃣ Optional visualization
# ------------------------------------------------------------
'''
plt.figure(figsize=(12,5))
plt.plot(df["timestamp"], df["glucose_mg_dl"], label="Raw Glucose")
plt.plot(df["timestamp"], df["glucose_ma_6"], label="6-pt Moving Avg", alpha=0.8)
plt.xlabel("Time"); plt.ylabel("Glucose (mg/dL)")
plt.title("Glucose Smoothing & Trend Features")
plt.legend(); plt.tight_layout(); plt.show()
'''

df1 = pd.read_csv("data/glucose_preprocessed.csv")
sns.heatmap(df1[["glucose_mg_dl", "glucose_ma_6", "glucose_change", "glucose_norm"]].corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlations")
plt.show()
