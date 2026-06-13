# models/lstm/evaluate.py
#
# WHAT THIS FILE DOES
# -------------------
# Loads the best saved model, runs it on the test set
# (data it has NEVER seen), and reports:
#
#   MAE  — Mean Absolute Error
#          "On average, how many mg/dL off is the prediction?"
#          Easier to interpret: MAE of 8 means ±8 mg/dL average error.
#
#   RMSE — Root Mean Square Error
#          Penalises large errors more than MAE.
#          If RMSE >> MAE, you have occasional big misses.
#
#   R²   — Coefficient of Determination (0 to 1)
#          How much of the variance in glucose does the model explain?
#          R²=1.0 is perfect. R²=0 means no better than predicting the mean.
#
# Clinical context:
#   CGM devices (like Dexcom) have a MARD of ~9% (≈10–15 mg/dL typical).
#   If your MAE is under 15 mg/dL on synthetic data, you're on the right track.

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.lstm.dataset import (
    load_and_split, fit_scalers, apply_scalers, GlucoseDataset, FEATURE_COLS, TARGET_COL
)
from models.lstm.model import GlucoseLSTM


def evaluate():
    # ── Paths ──────────────────────────────────────────────────
    root       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path  = os.path.join(root, "data", "glucose_preprocessed.csv")
    model_path = os.path.join(root, "models", "saved", "glucose_lstm.pt")

    if not os.path.exists(model_path):
        print("❌ No saved model found. Run train.py first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Prepare data (same pipeline as training) ────────────────
    train_df, val_df, test_df = load_and_split(data_path)
    scaler    = fit_scalers(train_df)          # fit only on train
    test_data = apply_scalers(test_df, scaler) # transform test

    # ── Load checkpoint ─────────────────────────────────────────
    checkpoint = torch.load(model_path, map_location=device)
    config     = checkpoint["config"]

    model = GlucoseLSTM(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"✅ Loaded model from epoch {checkpoint['epoch']+1} (val_loss={checkpoint['val_loss']:.6f})")

    # ── Run predictions on test set ─────────────────────────────
    test_ds = GlucoseDataset(test_data, seq_len=config["seq_len"], horizon=config["horizon"])
    loader  = DataLoader(test_ds, batch_size=64, shuffle=False)

    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            preds   = model(X_batch).cpu().numpy()
            targets = y_batch.numpy()
            all_preds.extend(preds.flatten())
            all_targets.extend(targets.flatten())

    preds_norm   = np.array(all_preds)
    targets_norm = np.array(all_targets)

    # ── Convert back from 0-1 scale to mg/dL ───────────────────
    # The model predicts glucose_norm (0–1).
    # To get mg/dL back: reverse the MinMax scaling.
    # glucose_mg_dl = norm * (max - min) + min  =  norm * 180 + 70
    GLUCOSE_MIN, GLUCOSE_MAX = 70.0, 250.0

    preds_mgdl   = preds_norm   * (GLUCOSE_MAX - GLUCOSE_MIN) + GLUCOSE_MIN
    targets_mgdl = targets_norm * (GLUCOSE_MAX - GLUCOSE_MIN) + GLUCOSE_MIN

    # ── Compute metrics ─────────────────────────────────────────
    mae  = mean_absolute_error(targets_mgdl, preds_mgdl)
    rmse = np.sqrt(mean_squared_error(targets_mgdl, preds_mgdl))
    r2   = r2_score(targets_mgdl, preds_mgdl)

    print("\n" + "="*48)
    print("📊  EVALUATION RESULTS (test set)")
    print("="*48)
    print(f"   MAE  (avg error):    {mae:.2f} mg/dL")
    print(f"   RMSE (penalises big errors): {rmse:.2f} mg/dL")
    print(f"   R²   (explained variance):  {r2:.4f}")
    print("="*48)

    # ── Interpret results ───────────────────────────────────────
    if mae < 10:
        print("🟢 Excellent! MAE under 10 mg/dL — clinically competitive")
    elif mae < 20:
        print("🟡 Good. MAE under 20 mg/dL — reasonable for synthetic data")
    else:
        print("🔴 MAE over 20 mg/dL — consider more epochs or hidden_size")

    # ── Plot: predicted vs actual ────────────────────────────────
    plot_n = min(200, len(targets_mgdl))  # show first 200 predictions
    t      = np.arange(plot_n) * 5        # convert to minutes

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # Top plot: time series comparison
    ax1.plot(t, targets_mgdl[:plot_n], label="Actual glucose",    color="#378ADD", linewidth=1.5)
    ax1.plot(t, preds_mgdl[:plot_n],   label="Predicted glucose", color="#E24B4A", linewidth=1.5, alpha=0.8)
    ax1.axhline(180, color="gray", linestyle="--", alpha=0.5, label="High threshold (180)")
    ax1.axhline(70,  color="gray", linestyle=":",  alpha=0.5, label="Low threshold (70)")
    ax1.fill_between(t, targets_mgdl[:plot_n], preds_mgdl[:plot_n], alpha=0.1, color="#E24B4A")
    ax1.set_xlabel("Time (minutes)")
    ax1.set_ylabel("Glucose (mg/dL)")
    ax1.set_title(f"LSTM Glucose Prediction — Test Set (first {plot_n} predictions)")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)

    # Bottom plot: scatter plot of actual vs predicted
    ax2.scatter(targets_mgdl, preds_mgdl, alpha=0.3, s=8, color="#6758DC")
    ax2.plot(
        [GLUCOSE_MIN, GLUCOSE_MAX], [GLUCOSE_MIN, GLUCOSE_MAX],
        "r--", linewidth=1.5, label="Perfect prediction"
    )
    ax2.set_xlabel("Actual glucose (mg/dL)")
    ax2.set_ylabel("Predicted glucose (mg/dL)")
    ax2.set_title(f"Actual vs Predicted  |  MAE={mae:.1f}  RMSE={rmse:.1f}  R²={r2:.3f}")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(root, "models", "saved", "evaluation_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\n📈 Plot saved → {plot_path}")
    plt.show()


if __name__ == "__main__":
    evaluate()