# models/lstm/evaluate.py
# CHANGED: GLUCOSE_MIN/MAX updated to 40/400 to match preprocess_data.py

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.lstm.dataset import load_and_split, df_to_array, GlucoseDataset, FEATURE_COLS, TARGET_COL
from models.lstm.model import GlucoseLSTM

# CHANGED: must match preprocess_data.py exactly
GLUCOSE_MIN, GLUCOSE_MAX = 40.0, 400.0


def evaluate():
    root       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path  = os.path.join(root, "data", "glucose_preprocessed.csv")
    model_path = os.path.join(root, "models", "saved", "glucose_lstm.pt")

    if not os.path.exists(model_path):
        print("❌ No saved model found. Run train.py first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")

    train_df, val_df, test_df = load_and_split(data_path)
    test_data = df_to_array(test_df)

    checkpoint = torch.load(model_path, map_location=device)
    config     = checkpoint["config"]

    model = GlucoseLSTM(
        input_size  = config["input_size"],
        hidden_size = config["hidden_size"],
        num_layers  = config["num_layers"],
        dropout     = config["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"✅ Loaded model from epoch {checkpoint['epoch']+1} "
          f"(val_loss={checkpoint['val_loss']:.6f})")
    print(f"   input_size={config['input_size']}  hidden_size={config['hidden_size']}")

    test_ds = GlucoseDataset(test_data, seq_len=config["seq_len"], horizon=config["horizon"])
    loader  = DataLoader(test_ds, batch_size=64, shuffle=False)

    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds = model(X_batch.to(device)).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_targets.extend(y_batch.numpy().flatten())

    preds_norm   = np.array(all_preds)
    targets_norm = np.array(all_targets)

    # De-normalize using the same range as preprocess_data.py
    preds_mgdl   = preds_norm   * (GLUCOSE_MAX - GLUCOSE_MIN) + GLUCOSE_MIN
    targets_mgdl = targets_norm * (GLUCOSE_MAX - GLUCOSE_MIN) + GLUCOSE_MIN

    mae  = mean_absolute_error(targets_mgdl, preds_mgdl)
    rmse = np.sqrt(mean_squared_error(targets_mgdl, preds_mgdl))
    r2   = r2_score(targets_mgdl, preds_mgdl)

    actual_dir    = np.diff(targets_mgdl)
    predicted_dir = np.diff(preds_mgdl)
    dir_accuracy  = np.mean(np.sign(actual_dir) == np.sign(predicted_dir)) * 100

    # ── Glucose range breakdown ───────────────────────────────────────
    # Tells us WHERE the model struggles (low / in-range / high)
    masks = {
        "Low    (<70)":      targets_mgdl < 70,
        "In range (70-180)": (targets_mgdl >= 70) & (targets_mgdl <= 180),
        "High  (180-400)":   targets_mgdl > 180,
    }

    print("\n" + "="*52)
    print("📊  EVALUATION RESULTS (test set)")
    print("="*52)
    print(f"   MAE  (avg error):            {mae:.2f} mg/dL")
    print(f"   RMSE (penalises big errors): {rmse:.2f} mg/dL")
    print(f"   R²   (explained variance):   {r2:.4f}")
    print(f"   Directional accuracy:        {dir_accuracy:.1f}%")
    print(f"\n   MAE by glucose zone:")
    for label, mask in masks.items():
        if mask.sum() > 0:
            zone_mae = mean_absolute_error(targets_mgdl[mask], preds_mgdl[mask])
            print(f"     {label}: {zone_mae:.1f} mg/dL  ({mask.sum():,} samples)")
    print("="*52)

    if mae < 15:
        print("🟢 Excellent MAE — clinically competitive")
    elif mae < 25:
        print("🟡 Good — reasonable for real CGM data")
    else:
        print("🔴 MAE over 25 — see zone breakdown above for where it struggles")

    if dir_accuracy > 75:
        print("🟢 Strong directional accuracy")
    elif dir_accuracy > 60:
        print("🟡 Moderate directional accuracy")
    else:
        print("🔴 Poor directional accuracy")

    # ── Plot ──────────────────────────────────────────────────────────
    plot_n = min(288, len(targets_mgdl))
    t      = np.arange(plot_n) * 5

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    ax1.plot(t, targets_mgdl[:plot_n], label="Actual",    color="#2a78d6", lw=1.5)
    ax1.plot(t, preds_mgdl[:plot_n],   label="Predicted", color="#e34948", lw=1.5, alpha=0.8)
    ax1.axhline(180, color="gray", ls="--", alpha=0.5, label="180 mg/dL")
    ax1.axhline(70,  color="gray", ls=":",  alpha=0.5, label="70 mg/dL")
    ax1.fill_between(t, targets_mgdl[:plot_n], preds_mgdl[:plot_n], alpha=0.1, color="#e34948")
    ax1.set_ylabel("Glucose (mg/dL)")
    ax1.set_title(f"LSTM — Test set  |  MAE={mae:.1f}  RMSE={rmse:.1f}  R²={r2:.3f}  Dir={dir_accuracy:.0f}%")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.scatter(targets_mgdl, preds_mgdl, alpha=0.2, s=6, color="#7c6ff7")
    ax2.plot([GLUCOSE_MIN, GLUCOSE_MAX], [GLUCOSE_MIN, GLUCOSE_MAX], "r--", lw=1.5, label="Perfect")
    ax2.set_xlabel("Actual (mg/dL)"); ax2.set_ylabel("Predicted (mg/dL)")
    ax2.set_title("Actual vs Predicted")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(root, "models", "saved", "evaluation_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\n📈 Plot saved → {plot_path}")
    plt.show()


if __name__ == "__main__":
    evaluate()