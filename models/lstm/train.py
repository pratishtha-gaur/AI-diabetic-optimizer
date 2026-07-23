# models/lstm/train.py
#
# WHAT CHANGED
# ------------
# LR scheduler patience: 5 → 10
#   Real CGM data is noisier than synthetic. The scheduler was seeing
#   noisy val_loss fluctuations and halving the LR after just 5 epochs
#   of no improvement — cutting it before the model had a chance to learn.
#   10 epochs gives the optimizer more room to work through noise.
#
# factor: 0.5 → 0.6
#   A smaller reduction step (cuts by 40% instead of 50%) is gentler
#   on noisy data — keeps enough LR to continue learning.
#
# epochs: 50 → 80
#   Real data with a gentler scheduler needs more epochs to converge.
#   Early stopping will still terminate it if it plateaus — 80 is a ceiling.
#
# patience (early stopping): 15 → 20
#   Same reasoning — give real data more room before giving up.

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.lstm.dataset import load_and_split, df_to_array, GlucoseDataset, N_FEATURES
from models.lstm.model import GlucoseLSTM

CONFIG = {
    "seq_len":     12,
    "horizon":     1,
    "input_size":  N_FEATURES,   # 7
    "hidden_size": 128,
    "num_layers":  2,
    "dropout":     0.2,
    "epochs":      80,           # was 50
    "batch_size":  64,
    "lr":          1e-3,
    "patience":    20,           # was 15
}


def train():
    root      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(root, "data", "glucose_preprocessed.csv")
    save_dir  = os.path.join(root, "models", "saved")
    os.makedirs(save_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"🖥️  Training on: {device}")

    train_df, val_df, _ = load_and_split(data_path)
    train_ds = GlucoseDataset(df_to_array(train_df), CONFIG["seq_len"], CONFIG["horizon"])
    val_ds   = GlucoseDataset(df_to_array(val_df),   CONFIG["seq_len"], CONFIG["horizon"])

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)

    model = GlucoseLSTM(
        input_size  = CONFIG["input_size"],
        hidden_size = CONFIG["hidden_size"],
        num_layers  = CONFIG["num_layers"],
        dropout     = CONFIG["dropout"],
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])

    # Gentler scheduler for noisy real data
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        patience=10,   # was 5
        factor=0.6,    # was 0.5
    )

    print(f"🧠 Parameters: {model.count_parameters():,}")
    print(f"📦 Train windows: {len(train_ds):,}  |  Val windows: {len(val_ds):,}")

    mlflow.set_experiment("glucose_lstm_ohio")

    with mlflow.start_run():
        mlflow.log_params(CONFIG)

        best_val_loss    = float("inf")
        patience_counter = 0
        save_path        = os.path.join(save_dir, "glucose_lstm.pt")

        for epoch in range(CONFIG["epochs"]):
            # ── Train ─────────────────────────────────────────────────
            model.train()
            epoch_train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_train_loss += loss.item()

            avg_train = epoch_train_loss / len(train_loader)

            # ── Validate ───────────────────────────────────────────────
            model.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    epoch_val_loss += criterion(model(X_batch), y_batch).item()

            avg_val = epoch_val_loss / len(val_loader)
            scheduler.step(avg_val)

            current_lr = optimizer.param_groups[0]["lr"]
            mlflow.log_metrics({"train_loss": avg_train, "val_loss": avg_val, "lr": current_lr}, step=epoch)

            print(f"Epoch [{epoch+1:>3}/{CONFIG['epochs']}]  "
                  f"Train: {avg_train:.6f}  Val: {avg_val:.6f}  LR: {current_lr:.2e}")

            if avg_val < best_val_loss:
                best_val_loss    = avg_val
                patience_counter = 0
                torch.save({
                    "epoch":              epoch,
                    "model_state_dict":   model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss":           best_val_loss,
                    "config":             CONFIG,
                }, save_path)
                print(f"   💾 Saved (val_loss={best_val_loss:.6f})")
            else:
                patience_counter += 1
                if patience_counter >= CONFIG["patience"]:
                    print(f"\n⏹️  Early stopping at epoch {epoch+1}")
                    break

        mlflow.pytorch.log_model(model, "model")
        mlflow.log_metric("best_val_loss", best_val_loss)
        print(f"\n✅ Done — best val_loss: {best_val_loss:.6f}")
        print(f"💾 Model → {save_path}")


if __name__ == "__main__":
    train()