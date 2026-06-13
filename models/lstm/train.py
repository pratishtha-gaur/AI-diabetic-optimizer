# models/lstm/train.py
#
# WHAT THIS FILE DOES
# -------------------
# Trains the GlucoseLSTM model on your glucose data and
# tracks every experiment with MLflow.
#
# THE TRAINING LOOP — the heart of all ML
# ----------------------------------------
# Training is just this repeated thousands of times:
#
#   1. Feed a batch of windows to the model (forward pass)
#   2. Compare model's prediction to the real answer (loss)
#   3. Figure out how each weight contributed to the error (backward pass)
#   4. Nudge all weights slightly in the direction that reduces error (optimizer step)
#   5. Repeat until loss stops improving
#
# WHAT IS MLFLOW?
# ---------------
# MLflow is an experiment tracker. Every time you run train.py
# it logs: hyperparameters, loss curves, and the trained model.
# Later you can compare runs and pick the best one.
# View the UI with: mlflow ui   (in your terminal)

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch

# Add project root to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.lstm.dataset import (
    load_and_split, fit_scalers, apply_scalers, GlucoseDataset
)
from models.lstm.model import GlucoseLSTM


# ──────────────────────────────────────────────
# HYPERPARAMETERS
# These are the knobs you can tune.
# Changing them and comparing results is called
# "hyperparameter tuning" — MLflow tracks all of it.
# ──────────────────────────────────────────────
CONFIG = {
    # Data
    "seq_len":     12,     # input window: 12 steps = 1 hour
    "horizon":     1,      # predict 1 step ahead = 5 minutes

    # Model architecture
    "input_size":  6,      # number of features (must match FEATURE_COLS)
    "hidden_size": 64,     # LSTM memory units — try 32 or 128
    "num_layers":  2,      # stacked LSTM layers — try 1 or 3
    "dropout":     0.2,    # regularization — try 0.1 to 0.4

    # Training
    "epochs":      50,     # how many full passes through training data
    "batch_size":  32,     # how many windows to process at once
    "lr":          1e-3,   # learning rate — how big each weight update is . 0.001 is standard Adam starting point
    "patience":    10,     # early stopping: stop if val loss doesn't improve
}


def train():
    # ── Paths ──────────────────────────────────────────────────
    root      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(root, "data", "glucose_preprocessed.csv")
    save_dir  = os.path.join(root, "models", "saved")
    os.makedirs(save_dir, exist_ok=True)

    # ── Device: use GPU if available, otherwise CPU ─────────────
    # On most laptops this will be CPU — that's fine for this dataset.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Training on: {device}")

    # ── Data preparation ────────────────────────────────────────
    train_df, val_df, test_df = load_and_split(data_path)
    scaler = fit_scalers(train_df)

    train_data = apply_scalers(train_df, scaler)
    val_data   = apply_scalers(val_df,   scaler)

    train_ds = GlucoseDataset(train_data, CONFIG["seq_len"], CONFIG["horizon"])
    val_ds   = GlucoseDataset(val_data,   CONFIG["seq_len"], CONFIG["horizon"])

    # DataLoader: handles batching, shuffling, and parallel loading
    # shuffle=True on training so model doesn't memorize the order
    # shuffle=False on validation so results are reproducible
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"], shuffle=False)

    # ── Model, loss, optimizer ──────────────────────────────────
    model = GlucoseLSTM(
        input_size=CONFIG["input_size"],
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"],
    ).to(device)

    # MSELoss = Mean Squared Error
    # Measures average squared difference between prediction and truth.
    # We use MSE during training, then MAE for human-readable evaluation.
    criterion = nn.MSELoss()

    # Adam optimizer: the most popular gradient descent algorithm.
    # It adapts the learning rate per parameter — smarter than plain SGD.
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])

    # Learning rate scheduler: if val loss plateaus for 5 epochs,
    # reduce lr by 50%. Helps escape flat spots in the loss landscape.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, verbose=True
    )

    print(f"\n🧠 Model parameters: {model.count_parameters():,}")
    print(f"📦 Training windows:  {len(train_ds)}")
    print(f"📦 Validation windows:{len(val_ds)}")

    # ── MLflow experiment tracking ──────────────────────────────
    mlflow.set_experiment("glucose_lstm")

    with mlflow.start_run():
        # Log all hyperparameters so you can compare runs later
        mlflow.log_params(CONFIG)

        best_val_loss = float("inf")
        patience_counter = 0
        train_losses, val_losses = [], []

        # ── Training loop ───────────────────────────────────────
        for epoch in range(CONFIG["epochs"]):

            # ── Training phase ──────────────────────────────────
            model.train()   # enables dropout, batch norm, etc.
            epoch_train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()        # clear gradients from last step
                predictions = model(X_batch) # forward pass
                loss = criterion(predictions, y_batch)  # compute error
                loss.backward()              # backward pass: compute gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # ↑ Gradient clipping: prevents "exploding gradients" in LSTMs
                #   where gradients can grow exponentially through time steps
                optimizer.step()             # update weights

                epoch_train_loss += loss.item()

            avg_train_loss = epoch_train_loss / len(train_loader)

            # ── Validation phase ────────────────────────────────
            model.eval()    # disables dropout for clean predictions
            epoch_val_loss = 0.0

            with torch.no_grad():  # no gradient computation needed for validation
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    predictions = model(X_batch)
                    loss = criterion(predictions, y_batch)
                    epoch_val_loss += loss.item()

            avg_val_loss = epoch_val_loss / len(val_loader)
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            # Step the LR scheduler
            scheduler.step(avg_val_loss)

            # Log metrics to MLflow
            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "val_loss":   avg_val_loss,
            }, step=epoch)

            print(
                f"Epoch [{epoch+1:>3}/{CONFIG['epochs']}] "
                f"Train Loss: {avg_train_loss:.6f} | "
                f"Val Loss: {avg_val_loss:.6f}"
            )

            # ── Save best model ─────────────────────────────────
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                save_path = os.path.join(save_dir, "glucose_lstm.pt")
                torch.save({
                    "epoch":      epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss":   best_val_loss,
                    "config":     CONFIG,
                }, save_path)
                print(f"   💾 Best model saved (val_loss={best_val_loss:.6f})")

            # ── Early stopping ───────────────────────────────────
            # If val loss hasn't improved for 'patience' epochs, stop.
            # This prevents overfitting — the model memorizing training data
            # instead of learning generalizable patterns.
            else:
                patience_counter += 1
                if patience_counter >= CONFIG["patience"]:
                    print(f"\n⏹️  Early stopping at epoch {epoch+1} (no improvement for {CONFIG['patience']} epochs)")
                    break

        # ── Log final model to MLflow ───────────────────────────
        mlflow.pytorch.log_model(model, "model")
        mlflow.log_metric("best_val_loss", best_val_loss)
        print(f"\n✅ Training complete. Best val loss: {best_val_loss:.6f}")
        print(f"💾 Model saved → {save_path}")
        print(f"📊 View MLflow UI: run 'mlflow ui' in your terminal")


if __name__ == "__main__":
    train()