# models/lstm/model.py
#
# WHAT THIS FILE DOES
# -------------------
# Defines the LSTM neural network architecture in PyTorch.
#
# WHAT IS AN LSTM?
# ----------------
# LSTM = Long Short-Term Memory. It's a type of neural network
# designed specifically for sequences — data where order matters.
#
# Unlike a regular neural net that sees one input at a time,
# an LSTM has a "memory cell" that carries information forward
# across the whole sequence. This is why it works well for
# glucose prediction: it can remember "there was a meal spike
# 30 minutes ago" while processing the current reading.
#
# Our architecture:
#
#   Input (seq_len=12, features=6)
#       ↓
#   LSTM layers (hidden_size=64, num_layers=2)
#       ↓ (only last timestep output)
#   Dropout (regularization — prevents overfitting)
#       ↓
#   Fully connected layer (64 → 1)
#       ↓
#   Output: predicted glucose_norm (single float, 0–1)

import torch
import torch.nn as nn


class GlucoseLSTM(nn.Module):
    """
    LSTM model for glucose sequence prediction.

    Args:
        input_size   : number of features per timestep (= len(FEATURE_COLS) = 6)
        hidden_size  : how many "memory units" the LSTM has per layer
                       More = more capacity, but slower and more prone to overfitting
        num_layers   : how many LSTM layers stacked on top of each other
                       Layer 2 learns patterns from layer 1's output
        dropout      : fraction of neurons randomly disabled during training
                       Forces the network to not rely on any single neuron
        output_size  : what we're predicting (1 = next glucose_norm value)
    """

    def __init__(
        self,
        input_size:  int = 6,
        hidden_size: int = 64,
        num_layers:  int = 2,
        dropout:     float = 0.2,
        output_size: int = 1,
    ):
        super(GlucoseLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # ── LSTM core ──────────────────────────────────────────
        # This is the recurrent part. It processes the sequence
        # step by step, maintaining hidden state (short-term memory)
        # and cell state (long-term memory) across timesteps.
        #
        # batch_first=True means input shape is:
        #   (batch_size, seq_len, input_size)
        # which is more intuitive than PyTorch's default.
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # ── Dropout ────────────────────────────────────────────
        # Applied AFTER the LSTM, before the final layer.
        # Randomly zeroes 20% of activations during training only.
        # At inference (prediction) time, dropout is automatically off.
        self.dropout = nn.Dropout(dropout)

        # ── Fully connected output layer ───────────────────────
        # Takes the LAST hidden state (the LSTM's final summary
        # of the whole sequence) and maps it to our 1 prediction.
        # hidden_size → output_size  (64 → 1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass — what happens when data flows through the model.

        Args:
            x : input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            out : predicted glucose_norm, shape (batch_size, 1)

        Step by step:
          1. LSTM processes all 12 timesteps, producing output at each step
          2. We only care about the LAST timestep's output — it's the
             LSTM's final "summary" of everything it saw
          3. Dropout is applied for regularization
          4. Linear layer maps the summary to a single prediction
        """

        # Step 1: Run through LSTM
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        # h_n, c_n: final hidden and cell states (we don't use these directly)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Step 2: Take only the last timestep's output
        # lstm_out[:, -1, :] means: all batches, last time step, all hidden units
        # Shape: (batch_size, hidden_size)
        last_output = lstm_out[:, -1, :]

        # Step 3: Apply dropout
        last_output = self.dropout(last_output)

        # Step 4: Linear layer → single prediction per batch item
        # Shape: (batch_size, 1)
        out = self.fc(last_output)

        return out

    def count_parameters(self):
        """Utility: how many trainable weights does this model have?"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────
# Quick test — run this file directly to verify
# ──────────────────────────────────────────────
if __name__ == "__main__":
    model = GlucoseLSTM(
        input_size=6,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
    )

    print("🧠 GlucoseLSTM architecture:")
    print(model)
    print(f"\n📊 Total trainable parameters: {model.count_parameters():,}")

    # Simulate a batch of 32 windows, each 12 steps, 6 features
    dummy_input = torch.randn(32, 12, 6)
    output = model(dummy_input)
    print(f"\n✅ Forward pass OK")
    print(f"   Input shape:  {dummy_input.shape}  (batch=32, seq=12, features=6)")
    print(f"   Output shape: {output.shape}        (batch=32, prediction=1)")