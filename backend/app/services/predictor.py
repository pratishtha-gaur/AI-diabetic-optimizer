# backend/app/services/predictor.py
#
# WHAT THIS FILE DOES
# -------------------
# Handles everything related to the LSTM model:
#   1. Loading glucose_lstm.pt from disk (once, at startup)
#   2. Preprocessing incoming glucose history into the exact
#      format the LSTM expects (same pipeline as dataset.py)
#   3. Running inference and returning a mg/dL prediction
#
# WHY A SEPARATE SERVICE FILE?
# ----------------------------
# The router (predict.py) handles HTTP concerns: parse request,
# return response, set status codes. The service handles ML
# concerns: load model, preprocess data, run inference. Keeping
# them separate means you can test the ML logic without running
# a web server, and swap the HTTP layer without touching the ML.

import os
import sys
import numpy as np
import torch

# Add project root to path so we can import from models/lstm/
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root)

from models.lstm.model import GlucoseLSTM
from models.lstm.dataset import FEATURE_COLS


# ──────────────────────────────────────────────
# Constants — must match Phase 2 exactly
# ──────────────────────────────────────────────
GLUCOSE_MIN  = 70.0    # same as evaluate.py
GLUCOSE_MAX  = 250.0   # same as evaluate.py
MODEL_MAE    = 2.51    # from our Phase 2 evaluation — used as confidence range

# Map from the 6 FEATURE_COLS to their position index
# glucose_norm, glucose_ma_3, glucose_ma_6, glucose_change, sin_hour, cos_hour
FEAT_IDX = {col: i for i, col in enumerate(FEATURE_COLS)}


class GlucosePredictor:
    """
    Wraps the trained LSTM model for production inference.

    Usage:
        predictor = GlucosePredictor()
        predictor.load("models/saved/glucose_lstm.pt")
        result = predictor.predict(glucose_history=[...], current_hour=8.5)
    """

    def __init__(self):
        self.model  = None
        self.config = None
        self.device = torch.device("cpu")  # CPU for API serving is fine
        self.loaded = False

    def load(self, model_path: str) -> None:
        """
        Load the trained LSTM checkpoint from disk.
        Called ONCE at API startup — not on every request.

        WHY LOAD ONCE?
        Loading a PyTorch model from disk takes ~100ms. If we loaded
        it on every request, a user hitting the API 10 times/second
        would waste 1 second per second just loading models. Loading
        once at startup and keeping the model in memory means each
        request only pays the inference cost (~1ms on CPU).
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LSTM model not found at {model_path}")

        # Load checkpoint dictionary
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = checkpoint["config"]

        # Rebuild the architecture (same as evaluate.py does)
        self.model = GlucoseLSTM(
            input_size  = self.config["input_size"],
            hidden_size = self.config["hidden_size"],
            num_layers  = self.config["num_layers"],
            dropout     = self.config["dropout"],
        ).to(self.device)

        # Pour saved weights into the architecture
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # CRITICAL: switch to eval mode
        # Disables dropout → deterministic predictions
        # Without this, predictions would randomly vary each call
        self.model.eval()

        self.loaded = True
        print(f"✅ LSTM loaded from epoch {checkpoint['epoch']+1} "
              f"(val_loss={checkpoint['val_loss']:.6f})")

    def _build_features(self, glucose_history: list, current_hour: float) -> np.ndarray:
        """
        Convert raw glucose readings into the 6-feature format the LSTM expects.
        Replicates the exact transformations from preprocess_data.py and dataset.py.

        This is the most important preprocessing detail in Phase 4:
        the model was trained on specific engineered features, so we
        must reproduce those features EXACTLY at inference time.

        Args:
            glucose_history : list of 12 glucose values in mg/dL
            current_hour    : float, e.g. 8.5 for 8:30am

        Returns:
            numpy array of shape (12, 6) — the input window for the LSTM
        """
        glucose = np.array(glucose_history, dtype=np.float64)
        n = len(glucose)  # = 12

        # ── Feature 1: glucose_norm ────────────────────────────────
        # Same MinMax formula from preprocess_data.py
        glucose_norm = (glucose - GLUCOSE_MIN) / (GLUCOSE_MAX - GLUCOSE_MIN)

        # ── Feature 2 & 3: moving averages ────────────────────────
        # Rolling mean of last 3 and last 6 readings
        # These replicate pandas .rolling(window=N, min_periods=1).mean()
        glucose_ma_3 = np.array([
            np.mean(glucose[max(0, i-2):i+1]) for i in range(n)
        ])
        glucose_ma_6 = np.array([
            np.mean(glucose[max(0, i-5):i+1]) for i in range(n)
        ])

        # Normalize the moving averages the same way
        ma_3_norm = (glucose_ma_3 - GLUCOSE_MIN) / (GLUCOSE_MAX - GLUCOSE_MIN)
        ma_6_norm = (glucose_ma_6 - GLUCOSE_MIN) / (GLUCOSE_MAX - GLUCOSE_MIN)

        # ── Feature 4: glucose_change ───────────────────────────────
        # Rate of change: difference between consecutive readings
        glucose_change = np.diff(glucose, prepend=glucose[0])
        # Normalize change to approximately 0-1 scale
        # (max realistic change ~30 mg/dL per step)
        change_norm = np.clip(glucose_change / 30.0, -1.0, 1.0) * 0.5 + 0.5

        # ── Feature 5 & 6: cyclical time encoding ──────────────────
        # Each timestep has the SAME hour encoding for this request
        # (in real deployment, each step would have its true timestamp)
        sin_hour = np.full(n, np.sin(2 * np.pi * current_hour / 24))
        cos_hour = np.full(n, np.cos(2 * np.pi * current_hour / 24))

        # ── Stack into (12, 6) array ────────────────────────────────
        # Order must match FEATURE_COLS exactly:
        # [glucose_norm, glucose_ma_3, glucose_ma_6, glucose_change, sin_hour, cos_hour]
        features = np.stack([
            glucose_norm,
            ma_3_norm,
            ma_6_norm,
            change_norm,
            sin_hour,
            cos_hour,
        ], axis=1)  # shape: (12, 6)

        return features.astype(np.float32)

    def predict(self, glucose_history: list, current_hour: float) -> dict:
        """
        Run the LSTM on one window of glucose history.

        Args:
            glucose_history : list of 12 recent glucose readings in mg/dL
            current_hour    : current hour as decimal

        Returns:
            dict with predicted_glucose (mg/dL), confidence_range, status
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded — call load() first")

        # Build the (12, 6) feature matrix
        features = self._build_features(glucose_history, current_hour)

        # Add batch dimension: (12, 6) → (1, 12, 6)
        # The model always expects a batch, even if it's just 1 window
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        # Run inference — no gradient tracking needed
        with torch.no_grad():
            prediction_norm = self.model(x).item()  # scalar, 0-1 scale

        # De-normalize: same reverse formula from evaluate.py
        predicted_mgdl = prediction_norm * (GLUCOSE_MAX - GLUCOSE_MIN) + GLUCOSE_MIN

        # Clip to physiological range (model can occasionally predict
        # slightly outside the training range — that's expected and fine)
        predicted_mgdl = float(np.clip(predicted_mgdl, 40.0, 400.0))

        # Clinical status classification
        status = self._classify_glucose(predicted_mgdl)

        return {
            "predicted_glucose": round(predicted_mgdl, 1),
            "confidence_range":  MODEL_MAE,
            "status":            status,
        }

    def _classify_glucose(self, glucose_mgdl: float) -> str:
        """Classify a glucose value into clinical categories."""
        if glucose_mgdl < 54:
            return "critical_low"
        elif glucose_mgdl < 70:
            return "low"
        elif glucose_mgdl <= 180:
            return "in_range"
        elif glucose_mgdl <= 250:
            return "high"
        else:
            return "critical_high"


# ──────────────────────────────────────────────
# Singleton instance — shared across all requests
# ──────────────────────────────────────────────
# FastAPI creates one instance of GlucosePredictor at startup
# and reuses it for every request. This is the "load once,
# serve many" pattern standard in production ML APIs.
predictor = GlucosePredictor()