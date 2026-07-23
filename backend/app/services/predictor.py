# backend/app/services/predictor.py
# CHANGED: GLUCOSE_MIN/MAX updated to 40/400 to match preprocess_data.py

import os
import sys
import numpy as np
import torch

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root)

from models.lstm.model import GlucoseLSTM
from models.lstm.dataset import FEATURE_COLS

# CHANGED: must match preprocess_data.py exactly
GLUCOSE_MIN  = 40.0
GLUCOSE_MAX  = 400.0
CHANGE_SCALE = 30.0
MODEL_MAE    = 22.67   # update this after retraining


def _normalize_glucose(v):
    return (v - GLUCOSE_MIN) / (GLUCOSE_MAX - GLUCOSE_MIN)


def _normalize_change(v):
    return np.clip(v / CHANGE_SCALE, -1.0, 1.0)


class GlucosePredictor:

    def __init__(self):
        self.model  = None
        self.config = None
        self.device = torch.device("cpu")
        self.loaded = False

    def load(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LSTM model not found at {model_path}")

        checkpoint  = torch.load(model_path, map_location=self.device)
        self.config = checkpoint["config"]

        self.model = GlucoseLSTM(
            input_size  = self.config["input_size"],
            hidden_size = self.config["hidden_size"],
            num_layers  = self.config["num_layers"],
            dropout     = self.config["dropout"],
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.loaded = True
        print(f"✅ LSTM loaded from epoch {checkpoint['epoch']+1} "
              f"(val_loss={checkpoint['val_loss']:.6f})")

    def _build_features(self, glucose_history: list, current_hour: float) -> np.ndarray:
        glucose = np.array(glucose_history, dtype=np.float64)
        n = len(glucose)

        glucose_norm = _normalize_glucose(glucose)

        glucose_ma_3 = np.array([np.mean(glucose[max(0,i-2):i+1]) for i in range(n)])
        glucose_ma_6 = np.array([np.mean(glucose[max(0,i-5):i+1]) for i in range(n)])
        ma_3_norm    = _normalize_glucose(glucose_ma_3)
        ma_6_norm    = _normalize_glucose(glucose_ma_6)

        raw_change     = np.diff(glucose, prepend=glucose[0])
        glucose_change = _normalize_change(raw_change)

        rolling_mean   = np.array([np.mean(raw_change[max(0,i-2):i+1]) for i in range(n)])
        recent_slope   = _normalize_change(rolling_mean)

        sin_hour = np.full(n, np.sin(2 * np.pi * current_hour / 24))
        cos_hour = np.full(n, np.cos(2 * np.pi * current_hour / 24))

        return np.stack([
            glucose_norm, ma_3_norm, ma_6_norm,
            glucose_change, recent_slope,
            sin_hour, cos_hour,
        ], axis=1).astype(np.float32)

    def predict(self, glucose_history: list, current_hour: float) -> dict:
        if not self.loaded:
            raise RuntimeError("Model not loaded — call load() first")

        features = self._build_features(glucose_history, current_hour)
        x        = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pred_norm = self.model(x).item()

        predicted_mgdl = pred_norm * (GLUCOSE_MAX - GLUCOSE_MIN) + GLUCOSE_MIN
        predicted_mgdl = float(np.clip(predicted_mgdl, 40.0, 400.0))

        return {
            "predicted_glucose": round(predicted_mgdl, 1),
            "confidence_range":  MODEL_MAE,
            "status":            self._classify_glucose(predicted_mgdl),
        }

    def _classify_glucose(self, g: float) -> str:
        if g < 54:   return "critical_low"
        if g < 70:   return "low"
        if g <= 180: return "in_range"
        if g <= 250: return "high"
        return "critical_high"


predictor = GlucosePredictor()