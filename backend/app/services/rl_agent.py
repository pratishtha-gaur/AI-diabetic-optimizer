# backend/app/services/rl_agent.py
#
# WHAT THIS FILE DOES
# -------------------
# Handles everything related to the PPO RL agent:
#   1. Loading glucose_rl_agent.zip from disk (once, at startup)
#   2. Converting the incoming request into the 5-number state
#      vector the agent expects (matching glucose_env.py exactly)
#   3. Running deterministic inference (always picks the best action)
#   4. Returning the action with a human-readable explanation

import os
import sys
import numpy as np

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root)

from stable_baselines3 import PPO
from models.rl.glucose_env import ACTION_NAMES


# ──────────────────────────────────────────────
# Constants — must match glucose_env.py exactly
# ──────────────────────────────────────────────
NORM_MIN = 70.0
NORM_MAX  = 250.0

# Clinical urgency thresholds
CRITICAL_LOW    = 54.0
LOW_THRESHOLD   = 70.0
HIGH_THRESHOLD  = 180.0
CRITICAL_HIGH   = 250.0


class RLRecommender:
    """
    Wraps the trained PPO agent for production inference.

    Usage:
        recommender = RLRecommender()
        recommender.load("models/saved/glucose_rl_agent.zip")
        result = recommender.recommend(glucose_mgdl=145, glucose_trend=3.5,
                                       current_hour=13.5, insulin_on_board=0.1)
    """

    def __init__(self):
        self.model  = None
        self.loaded = False

    def load(self, model_path: str) -> None:
        """
        Load the trained PPO agent from disk.
        Called ONCE at startup — same pattern as GlucosePredictor.load().

        PPO.load() is simpler than the LSTM case — Stable-Baselines3
        saves the full architecture + weights together in the .zip file,
        so we don't need to manually rebuild the network architecture.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"RL agent not found at {model_path}")

        # PPO.load() handles architecture + weights in one call
        self.model = PPO.load(model_path)
        self.loaded = True
        print(f"✅ RL agent loaded from {model_path}")

    def _build_state(
        self,
        glucose_mgdl:     float,
        glucose_trend:    float,
        current_hour:     float,
        insulin_on_board: float,
    ) -> np.ndarray:
        """
        Convert raw inputs into the 5-number state vector the RL agent
        was trained on — must match _get_obs() in glucose_env.py exactly.

        The agent learned from states in this specific format. If we
        pass numbers in a different scale or order, its predictions
        are meaningless — like asking someone to drive on the wrong
        side of the road.
        """
        # Feature 1: glucose_norm — exact same formula as glucose_env.py
        glucose_norm = (glucose_mgdl - NORM_MIN) / (NORM_MAX - NORM_MIN)

        # Feature 2: glucose_trend — normalize rate of change
        # glucose_env.py uses: clip((current - previous) / 20.0, -1, 1)
        # We receive the raw mg/dL trend and normalize the same way
        glucose_trend_norm = float(np.clip(glucose_trend / 20.0, -1.0, 1.0))

        # Features 3 & 4: cyclical time encoding — identical to glucose_env.py
        sin_hour = np.sin(2 * np.pi * current_hour / 24)
        cos_hour = np.cos(2 * np.pi * current_hour / 24)

        # Feature 5: insulin_on_board — already in 0-1 scale from the request
        iob = float(np.clip(insulin_on_board, 0.0, 1.0))

        return np.array([
            glucose_norm,
            glucose_trend_norm,
            sin_hour,
            cos_hour,
            iob,
        ], dtype=np.float32)

    def recommend(
        self,
        glucose_mgdl:     float,
        glucose_trend:    float,
        current_hour:     float,
        insulin_on_board: float = 0.0,
    ) -> dict:
        """
        Get a lifestyle recommendation from the trained RL agent.

        Args:
            glucose_mgdl        : current glucose in mg/dL
            glucose_trend       : rate of change in mg/dL per step
            current_hour        : current hour as decimal
            insulin_on_board    : recent insulin still active (0-1)

        Returns:
            dict with action_id, action_name, reasoning, urgency
        """
        if not self.loaded:
            raise RuntimeError("RL agent not loaded — call load() first")

        # Build the 5-number state vector
        state = self._build_state(
            glucose_mgdl, glucose_trend, current_hour, insulin_on_board
        )

        # Run deterministic inference — always pick the BEST action
        # Same as evaluate_agent.py: deterministic=True means no
        # random exploration, just the highest-probability action
        action, _ = self.model.predict(state, deterministic=True)
        action_id = int(action)
        action_name = ACTION_NAMES[action_id]

        # Build a human-readable explanation
        reasoning = self._build_reasoning(
            action_id, glucose_mgdl, glucose_trend, current_hour, insulin_on_board
        )

        # Determine urgency from current glucose level
        urgency = self._classify_urgency(glucose_mgdl, action_id)

        return {
            "action_id":   action_id,
            "action_name": action_name,
            "reasoning":   reasoning,
            "urgency":     urgency,
        }

    def _build_reasoning(
        self,
        action_id:        int,
        glucose_mgdl:     float,
        glucose_trend:    float,
        current_hour:     float,
        insulin_on_board: float,
    ) -> str:
        """
        Generate a plain-English explanation of why the agent
        chose this action, based on the current context.
        """
        trend_str = (
            "rising quickly" if glucose_trend > 5 else
            "rising"         if glucose_trend > 2 else
            "stable"         if abs(glucose_trend) <= 2 else
            "falling"        if glucose_trend > -5 else
            "falling quickly"
        )

        hour_str = (
            "morning"   if 6 <= current_hour < 12 else
            "afternoon" if 12 <= current_hour < 17 else
            "evening"   if 17 <= current_hour < 21 else
            "night"
        )

        glucose_str = f"{glucose_mgdl:.0f} mg/dL"

        if action_id == 0:  # Do nothing
            if 70 <= glucose_mgdl <= 180:
                return (f"Glucose is {glucose_str} and {trend_str} — "
                        f"well within the target range this {hour_str}. "
                        f"No intervention needed right now.")
            else:
                return (f"Glucose is {glucose_str} and {trend_str}. "
                        f"The agent judges the current trajectory will "
                        f"self-correct without intervention.")

        elif action_id == 1:  # Small correction insulin
            return (f"Glucose is {glucose_str} and {trend_str} this {hour_str}. "
                    f"A small correction dose may help bring it back toward range "
                    f"without risking a rebound low."
                    + (f" Note: you have active insulin on board ({insulin_on_board:.2f})."
                       if insulin_on_board > 0.2 else ""))

        elif action_id == 2:  # Large correction insulin
            return (f"Glucose is {glucose_str} and {trend_str} this {hour_str} — "
                    f"a larger correction is recommended to address this meaningfully. "
                    f"Monitor closely for the next 30-60 minutes."
                    + (f" Caution: significant insulin already on board ({insulin_on_board:.2f})."
                       if insulin_on_board > 0.3 else ""))

        else:  # Eat fast-acting carbs (action_id == 3)
            return (f"Glucose is {glucose_str} and {trend_str} this {hour_str}. "
                    f"Fast-acting carbohydrates are recommended to raise glucose "
                    f"back into the safe range quickly. "
                    f"Aim for 15-20g of fast carbs and recheck in 15 minutes.")

    def _classify_urgency(self, glucose_mgdl: float, action_id: int) -> str:
        """Classify urgency based on glucose level and recommended action."""
        if glucose_mgdl < CRITICAL_LOW or glucose_mgdl > CRITICAL_HIGH:
            return "urgent"
        elif glucose_mgdl < LOW_THRESHOLD or glucose_mgdl > HIGH_THRESHOLD:
            return "moderate"
        elif action_id == 0:
            return "routine"
        else:
            return "moderate"


# ──────────────────────────────────────────────
# Singleton instance — shared across all requests
# ──────────────────────────────────────────────
recommender = RLRecommender()