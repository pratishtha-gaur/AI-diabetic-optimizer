# models/rl/glucose_env.py
#
# WHAT THIS FILE DOES
# -------------------
# Defines a custom Gymnasium (formerly OpenAI Gym) environment —
# a simulated diabetic "body" that the RL agent interacts with.
#
# THE RL LOOP (this is the core idea of Phase 3)
# -------------------------------------------------
#   1. Environment gives agent the current STATE
#   2. Agent picks an ACTION (do nothing / insulin / carbs)
#   3. Environment simulates what happens to glucose (step forward 5 min)
#   4. Environment returns: new STATE, REWARD, and whether episode is done
#   5. Repeat for 288 steps (= 1 simulated day)
#
# WHY A SEPARATE SIMULATOR FROM Phase 1's synthetic.py?
# ------------------------------------------------------
# Phase 1's simulator generates a FIXED sequence — it doesn't
# react to anything. This environment must REACT to the agent's
# actions: if the agent takes insulin, glucose must actually drop
# in response. That's the fundamental difference between a
# "dataset" and an "environment".

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ──────────────────────────────────────────────
# Action definitions — what the agent can choose
# ──────────────────────────────────────────────
ACTION_NAMES = {
    0: "Do nothing",
    1: "Small correction insulin",
    2: "Large correction insulin",
    3: "Eat fast-acting carbs",
}

# How much each action immediately affects glucose (mg/dL per step)
# and how much "insulin on board" or "carbs on board" it adds.
# These are simplified physiological effects — real insulin/carb
# action curves are more complex (peak after ~60-90 min), but this
# captures the essential dynamic: insulin lowers glucose gradually,
# carbs raise it quickly.
INSULIN_EFFECT = {0: 0.0, 1: 0.3, 2: 0.6, 3: 0.0}   # "insulin on board" added
CARB_EFFECT    = {0: 0.0, 1: 0.0, 2: 0.0, 3: 15.0}  # immediate mg/dL bump


class GlucoseEnv(gym.Env):
    """
    A simplified simulated environment for Type 1 diabetes management.

    STATE (observation) — 5 numbers, all roughly 0-1 scale:
        [glucose_norm, glucose_trend, sin_hour, cos_hour, insulin_on_board]

    ACTION — discrete choice of 4:
        0 = Do nothing
        1 = Small correction insulin (lowers glucose gradually)
        2 = Large correction insulin (lowers glucose faster)
        3 = Eat fast-acting carbs (raises glucose quickly)

    REWARD — shaped to encourage staying in 70-180 mg/dL range,
    with hypoglycemia (low glucose) penalized MORE than
    hyperglycemia (high glucose) — this mirrors real clinical
    guidance: "when in doubt, run a little high, never low."

    EPISODE — one simulated day = 288 steps (5-min intervals × 24hr)
    """

    # Glucose bounds (mg/dL) — same as Phase 1's simulator
    GLUCOSE_MIN = 40.0
    GLUCOSE_MAX = 400.0
    NORM_MIN    = 70.0   # for normalization (matches Phase 2's scaler range)
    NORM_MAX    = 250.0

    STEPS_PER_EPISODE = 288  # 24 hours / 5-min steps

    def __init__(self):
        super().__init__()

        # ── Action space: 4 discrete choices ─────────────────────
        self.action_space = spaces.Discrete(4)

        # ── Observation space: 5 continuous values ───────────────
        # Box defines the valid range for each dimension.
        # We allow values slightly outside [0,1] for glucose_norm
        # since real glucose can exceed our normal 70-250 reference range.
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([2.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Internal state variables (set in reset())
        self.glucose = None
        self.insulin_on_board = None
        self.step_count = None
        self.hour = None

    # ──────────────────────────────────────────────
    # reset() — start a new episode (new simulated day)
    # ──────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Start glucose somewhere reasonable but randomized
        # so the agent sees a variety of starting situations.
        self.glucose = self.np_random.uniform(90, 160)

        # No insulin active at the start of the day
        self.insulin_on_board = 0.0

        # Random starting hour — agent must learn to handle any time of day
        self.hour = self.np_random.uniform(0, 24)

        self.step_count = 0

        observation = self._get_obs()
        info = {"glucose_mgdl": self.glucose}
        return observation, info

    # ──────────────────────────────────────────────
    # step() — apply one action, advance time by 5 minutes
    # ──────────────────────────────────────────────
    def step(self, action):
        action = int(action)

        # ── 1. Apply the agent's action ───────────────────────────
        # Insulin: adds to "insulin on board" which decays glucose
        # over time (simulating gradual insulin action)
        self.insulin_on_board += INSULIN_EFFECT[action]

        # Carbs: immediate glucose bump (fast-acting carbs hit quickly)
        carb_bump = CARB_EFFECT[action]

        # ── 2. Simulate natural glucose dynamics for this step ────
        # (a) Meal effects — same meal times as Phase 1's simulator
        meal_effect = 0.0
        meals = {8: 1.0, 13: 1.0, 19: 1.0}  # breakfast, lunch, dinner
        for meal_hour in meals:
            if abs(self.hour - meal_hour) < 0.25:  # 15-min window
                meal_effect += self.np_random.uniform(15, 30)

        # (b) Circadian rhythm — same as Phase 1
        circadian = 2 * np.sin((self.hour - 7) * np.pi / 12)

        # (c) Insulin effect — "insulin on board" pulls glucose down
        insulin_effect = -self.insulin_on_board * 8.0

        # (d) Random physiological noise
        noise = self.np_random.normal(0, 2)

        # ── 3. Update glucose ──────────────────────────────────────
        self.glucose += meal_effect + circadian + insulin_effect + carb_bump + noise
        self.glucose = np.clip(self.glucose, self.GLUCOSE_MIN, self.GLUCOSE_MAX)

        # ── 4. Insulin decays over time (gets "used up") ──────────
        self.insulin_on_board *= 0.9  # 10% decay per 5-min step
        self.insulin_on_board = np.clip(self.insulin_on_board, 0.0, 1.0)

        # ── 5. Advance time ────────────────────────────────────────
        self.hour = (self.hour + 5 / 60.0) % 24
        self.step_count += 1

        # ── 6. Compute reward ──────────────────────────────────────
        reward = self._compute_reward(self.glucose, action)

        # ── 7. Check if episode is done ───────────────────────────
        terminated = False  # we don't end early for "death" in this simple version
        truncated = self.step_count >= self.STEPS_PER_EPISODE

        observation = self._get_obs()
        info = {
            "glucose_mgdl": self.glucose,
            "action_name": ACTION_NAMES[action],
        }

        return observation, reward, terminated, truncated, info

    # ──────────────────────────────────────────────
    # _get_obs() — build the 5-number state vector
    # ──────────────────────────────────────────────
    def _get_obs(self):
        # Normalize glucose the same way Phase 2 did: (val - 70) / (250 - 70)
        glucose_norm = (self.glucose - self.NORM_MIN) / (self.NORM_MAX - self.NORM_MIN)

        # Trend: store previous glucose to compute change
        # (simple approach: recompute each call from a stored previous value)
        if not hasattr(self, "_prev_glucose"):
            self._prev_glucose = self.glucose
        glucose_trend = np.clip((self.glucose - self._prev_glucose) / 20.0, -1.0, 1.0)
        self._prev_glucose = self.glucose

        # Cyclical time encoding — identical to Phase 2's sin_hour/cos_hour
        sin_hour = np.sin(2 * np.pi * self.hour / 24)
        cos_hour = np.cos(2 * np.pi * self.hour / 24)

        return np.array([
            glucose_norm,
            glucose_trend,
            sin_hour,
            cos_hour,
            self.insulin_on_board,
        ], dtype=np.float32)

    # ──────────────────────────────────────────────
    # _compute_reward() — the reward shaping logic
    # ──────────────────────────────────────────────
    def _compute_reward(self, glucose, action):
        """
        Reward design (see Phase 3 plan for full reasoning):
          - Severe hypo  (<54):       -10  (clinically dangerous)
          - Mild hypo    (54-70):      -3
          - In range     (70-180):     +1  (the goal)
          - Mild hyper   (180-250):    -1
          - Severe hyper (>250):       -3
          - Small penalty for taking unnecessary action (-0.1)
            → discourages the agent from constantly intervening
        """
        if glucose < 54:
            reward = -10.0
        elif glucose < 70:
            reward = -3.0
        elif glucose <= 180:
            reward = 1.0
        elif glucose <= 250:
            reward = -1.0
        else:
            reward = -3.0

        # Small cost for taking any action other than "do nothing"
        # This prevents the agent from spamming insulin/carbs when
        # not needed — actions should be deliberate.
        if action != 0:
            reward -= 0.1

        return reward

    # ──────────────────────────────────────────────
    # render() — optional human-readable printout
    # ──────────────────────────────────────────────
    def render(self):
        print(
            f"Hour: {self.hour:5.2f} | "
            f"Glucose: {self.glucose:6.1f} mg/dL | "
            f"Insulin on board: {self.insulin_on_board:.2f}"
        )


# ──────────────────────────────────────────────
# Quick test — run this file directly to verify
# ──────────────────────────────────────────────
if __name__ == "__main__":
    env = GlucoseEnv()
    obs, info = env.reset(seed=42)

    print("🎮 GlucoseEnv created successfully")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space:      {env.action_space}")
    print(f"\n📊 Initial state: {obs.round(3)}")
    print(f"   (glucose_norm, trend, sin_hour, cos_hour, insulin_on_board)")
    print(f"   Starting glucose: {info['glucose_mgdl']:.1f} mg/dL\n")

    # Run a few random steps to verify everything works
    print("Running 5 random steps:")
    for i in range(5):
        action = env.action_space.sample()  # random action
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"  Step {i+1}: action={ACTION_NAMES[action]:28s} "
            f"glucose={info['glucose_mgdl']:6.1f} reward={reward:5.2f}"
        )

    print("\n✅ Environment working correctly")