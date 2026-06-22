# models/rl/train_agent.py
#
# WHAT THIS FILE DOES
# -------------------
# Trains an RL agent using PPO (Proximal Policy Optimization) on
# our GlucoseEnv. Tracks training progress with MLflow, just like
# Phase 2's train.py did for the LSTM.
#
# WHAT IS PPO?
# ------------
# PPO is the most widely used RL algorithm — used to train
# everything from game-playing agents to RLHF for language models.
#
# The core idea, in plain English:
#   1. The agent has a "policy" — a neural network that takes a
#      state and outputs probabilities for each action.
#   2. The agent plays many episodes, collecting (state, action, reward)
#      sequences.
#   3. Actions that led to HIGH total reward get their probability
#      INCREASED. Actions that led to LOW reward get DECREASED.
#   4. "Proximal" = the policy is only allowed to change a LITTLE
#      each update — this prevents catastrophic forgetting and
#      keeps training stable.
#
# This is fundamentally similar to Phase 2's gradient descent —
# adjust weights to do better next time — but the "correct answer"
# isn't a fixed label, it's "whatever led to more reward".

import os
import sys
import numpy as np
import mlflow
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.rl.glucose_env import GlucoseEnv


# ──────────────────────────────────────────────
# HYPERPARAMETERS
# ──────────────────────────────────────────────
CONFIG = {
    "total_timesteps": 200_000,  # total simulated 5-min steps across all episodes
                                  # 200,000 steps ÷ 288 steps/episode ≈ 694 simulated days
    "learning_rate":   3e-4,     # PPO's standard default
    "n_steps":         2048,     # how many steps to collect before each policy update
    "batch_size":      64,       # minibatch size for the update
    "gamma":           0.99,     # discount factor — how much future reward matters
    "n_envs":          4,        # parallel environments for faster data collection
}


# ──────────────────────────────────────────────
# MLflowCallback — logs training progress every episode
# ──────────────────────────────────────────────
class MLflowCallback(BaseCallback):
    """
    Stable-Baselines3 callback that logs episode rewards to MLflow.
    A "callback" is a function SB3 calls automatically during training
    — similar to how train.py logged metrics each epoch in Phase 2,
    but here it's triggered by SB3's internal loop instead of ours.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # Check each parallel environment for completed episodes
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                self.episode_rewards.append(ep_reward)

                mlflow.log_metrics({
                    "episode_reward": ep_reward,
                    "episode_length": ep_length,
                }, step=len(self.episode_rewards))

                # Print progress every 50 episodes
                if len(self.episode_rewards) % 50 == 0:
                    recent_avg = np.mean(self.episode_rewards[-50:])
                    print(
                        f"Episode {len(self.episode_rewards):>4} | "
                        f"Reward: {ep_reward:7.1f} | "
                        f"Avg (last 50): {recent_avg:7.1f}"
                    )
        return True


def train():
    # ── Paths ──────────────────────────────────────────────────
    root     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    save_dir = os.path.join(root, "models", "saved")
    os.makedirs(save_dir, exist_ok=True)

    # ── Create vectorized environment ───────────────────────────
    # "Vectorized" means running multiple copies of the environment
    # in parallel — like training 4 simulated patients simultaneously.
    # This speeds up data collection significantly.
    # Monitor wrapper tracks episode rewards/lengths for our callback.
    env = make_vec_env(
        lambda: Monitor(GlucoseEnv()),
        n_envs=CONFIG["n_envs"],
    )

    print(f"🎮 Environment: {CONFIG['n_envs']} parallel GlucoseEnv instances")
    print(f"📊 Action space: {env.action_space}")
    print(f"📊 Observation space: {env.observation_space}")

    # ── Create PPO agent ─────────────────────────────────────────
    # policy="MlpPolicy" means the policy network is a simple
    # Multi-Layer Perceptron (regular feedforward neural net) —
    # appropriate since our state is just 5 numbers, not images.
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=CONFIG["learning_rate"],
        n_steps=CONFIG["n_steps"],
        batch_size=CONFIG["batch_size"],
        gamma=CONFIG["gamma"],
        verbose=0,
    )

    print(f"\n🧠 PPO agent created")
    print(f"   Policy network: {model.policy}")

    # ── MLflow tracking ──────────────────────────────────────────
    mlflow.set_experiment("glucose_rl_agent")

    with mlflow.start_run():
        mlflow.log_params(CONFIG)

        callback = MLflowCallback()

        print(f"\n🚀 Training for {CONFIG['total_timesteps']:,} timesteps...")
        print(f"   (≈ {CONFIG['total_timesteps'] // 288} simulated days)\n")

        model.learn(
            total_timesteps=CONFIG["total_timesteps"],
            callback=callback,
        )

        # ── Save the trained agent ─────────────────────────────
        save_path = os.path.join(save_dir, "glucose_rl_agent")
        model.save(save_path)

        # Log final summary metrics
        if callback.episode_rewards:
            final_avg = np.mean(callback.episode_rewards[-50:])
            mlflow.log_metric("final_avg_reward_last50", final_avg)
            print(f"\n✅ Training complete.")
            print(f"   Final avg reward (last 50 episodes): {final_avg:.1f}")
            print(f"   (Max possible per episode ≈ 288, if always in-range)")

        print(f"💾 Agent saved → {save_path}.zip")
        print(f"📊 View MLflow UI: run 'mlflow ui' in your terminal")


if __name__ == "__main__":
    train()