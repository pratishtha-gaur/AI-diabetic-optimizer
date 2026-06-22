# models/rl/evaluate_agent.py
#
# WHAT THIS FILE DOES
# -------------------
# Loads the trained PPO agent and runs it on fresh simulated days
# (the environment generates new random scenarios each reset, so
# this is automatically "unseen" data — similar in spirit to
# Phase 2's test set).
#
# METRICS — the RL equivalent of MAE/RMSE/R²:
#
#   % Time in Range (70-180 mg/dL)
#       The single most important clinical metric for diabetes
#       management. Real CGM reports lead with this number.
#
#   Hypo events (glucose < 70)
#       How often the agent let glucose drop dangerously low.
#       This should be RARE — hypos are the most acute danger.
#
#   Hyper events (glucose > 180)
#       How often glucose ran high. Less acutely dangerous than
#       hypo, but still not ideal.
#
#   Average reward per episode
#       The raw number PPO was optimizing — useful for comparing
#       training runs, less meaningful to a non-technical audience.

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.rl.glucose_env import GlucoseEnv, ACTION_NAMES


def evaluate(n_episodes=20):
    # ── Paths ──────────────────────────────────────────────────
    root       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(root, "models", "saved", "glucose_rl_agent.zip")

    if not os.path.exists(model_path):
        print("❌ No saved agent found. Run train_agent.py first.")
        return

    # ── Load trained agent ──────────────────────────────────────
    model = PPO.load(model_path)
    print(f"✅ Loaded trained agent from {model_path}")

    env = GlucoseEnv()

    # ── Run evaluation episodes ──────────────────────────────────
    all_glucose = []       # for plotting one example episode
    all_actions = []
    episode_rewards = []
    time_in_range = []     # % per episode
    hypo_counts = []
    hyper_counts = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=1000 + ep)  # fixed seeds = reproducible eval
        done = False
        ep_reward = 0
        glucose_trace, action_trace = [], []

        while not done:
            # deterministic=True: always pick the BEST action,
            # not a random sample from the policy's probability
            # distribution. We want the agent's true best behavior,
            # not exploration, during evaluation.
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_reward += reward
            glucose_trace.append(info["glucose_mgdl"])
            action_trace.append(int(action))

        episode_rewards.append(ep_reward)

        glucose_arr = np.array(glucose_trace)
        in_range = np.mean((glucose_arr >= 70) & (glucose_arr <= 180)) * 100
        hypo = np.sum(glucose_arr < 70)
        hyper = np.sum(glucose_arr > 180)

        time_in_range.append(in_range)
        hypo_counts.append(hypo)
        hyper_counts.append(hyper)

        if ep == 0:  # save first episode for plotting
            all_glucose = glucose_trace
            all_actions = action_trace

    # ── Print summary ─────────────────────────────────────────────
    print("\n" + "=" * 52)
    print(f"📊  EVALUATION RESULTS ({n_episodes} simulated days)")
    print("=" * 52)
    print(f"   Avg reward per episode:  {np.mean(episode_rewards):7.1f}")
    print(f"   Time in range (70-180): {np.mean(time_in_range):6.1f}%")
    print(f"   Avg hypo steps/day:      {np.mean(hypo_counts):6.1f}  (out of 288)")
    print(f"   Avg hyper steps/day:     {np.mean(hyper_counts):6.1f}  (out of 288)")
    print("=" * 52)

    # ── Interpret results ────────────────────────────────────────
    avg_tir = np.mean(time_in_range)
    if avg_tir > 70:
        print("🟢 Excellent! >70% time in range matches real-world clinical targets")
    elif avg_tir > 50:
        print("🟡 Reasonable — room to improve via more training or reward tuning")
    else:
        print("🔴 Low time in range — consider more training timesteps or reward redesign")

    # ── Action usage breakdown ──────────────────────────────────
    action_counts = np.bincount(all_actions, minlength=4)
    print("\n📋 Action usage (first episode):")
    for a, name in ACTION_NAMES.items():
        pct = action_counts[a] / len(all_actions) * 100
        print(f"   {name:28s}: {pct:5.1f}%")

    # ── Plot: one example day ─────────────────────────────────────
    t = np.arange(len(all_glucose)) * 5 / 60  # convert to hours

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})

    # Top: glucose trace with safe range
    ax1.plot(t, all_glucose, color="#378ADD", linewidth=1.5, label="Glucose")
    ax1.axhspan(70, 180, alpha=0.1, color="#1D9E75", label="Target range (70-180)")
    ax1.axhline(70, color="#E24B4A", linestyle="--", alpha=0.6, label="Hypo threshold")
    ax1.axhline(180, color="#E08B30", linestyle="--", alpha=0.6, label="Hyper threshold")
    ax1.set_ylabel("Glucose (mg/dL)")
    ax1.set_title(f"RL Agent — Example Day (Time in Range: {time_in_range[0]:.1f}%)")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)

    # Bottom: actions taken over time
    colors = {0: "#ccc", 1: "#378ADD", 2: "#6758DC", 3: "#E08B30"}
    for a in range(4):
        mask = np.array(all_actions) == a
        if mask.any():
            ax2.scatter(t[mask], np.array(all_actions)[mask], 
                        color=colors[a], label=ACTION_NAMES[a], s=15)
    ax2.set_xlabel("Time (hours)")
    ax2.set_ylabel("Action")
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_yticklabels(["None", "Small\ninsulin", "Large\ninsulin", "Carbs"], fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(root, "models", "saved", "rl_evaluation_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\n📈 Plot saved → {plot_path}")
    plt.show()


if __name__ == "__main__":
    evaluate()