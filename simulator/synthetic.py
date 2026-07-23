# simulator/synthetic.py
#
# WHAT CHANGED FROM V1
# --------------------
# 1. Default days: 7 → 365 for much more training data diversity
#
# 2. Meal time jitter (± 45 min): forces model to learn spike shape,
#    not just memorize fixed meal times
#
# 3. Variable glucose sensitivity per day: simulates good vs bad
#    insulin response days
#
# 4. CRITICAL FIX: explicit mean-reversion (insulin-like return force)
#    The original v1 had no mechanism to bring glucose back down after
#    a meal spike — the Gaussian spike adds energy but nothing removes it.
#    Real physiology: insulin (endogenous or injected) acts continuously
#    to pull glucose back toward a target level. We model this as a
#    proportional restoring force: the further glucose is above target,
#    the stronger the pull downward.
#
# 5. Day-level baseline drift: each day starts slightly differently
#
# MEAN REVERSION EXPLAINED
# ------------------------
# Without it: glucose goes up at breakfast → stays up all day → climbs
#             further at lunch → climbs even more at dinner
# With it:    glucose goes up at breakfast → insulin response pulls it
#             back toward ~110 over the next 1-2 hours → ready for lunch
#
# Formula: reversion = -k * (glucose - target)
#   k = 0.03 means: if glucose is 50 mg/dL above target (110),
#   each 5-min step pulls it down by 0.03 × 50 = 1.5 mg/dL
#   Over 60 min (12 steps): 12 × 1.5 = 18 mg/dL of correction
#   This roughly matches real insulin action curves.

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os


def generate_glucose_day(step_minutes=5, day_index=0, rng=None):
    """
    Generates 1 day of realistic glucose data with proper mean-reversion.

    Args:
        step_minutes : CGM sampling interval (default 5 min)
        day_index    : offsets timestamps so multi-day data is continuous
        rng          : numpy RandomState for reproducibility
    """
    if rng is None:
        rng = np.random.RandomState()

    # ── Day-level parameters ───────────────────────────────────────────
    sensitivity   = rng.uniform(0.7, 1.3)      # meal spike scaling
    baseline      = rng.uniform(95, 115)        # starting and target glucose
    reversion_k   = rng.uniform(0.025, 0.040)  # mean reversion speed

    # Meal times with jitter
    meal_hours = {
        "breakfast": 8.0  + rng.uniform(-0.75, 0.75),
        "lunch":     13.0 + rng.uniform(-0.75, 0.75),
        "dinner":    19.0 + rng.uniform(-0.75, 0.75),
    }
    # Randomly skip ~12% of meals
    meal_hours = {k: v for k, v in meal_hours.items() if rng.random() > 0.12}

    # Optional snack (~20% chance)
    snack_hour = rng.uniform(15.0, 17.0) if rng.random() < 0.20 else None

    # Activity window (~70% of days have some exercise)
    has_activity   = rng.random() < 0.70
    activity_start = rng.uniform(16.5, 18.5)
    activity_dur   = rng.uniform(0.5, 1.5)

    # ── Simulation ────────────────────────────────────────────────────
    base_date = datetime(2024, 1, 1) + timedelta(days=day_index)
    start     = base_date.replace(hour=0, minute=0, second=0, microsecond=0)

    timestamps, glucose_values = [], []
    g = baseline

    steps_per_day = int(24 * 60 / step_minutes)

    for i in range(steps_per_day):
        current_time = start + timedelta(minutes=i * step_minutes)
        hour = current_time.hour + current_time.minute / 60.0

        # ── Mean reversion (simulates insulin action) ─────────────────
        # This is the key fix. Without this, every meal spike accumulates.
        # k × (g - target) is a restoring force proportional to deviation.
        target = baseline  # the body's glucose set-point for this person/day
        reversion = -reversion_k * (g - target)

        # ── Circadian rhythm (small amplitude) ────────────────────────
        # Dawn phenomenon: slight rise 5-9am; siesta dip: slight fall 2-4pm
        circadian = 3.0 * np.sin((hour - 7) * np.pi / 12)

        # ── Meal spikes ───────────────────────────────────────────────
        # Gaussian profile: peaks ~20min after meal_hour, decays over ~90min
        # The -0.25 to 2.0hr window catches the approach and tail
        meal_spike = 0.0
        for meal_h in meal_hours.values():
            t = hour - meal_h
            if -0.25 <= t <= 2.5:
                peak_size = rng.uniform(30, 60) * sensitivity
                meal_spike += peak_size * np.exp(-(t ** 2) / 0.10)

        # ── Snack ─────────────────────────────────────────────────────
        snack_spike = 0.0
        if snack_hour is not None:
            t = hour - snack_hour
            if -0.25 <= t <= 1.5:
                snack_spike = rng.uniform(10, 22) * np.exp(-(t ** 2) / 0.06)

        # ── Activity ──────────────────────────────────────────────────
        activity_effect = 0.0
        if has_activity and activity_start <= hour <= activity_start + activity_dur:
            activity_effect = -rng.uniform(8, 16)

        # ── Physiological noise ───────────────────────────────────────
        noise = rng.normal(0, 1.2 if (hour >= 23 or hour < 6) else 2.5)

        # ── Update glucose ────────────────────────────────────────────
        # Note the order: reversion acts on current g,
        # then spikes and noise push it away from target
        delta = reversion + circadian * 0.10 + meal_spike + snack_spike + activity_effect + noise
        g = g + delta
        g = np.clip(g, 54, 350)

        timestamps.append(current_time)
        glucose_values.append(round(g, 1))

    return pd.DataFrame({"timestamp": timestamps, "glucose_mg_dl": glucose_values})


def generate_multi_day(days=365, step_minutes=5, seed=42, plot=False):
    """
    Generates glucose data for N consecutive days and saves to CSV.

    Args:
        days : number of simulated days (default 365)
        seed : for reproducibility — same seed = same dataset
    """
    rng = np.random.RandomState(seed)
    dfs = []

    for d in range(days):
        if (d + 1) % 50 == 0 or d == 0:
            print(f"  Simulating day {d+1}/{days}...")
        day_df        = generate_glucose_day(step_minutes, day_index=d, rng=rng)
        day_df["day"] = d + 1
        dfs.append(day_df)

    all_data = pd.concat(dfs, ignore_index=True)

    # ── Save ──────────────────────────────────────────────────────────
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, f"glucose_{days}days.csv")
    all_data.to_csv(csv_path, index=False)

    # ── Stats ─────────────────────────────────────────────────────────
    in_range = ((all_data["glucose_mg_dl"] >= 70) & (all_data["glucose_mg_dl"] <= 180)).mean()
    print(f"\n✅ Generated {days}-day dataset")
    print(f"   Total readings:     {len(all_data):,}")
    print(f"   Glucose range:      {all_data['glucose_mg_dl'].min():.0f} – {all_data['glucose_mg_dl'].max():.0f} mg/dL")
    print(f"   Mean glucose:       {all_data['glucose_mg_dl'].mean():.1f} mg/dL")
    print(f"   In-range (70-180):  {in_range*100:.1f}%")
    print(f"   Saved → {csv_path}")

    if plot:
        sample = all_data[all_data["day"] <= 3]
        plt.figure(figsize=(14, 5))
        plt.plot(sample["timestamp"], sample["glucose_mg_dl"], lw=1, color="#2a78d6")
        plt.axhline(180, color="#e34948", ls="--", alpha=0.6, label="High (180)")
        plt.axhline(70,  color="#eda100", ls="--", alpha=0.6, label="Low (70)")
        plt.axhspan(70, 180, alpha=0.06, color="#1baf7a")
        plt.title("Synthetic glucose — first 3 days")
        plt.xlabel("Time"); plt.ylabel("mg/dL")
        plt.legend(); plt.xticks(rotation=45); plt.tight_layout()
        plt.show()

    return all_data


if __name__ == "__main__":
    print("🩸 Generating 365-day synthetic glucose dataset...")
    df = generate_multi_day(days=365, seed=42, plot=False)