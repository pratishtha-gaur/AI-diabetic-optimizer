# simulator/synthetic.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# ------------------------------------------------------------
#  FUNCTION: Generate glucose readings for a single day
# ------------------------------------------------------------
def generate_glucose_day(step_minutes=5):
    """
    Generates 1 day of realistic glucose data influenced by meals, activity, and sleep.
    Returns a DataFrame with timestamps and glucose_mg_dl values.
    """
    start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    timestamps, glucose = [], []

    g = 100  # baseline glucose level (mg/dL)

    # Key events through the day
    meals = {"breakfast": 8, "lunch": 13, "dinner": 19}  # hours
    activity_hours = range(17, 19)   # light exercise after work
    sleep_hours = range(23, 24)      # stable at night

    for i in range(int(24 * 60 / step_minutes)):
        current_time = start + timedelta(minutes=i * step_minutes)
        hour = current_time.hour + current_time.minute / 60.0

        # Baseline circadian rhythm
        circadian = 5 * np.sin((hour - 7) * np.pi / 12)

        # Meal spikes (exponential peaks)
        spike = 0
        for meal, meal_hour in meals.items():
            if abs(hour - meal_hour) < 0.5:  # within 30 min window
                spike += np.random.uniform(30, 60) * np.exp(-((hour - meal_hour) ** 2) / 0.1)

        # Activity dips (reduced glucose due to exercise)
        activity_effect = -np.random.uniform(10, 20) if int(hour) in activity_hours else 0

        # Noise and sleep stability
        noise = np.random.normal(0, 1) if (hour >= 23 or hour < 6) else np.random.normal(0, 3)

        # Update glucose value
        g = g + circadian + spike + activity_effect + noise
        g = np.clip(g, 70, 250)  # safe physiological range

        timestamps.append(current_time)
        glucose.append(g)

    df = pd.DataFrame({"timestamp": timestamps, "glucose_mg_dl": glucose})
    return df


# ------------------------------------------------------------
#  FUNCTION: Generate data for multiple days
# ------------------------------------------------------------
def generate_multi_day(days=7, step_minutes=5, plot=False):
    """
    Generates glucose data for multiple consecutive days.
    Saves the final combined dataset to data/glucose_7days.csv.
    """
    dfs = []
    for d in range(days):
        print(f"🩸 Simulating day {d+1}...")
        day_df = generate_glucose_day(step_minutes)
        day_df["day"] = d + 1
        dfs.append(day_df)

    all_data = pd.concat(dfs, ignore_index=True)

    # Ensure /data directory exists
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Save dataset
    csv_path = os.path.join(data_dir, f"glucose_{days}days.csv")
    all_data.to_csv(csv_path, index=False)
    print(f"✅ Saved {days}-day dataset → {csv_path}")

    # Optional visualization
    if plot:
        plt.figure(figsize=(12, 5))
        for day in all_data["day"].unique():
            subset = all_data[all_data["day"] == day]
            plt.plot(subset["timestamp"], subset["glucose_mg_dl"], label=f"Day {day}")
        plt.axhline(180, color="r", linestyle="--", label="High (180 mg/dL)")
        plt.axhline(70, color="b", linestyle="--", label="Low (70 mg/dL)")
        plt.title(f"{days}-Day Synthetic Glucose Simulation")
        plt.xlabel("Time"); plt.ylabel("Glucose (mg/dL)")
        plt.legend(); plt.xticks(rotation=45); plt.tight_layout(); plt.show()

    return all_data


# ------------------------------------------------------------
#  MAIN EXECUTION
# ------------------------------------------------------------
if __name__ == "__main__":
    df = generate_multi_day(days=7, plot=False)




