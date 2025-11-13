# simulator/synthetic.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

def generate_glucose_day(step_minutes=5):
    """
    Generates 1 day of synthetic glucose readings (every few minutes).
    Simulates random fluctuations in a realistic diabetic pattern.
    """
    start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    timestamps = []
    glucose = []

    g = 100  # base glucose level (mg/dL)

    for i in range(int(24 * 60 / step_minutes)):
        # Simulate daily variation
        hour = (i * step_minutes) / 60
        # Add baseline rhythm (morning rise, evening dip)
        base_shift = 10 * np.sin((hour - 7) * np.pi / 12)
        # Add random variation
        noise = np.random.normal(0, 3)
        g += noise + base_shift * 0.01
        g = np.clip(g, 70, 180)

        timestamps.append(start + timedelta(minutes=i * step_minutes))
        glucose.append(g)

    df = pd.DataFrame({"timestamp": timestamps, "glucose_mg_dl": glucose})
    return df


if __name__ == "__main__":
    # Always resolve absolute path safely
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    df = generate_glucose_day()

    csv_path = os.path.join(data_dir, "sample_glucose.csv")
    df.to_csv(csv_path, index=False)

    print(f"✅ Synthetic glucose data saved successfully → {csv_path}")

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(df["timestamp"], df["glucose_mg_dl"])
    plt.xlabel("Time")
    plt.ylabel("Glucose (mg/dL)")
    plt.title("Synthetic Glucose Readings (1 Day)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

