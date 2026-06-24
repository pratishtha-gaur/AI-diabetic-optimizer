# train_all.py
#
# Regenerates both trained models from scratch.
# Called during Docker build so the server has model files
# even though they are gitignored.
#
# Run order:
#   1. Generate synthetic glucose data
#   2. Preprocess the data
#   3. Train LSTM (saves glucose_lstm.pt)
#   4. Train RL agent (saves glucose_rl_agent.zip)

import subprocess
import sys
import os

root = os.path.dirname(os.path.abspath(__file__))

steps = [
    ("Generating synthetic data",   ["python", "simulator/synthetic.py"]),
    ("Preprocessing data",          ["python", "notebooks/preprocess_data.py"]),
    ("Training LSTM model",         ["python", "models/lstm/train.py"]),
    ("Training RL agent",           ["python", "models/rl/train_agent.py"]),
]

for label, cmd in steps:
    print(f"\n{'='*52}")
    print(f"  {label}")
    print(f"{'='*52}")
    result = subprocess.run(cmd, cwd=root)
    if result.returncode != 0:
        print(f"\n❌ Failed at: {label}")
        sys.exit(1)

print("\n✅ All models trained and saved to models/saved/")