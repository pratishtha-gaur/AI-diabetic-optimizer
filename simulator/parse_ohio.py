# simulator/parse_ohio.py
#
# WHAT CHANGED
# ------------
# Per-patient time-based splitting instead of global time sort.
#
# OLD approach (broken):
#   Concat all patients → sort by timestamp → split 70/15/15 globally
#   Result: test set = last patients chronologically (unseen physiology)
#
# NEW approach (correct):
#   For each patient: split their data 70/15/15 by time
#   Then merge: all train portions → train, all val → val, all test → test
#   Result: every patient appears in every split
#
# This is the standard approach in clinical ML — never leave an entire
# patient out of training unless you're explicitly testing generalization
# (leave-one-patient-out cross-validation, a different experiment).
#
# The output CSV now has a 'split' column ('train'/'val'/'test') so
# dataset.py can use it directly instead of doing its own time split.

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import os
import sys
import glob

OHIO_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "ohio_raw"
)
GLUCOSE_FLOOR   = 39.0
GLUCOSE_CEILING = 401.0
MAX_FILL_GAP_MIN = 15


def parse_patient_xml(xml_path: str) -> pd.DataFrame:
    tree       = ET.parse(xml_path)
    root       = tree.getroot()
    patient_id = root.attrib.get("id", "unknown")

    glucose_events = root.find("glucose_level")
    if glucose_events is None:
        print(f"  ⚠️  No glucose_level in {os.path.basename(xml_path)}")
        return pd.DataFrame()

    records = []
    for event in glucose_events.findall("event"):
        ts_str = event.attrib.get("ts", "")
        value  = event.attrib.get("value", "")
        if not ts_str or not value:
            continue
        try:
            ts = pd.to_datetime(ts_str, dayfirst=True)
            g  = float(value)
            records.append({"timestamp": ts, "glucose_mg_dl": g, "patient_id": patient_id})
        except Exception:
            continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.drop_duplicates(subset=["timestamp"], keep="first")
    df = df[(df["glucose_mg_dl"] >= GLUCOSE_FLOOR) & (df["glucose_mg_dl"] <= GLUCOSE_CEILING)]

    print(f"  ✅ {os.path.basename(xml_path):35s} | "
          f"patient={patient_id:>4} | "
          f"{len(df):>5} readings | "
          f"{df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    return df


def resample_to_5min(df: pd.DataFrame) -> pd.DataFrame:
    """Resample to strict 5-min grid, forward-fill short gaps."""
    df    = df.set_index("timestamp").sort_index()
    limit = MAX_FILL_GAP_MIN // 5
    resampled = df["glucose_mg_dl"].resample("5min").mean()
    filled    = resampled.interpolate(method="linear", limit=limit)
    result    = filled.reset_index()
    result.columns = ["timestamp", "glucose_mg_dl"]
    return result.dropna(subset=["glucose_mg_dl"]).reset_index(drop=True)


def split_patient(df: pd.DataFrame, train_frac=0.70, val_frac=0.15) -> pd.DataFrame:
    """
    Split one patient's data chronologically into train/val/test.

    WHY per-patient splitting?
    A global time sort puts all of patient 6's data after patient 5's.
    A 70/15/15 global split then puts patient 6 entirely in the test set —
    the model has never seen their physiology during training.
    Per-patient splitting ensures every patient contributes to all three sets.

    Each patient's data is already in chronological order.
    We take:
      - First 70%: training (the past)
      - Next 15%:  validation (tuning)
      - Last 15%:  test (final evaluation)
    """
    n         = len(df)
    train_end = int(n * train_frac)
    val_end   = int(n * (train_frac + val_frac))

    df = df.copy()
    df["split"] = "test"
    df.iloc[:train_end, df.columns.get_loc("split")]       = "train"
    df.iloc[train_end:val_end, df.columns.get_loc("split")] = "val"

    return df


def parse_all(ohio_dir: str, output_path: str) -> pd.DataFrame:
    xml_files = sorted(glob.glob(os.path.join(ohio_dir, "*.xml")))

    if not xml_files:
        print(f"❌ No XML files found in: {ohio_dir}")
        sys.exit(1)

    print(f"📂 Found {len(xml_files)} XML files\n")

    # Group files by patient_id — each patient has 2 files (2018 + 2020)
    patient_dfs = {}

    for xml_path in xml_files:
        df_raw = parse_patient_xml(xml_path)
        if df_raw.empty:
            continue

        patient_id = df_raw["patient_id"].iloc[0]
        df_5min    = resample_to_5min(df_raw)
        df_5min["patient_id"] = patient_id

        if patient_id in patient_dfs:
            patient_dfs[patient_id] = pd.concat(
                [patient_dfs[patient_id], df_5min], ignore_index=True
            ).sort_values("timestamp").reset_index(drop=True)
        else:
            patient_dfs[patient_id] = df_5min

    print(f"\n📊 Applying per-patient train/val/test split:")
    all_dfs = []

    for patient_id, df_patient in patient_dfs.items():
        # Remove any duplicate timestamps (can arise when merging 2 files)
        df_patient = df_patient.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

        df_split = split_patient(df_patient)
        all_dfs.append(df_split)

        counts = df_split["split"].value_counts()
        print(f"   Patient {patient_id}: "
              f"train={counts.get('train',0):>4}  "
              f"val={counts.get('val',0):>4}  "
              f"test={counts.get('test',0):>4}  "
              f"total={len(df_split):>5}")

    merged = pd.concat(all_dfs, ignore_index=True)

    # Assign global day numbers (1-based, continuous)
    first_date   = merged["timestamp"].dt.date.min()
    merged["day"] = merged["timestamp"].dt.date.apply(
        lambda d: (d - first_date).days + 1
    )

    # ── Final stats ───────────────────────────────────────────────────
    in_range = ((merged["glucose_mg_dl"] >= 70) & (merged["glucose_mg_dl"] <= 180)).mean()
    split_counts = merged["split"].value_counts()

    print(f"\n{'='*60}")
    print(f"✅ Merged dataset summary")
    print(f"{'='*60}")
    print(f"   Total readings:     {len(merged):,}")
    print(f"   Patients:           {merged['patient_id'].nunique()}")
    print(f"   Glucose range:      {merged['glucose_mg_dl'].min():.0f} – {merged['glucose_mg_dl'].max():.0f} mg/dL")
    print(f"   Mean glucose:       {merged['glucose_mg_dl'].mean():.1f} mg/dL")
    print(f"   In-range (70-180):  {in_range*100:.1f}%")
    print(f"\n   Split sizes:")
    print(f"     Train: {split_counts.get('train',0):>6,} readings")
    print(f"     Val:   {split_counts.get('val',0):>6,} readings")
    print(f"     Test:  {split_counts.get('test',0):>6,} readings")
    print(f"{'='*60}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"\n💾 Saved → {output_path}")
    print(f"▶️  Next: python notebooks/preprocess_data.py")

    return merged


if __name__ == "__main__":
    ohio_dir    = sys.argv[1] if len(sys.argv) > 1 else OHIO_DATA_DIR
    root        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(root, "data", "glucose_ohio.csv")
    parse_all(ohio_dir, output_path)