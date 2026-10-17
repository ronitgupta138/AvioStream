"""NASA C-MAPSS Turbofan Engine Dataset Loader.

Loads sub-datasets FD001, FD002, FD003, and FD004, handles unit cycle tracking,
calculates target Remaining Useful Life (RUL), and prunes invariant sensors.
"""

from pathlib import Path
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np

SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
SETTING_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
BASE_COLS = ["unit_id", "time_cycle"] + SETTING_COLS + SENSOR_COLS

INVARIANT_SENSORS = ["s_1", "s_5", "s_6", "s_10", "s_16", "s_18", "s_19"]
ACTIVE_SENSORS = [s for s in SENSOR_COLS if s not in INVARIANT_SENSORS]

from src.config import DATA_DIR


def load_cmapss_subdataset(
    subset_id: str = "FD001",
    data_dir: Path = DATA_DIR,
    rul_clip: int = 125,
    prune_invariant: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads train, test, and ground-truth RUL DataFrames for any NASA C-MAPSS subset.

    Args:
        subset_id: Subset identifier ('FD001', 'FD002', 'FD003', or 'FD004').
        data_dir: Directory containing NASA C-MAPSS text files.
        rul_clip: Piecewise linear upper threshold for RUL (default 125 cycles).
        prune_invariant: If True, drops the 7 invariant/flatline sensors.

    Returns:
        train_df: Full run-to-failure cycles with true and clipped RUL.
        test_df: Partial operational trajectories for test engine units.
        true_rul_df: Ground truth end-of-test RUL for test engine units.
    """
    train_path = data_dir / f"train_{subset_id}.txt"
    test_path = data_dir / f"test_{subset_id}.txt"
    rul_path = data_dir / f"RUL_{subset_id}.txt"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing train data at: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test data at: {test_path}")
    if not rul_path.exists():
        raise FileNotFoundError(f"Missing ground truth RUL data at: {rul_path}")

    # 1. Load Train Telemetry
    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=BASE_COLS)

    # Calculate run-to-failure RUL for training units
    max_cycles = train_df.groupby("unit_id")["time_cycle"].max().reset_index()
    max_cycles.rename(columns={"time_cycle": "max_cycle"}, inplace=True)
    train_df = train_df.merge(max_cycles, on="unit_id", how="left")
    train_df["RUL_actual"] = train_df["max_cycle"] - train_df["time_cycle"]
    train_df["RUL_clipped"] = train_df["RUL_actual"].clip(upper=rul_clip)

    # 2. Load Test Telemetry
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=BASE_COLS)

    # 3. Load Ground Truth RUL for Test Set
    true_rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["RUL_true"])
    true_rul_df["unit_id"] = true_rul_df.index + 1

    # 4. Optional Invariant Sensor Pruning
    if prune_invariant:
        # In multi-condition datasets (FD002/FD004), verify which sensors are strictly invariant
        train_df.drop(columns=INVARIANT_SENSORS, inplace=True, errors="ignore")
        test_df.drop(columns=INVARIANT_SENSORS, inplace=True, errors="ignore")

    return train_df, test_df, true_rul_df


def load_cmapss_fd001(
    data_dir: Path = DATA_DIR,
    rul_clip: int = 125,
    prune_invariant: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Backwards-compatible convenience loader for FD001."""
    return load_cmapss_subdataset(
        subset_id="FD001",
        data_dir=data_dir,
        rul_clip=rul_clip,
        prune_invariant=prune_invariant
    )


if __name__ == "__main__":
    for sid in ["FD001", "FD002", "FD003", "FD004"]:
        tr, te, r = load_cmapss_subdataset(sid)
        print(f"[{sid}] Train: {tr.shape} ({tr['unit_id'].nunique()} units) | Test: {te.shape} ({te['unit_id'].nunique()} units) | True RUL: {len(r)}")
