"""Multi-Dataset Cross-Benchmark Evaluator for NASA C-MAPSS (FD001–FD004).

Evaluates the StrataSense prognostic engine across all 4 benchmark datasets:
  - FD001: 1 Condition, 1 Fault Mode (HPC Degradation)
  - FD002: 6 Flight Conditions, 1 Fault Mode (HPC Degradation)
  - FD003: 1 Condition, 2 Fault Modes (HPC & Fan Degradation)
  - FD004: 6 Flight Conditions, 2 Fault Modes (HPC & Fan Degradation)
"""

from typing import Dict, List, Tuple
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

from src.pipeline.dataset_loader import load_cmapss_subdataset, ACTIVE_SENSORS
from src.pipeline.regime_normalizer import OperatingRegimeNormalizer
from src.pipeline.feature_extractor import MultiScaleRollingFeatureExtractor
from src.ml.metrics import nasa_phm_score

from src.config import MODELS_DIR

OUTPUT_PATH = MODELS_DIR / "multi_dataset_benchmark_results.json"


def train_and_evaluate_subset(
    subset_id: str,
    n_regimes: int = 6,
    fast_mode: bool = False
) -> Dict[str, float]:
    """Trains and tests StrataSense pipeline on a specific C-MAPSS subset."""
    start_time = time.time()
    print(f"\n=======================================================")
    print(f"=== EVALUATING SUBSET: {subset_id} ===")
    print(f"=======================================================")

    # 1. Load Data
    train_df, test_df, true_rul_df = load_cmapss_subdataset(subset_id, rul_clip=125)
    print(f"[{subset_id}] Loaded Train: {train_df.shape} | Test: {test_df.shape} | Test Units: {len(true_rul_df)}")

    # 2. Operating Regime Normalization
    normalizer = OperatingRegimeNormalizer(n_regimes=n_regimes)
    norm_train = normalizer.fit_transform(train_df, ACTIVE_SENSORS)
    norm_test = normalizer.transform(test_df)

    # 3. Multi-Scale Feature Extraction
    extractor = MultiScaleRollingFeatureExtractor(windows=[5, 15, 30])
    feat_train, feature_cols = extractor.extract_features(norm_train, ACTIVE_SENSORS)
    feat_test, _ = extractor.extract_features(norm_test, ACTIVE_SENSORS)

    X_train = feat_train[feature_cols].values
    y_train = feat_train["RUL_clipped"].values

    # 4. Train LightGBM Regressor
    n_est = 150 if fast_mode else 300
    model = lgb.LGBMRegressor(
        n_estimators=n_est,
        learning_rate=0.04,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    model.fit(X_train, y_train)

    # 5. Predict on Last Recorded Cycle of each Test Unit
    last_cycles = feat_test.groupby("unit_id").last().reset_index()
    X_test_last = last_cycles[feature_cols].values
    y_pred = np.clip(model.predict(X_test_last), 0.0, None)
    y_true = true_rul_df.sort_values("unit_id")["RUL_true"].values

    # 6. Metrics
    rmse = float(root_mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    nasa_score = float(nasa_phm_score(y_true, y_pred))
    elapsed_sec = float(time.time() - start_time)

    metrics = {
        "subset": subset_id,
        "train_units": int(train_df["unit_id"].nunique()),
        "test_units": int(len(true_rul_df)),
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "nasa_score": round(nasa_score, 2),
        "duration_sec": round(elapsed_sec, 1)
    }

    print(f"[{subset_id}] Complete! RMSE: {rmse:.2f} | MAE: {mae:.2f} | NASA Score: {nasa_score:.2f} | Time: {elapsed_sec:.1f}s")
    return metrics


def run_all_benchmarks(fast_mode: bool = False) -> List[Dict]:
    """Runs complete 4-dataset benchmark matrix."""
    results = []
    # FD001 & FD003 have 1 condition; FD002 & FD004 have 6 conditions
    subsets_config = [
        ("FD001", 1),
        ("FD002", 6),
        ("FD003", 1),
        ("FD004", 6)
    ]

    for subset, regimes in subsets_config:
        metrics = train_and_evaluate_subset(subset, n_regimes=regimes, fast_mode=fast_mode)
        results.append(metrics)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("\n\n=======================================================")
    print("=== STRATASENSE MULTI-DATASET CONSOLIDATED MATRIX ===")
    print("=======================================================")
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))

    return results


if __name__ == "__main__":
    run_all_benchmarks(fast_mode=False)
