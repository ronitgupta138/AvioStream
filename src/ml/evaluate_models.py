"""Official NASA C-MAPSS FD001 test set evaluation pipeline.

Evaluates trained models against the ground truth RUL values (RUL_FD001.txt)
for the final operational cycles of all 100 test engine units.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from src.pipeline.dataset_loader import load_cmapss_fd001, ACTIVE_SENSORS
from src.pipeline.signal_filter import SavitzkyGolayFilter, RobustTelemetryScaler
from src.pipeline.feature_extractor import MultiScaleRollingFeatureExtractor
from src.ml.metrics import evaluate_rul_predictions

from src.config import MODELS_DIR


def prepare_test_features(
    test_df: pd.DataFrame,
    scaler: RobustTelemetryScaler,
    feature_names: list
) -> pd.DataFrame:
    """Preprocesses the test set using the exact training transformations."""
    # 1. Smooth signals per test unit
    sg = SavitzkyGolayFilter(window_length=7, polyorder=2)
    test_smoothed = sg.filter_dataframe(test_df, feature_cols=ACTIVE_SENSORS)

    # 2. Apply fitted Robust Scaler
    test_scaled = scaler.transform(test_smoothed)

    # 3. Extract multi-scale rolling features
    extractor = MultiScaleRollingFeatureExtractor(windows=[5, 15, 30])
    test_feat, _ = extractor.extract_features(test_scaled, sensor_cols=ACTIVE_SENSORS)

    return test_feat


def evaluate_on_test_benchmark(
    models_dir: Path = MODELS_DIR
) -> dict:
    """Evaluates final cycle prediction per test engine against ground truth."""
    _, test_df, true_rul_df = load_cmapss_fd001()

    # Load artifacts
    model = joblib.load(models_dir / "rul_lightgbm_best.joblib")
    scaler = joblib.load(models_dir / "robust_scaler.joblib")
    with open(models_dir / "feature_names.json") as f:
        feature_names = json.load(f)

    test_features = prepare_test_features(test_df, scaler, feature_names)

    # Filter to final observed cycle per test unit
    final_cycles = test_features.groupby("unit_id")["time_cycle"].max().reset_index()
    test_last = test_features.merge(final_cycles, on=["unit_id", "time_cycle"], how="inner")

    # Sort strictly by unit_id (1 to 100)
    test_last = test_last.sort_values("unit_id").reset_index(drop=True)
    true_rul_df = true_rul_df.sort_values("unit_id").reset_index(drop=True)

    X_test = test_last[feature_names].values
    y_true = true_rul_df["RUL_true"].values

    # Predict RUL
    y_pred = model.predict(X_test)
    # Clip predictions to non-negative
    y_pred = np.clip(y_pred, 0.0, None)

    # Comprehensive metrics
    metrics = evaluate_rul_predictions(y_true, y_pred)

    print("=========================================================")
    print("      NASA C-MAPSS FD001 TEST SET BENCHMARK RESULTS      ")
    print("=========================================================")
    print(f"Test Engine Units Evaluated: {len(y_true)}")
    print(f"Test RMSE:                   {metrics['rmse']:.2f} cycles")
    print(f"Test MAE:                    {metrics['mae']:.2f} cycles")
    print(f"Official NASA PHM Score S:   {metrics['nasa_score']:.2f}")
    print(f"Mean Prediction Bias (d):    {metrics['mean_error']:.2f} cycles")
    print(f"Max Late Error (d > 0):      {metrics['max_late_error']:.2f} cycles")
    print(f"Max Early Error (d < 0):     {metrics['max_early_error']:.2f} cycles")
    print("=========================================================")

    # Target threshold verifications
    target_met = metrics['rmse'] < 15.0
    print(f"Target Threshold Check (RMSE < 15.0, NASA Score < 350): {'PASSED' if target_met and metrics['nasa_score'] < 350 else 'NEEDS_TUNING'}")

    # Save test metrics
    with open(models_dir / "test_benchmark_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    evaluate_on_test_benchmark()
