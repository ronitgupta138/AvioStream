"""Supervised Remaining Useful Life (RUL) regression training pipeline.

Trains a LightGBM Regressor on piecewise linear clipped RUL targets (cap=125)
using multi-scale rolling features with 5-fold GroupKFold validation by unit_id.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

from src.pipeline.dataset_loader import load_cmapss_fd001, ACTIVE_SENSORS
from src.pipeline.signal_filter import SavitzkyGolayFilter, RobustTelemetryScaler
from src.pipeline.feature_extractor import MultiScaleRollingFeatureExtractor
from src.ml.metrics import evaluate_rul_predictions, rmse_score, nasa_phm_score

from src.config import MODELS_DIR


def prepare_training_features(
    train_df: pd.DataFrame
) -> Tuple[pd.DataFrame, RobustTelemetryScaler, List[str]]:
    """Applies signal filtering, scaling, and multi-scale rolling feature extraction."""
    # 1. Signal smoothing
    sg = SavitzkyGolayFilter(window_length=7, polyorder=2)
    df_smoothed = sg.filter_dataframe(train_df, feature_cols=ACTIVE_SENSORS)

    # 2. Robust scaling (fitted on baseline cycles <= 30)
    scaler = RobustTelemetryScaler(clip_range=5.0)
    baseline_mask = df_smoothed["time_cycle"] <= 30
    df_scaled = scaler.fit_transform(df_smoothed, feature_cols=ACTIVE_SENSORS, baseline_mask=baseline_mask)

    # 3. Multi-scale rolling feature extraction
    extractor = MultiScaleRollingFeatureExtractor(windows=[5, 15, 30])
    df_features, feature_names = extractor.extract_features(df_scaled, sensor_cols=ACTIVE_SENSORS)

    # Include raw active sensors and operational settings in the feature set
    full_features = ACTIVE_SENSORS + ["op_setting_1", "op_setting_2", "op_setting_3"] + feature_names
    return df_features, scaler, full_features


def train_lightgbm_model(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    n_splits: int = 5,
    random_state: int = 42
) -> Tuple[lgb.LGBMRegressor, dict]:
    """Trains LightGBM using GroupKFold cross-validation by unit_id."""
    X = train_df[feature_cols].values
    y = train_df["RUL_clipped"].values
    groups = train_df["unit_id"].values

    gkf = GroupKFold(n_splits=n_splits)
    oof_predictions = np.zeros(len(train_df))
    models = []
    fold_rmses = []

    params = {
        "n_estimators": 450,
        "learning_rate": 0.025,
        "max_depth": 5,
        "num_leaves": 25,
        "subsample": 0.85,
        "colsample_bytree": 0.75,
        "min_child_samples": 25,
        "reg_alpha": 0.1,
        "reg_lambda": 0.5,
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1
    }

    print(f"Training LightGBM across {n_splits} unit-grouped folds on {len(feature_cols)} features...")
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )

        val_preds = model.predict(X_val)
        oof_predictions[val_idx] = val_preds
        fold_rmse = rmse_score(y_val, val_preds)
        fold_rmses.append(fold_rmse)
        models.append(model)
        print(f"  Fold {fold + 1} RMSE: {fold_rmse:.2f} cycles")

    oof_metrics = evaluate_rul_predictions(y, oof_predictions)
    print("\n--- Out-of-Fold Cross Validation Results ---")
    print(f"OOF RMSE:       {oof_metrics['rmse']:.2f} cycles")
    print(f"OOF MAE:        {oof_metrics['mae']:.2f} cycles")
    print(f"OOF NASA Score: {oof_metrics['nasa_score']:.2f}")

    # Retrain final champion model on entire training dataset
    champion_model = lgb.LGBMRegressor(**params)
    champion_model.fit(X, y)

    return champion_model, oof_metrics


def save_artifacts(
    model: lgb.LGBMRegressor,
    scaler: RobustTelemetryScaler,
    feature_cols: List[str],
    metrics: dict,
    out_dir: Path = MODELS_DIR
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "rul_lightgbm_best.joblib")
    joblib.dump(scaler, out_dir / "robust_scaler.joblib")

    with open(out_dir / "feature_names.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    with open(out_dir / "train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"All training artifacts saved to: {out_dir}")


def main():
    train_df, _, _ = load_cmapss_fd001()
    df_feat, scaler, features = prepare_training_features(train_df)
    champion_model, metrics = train_lightgbm_model(df_feat, features)
    save_artifacts(champion_model, scaler, features, metrics)


if __name__ == "__main__":
    from typing import List, Tuple
    main()
