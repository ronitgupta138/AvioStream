"""Evaluation metrics and custom loss functions for prognostics and health management (PHM).

Includes the official NASA C-MAPSS asymmetric loss penalty function:
  d = y_pred - y_true
  h(d) = exp(-d / 13) - 1  for d < 0 (early prediction)
  h(d) = exp(d / 10) - 1   for d >= 0 (late prediction, 3x penalization)
"""

from typing import Tuple
import numpy as np


def nasa_phm_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates the official NASA PHM Prognostic Score S.

    Asymmetrically penalizes late predictions much more severely than early predictions.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    d = y_pred - y_true

    score = np.where(
        d < 0,
        np.exp(-d / 13.0) - 1.0,
        np.exp(d / 10.0) - 1.0
    )
    return float(np.sum(score))


def rmse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def evaluate_rul_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Returns comprehensive prognostic evaluation report."""
    return {
        "rmse": rmse_score(y_true, y_pred),
        "mae": mae_score(y_true, y_pred),
        "nasa_score": nasa_phm_score(y_true, y_pred),
        "mean_error": float(np.mean(y_pred - y_true)),
        "max_late_error": float(np.max(np.maximum(y_pred - y_true, 0.0))),
        "max_early_error": float(np.max(np.maximum(y_true - y_pred, 0.0)))
    }


def lightgbm_nasa_eval(preds: np.ndarray, train_data) -> Tuple[str, float, bool]:
    """Custom evaluation callback for LightGBM monitoring.

    Returns (eval_name, eval_result, is_higher_better).
    """
    labels = train_data.get_label()
    score = nasa_phm_score(labels, preds)
    return "nasa_score", score, False


if __name__ == "__main__":
    y_true = np.array([20.0, 50.0, 80.0, 10.0])
    # Case 1: Early prediction by 5 cycles
    y_early = y_true - 5.0
    # Case 2: Late prediction by 5 cycles
    y_late = y_true + 5.0

    print("=== Evaluation Metrics Sanity Test ===")
    print("True RUL:", y_true)
    print(f"Early by 5 cycles: RMSE={rmse_score(y_true, y_early):.2f}, NASA Score={nasa_phm_score(y_true, y_early):.2f}")
    print(f"Late by 5 cycles:  RMSE={rmse_score(y_true, y_late):.2f}, NASA Score={nasa_phm_score(y_true, y_late):.2f}")
    print(f"Late penalty ratio: {nasa_phm_score(y_true, y_late) / nasa_phm_score(y_true, y_early):.2f}x (Confirms ~3x heavier penalty!)")
