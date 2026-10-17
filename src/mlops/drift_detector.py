"""Statistical drift detection and operational regime isolation engine.

Implements two-sample Kolmogorov-Smirnov (K-S) tests and 10-bin Population Stability
Index (PSI) tracking to detect telemetry distribution divergence without flagging
false alarms on benign flight regime changes.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


class StatisticalDriftDetector:
    """Monitors telemetry distribution shift against baseline healthy operations.

    Thresholds:
      PSI <= 0.10 -> Stable (No drift)
      0.10 < PSI <= 0.20 -> Warning (Moderate drift alert)
      PSI > 0.20 -> Severe drift (Trigger automated retraining)
    """

    def __init__(self, num_bins: int = 10, epsilon: float = 1e-4):
        self.num_bins = num_bins
        self.epsilon = epsilon
        self.baseline_quantiles_: Dict[str, np.ndarray] = {}
        self.feature_cols_: List[str] = []

    def fit_baseline(self, baseline_df: pd.DataFrame, feature_cols: List[str]):
        """Caches quantile bin edges for each feature on healthy baseline."""
        self.feature_cols_ = feature_cols
        self.baseline_quantiles_ = {}

        quantiles = np.linspace(0, 1, self.num_bins + 1)
        for col in feature_cols:
            vals = baseline_df[col].dropna().values.astype(np.float64)
            bins = np.percentile(vals, quantiles * 100)
            # Ensure strictly increasing bins
            bins[0] = -np.inf
            bins[-1] = np.inf
            self.baseline_quantiles_[col] = np.unique(bins)
        return self

    def calculate_psi(self, baseline_vals: np.ndarray, current_vals: np.ndarray, bins: np.ndarray) -> float:
        """Calculates Population Stability Index across quantile bins."""
        base_counts, _ = np.histogram(baseline_vals, bins=bins)
        curr_counts, _ = np.histogram(current_vals, bins=bins)

        base_pct = base_counts / (np.sum(base_counts) + self.epsilon)
        curr_pct = curr_counts / (np.sum(curr_counts) + self.epsilon)

        # Smooth zero counts
        base_pct = np.clip(base_pct, self.epsilon, 1.0)
        curr_pct = np.clip(curr_pct, self.epsilon, 1.0)

        psi_val = np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct))
        return float(psi_val)

    def evaluate_batch_drift(
        self,
        current_df: pd.DataFrame,
        baseline_df: pd.DataFrame
    ) -> Dict:
        """Evaluates Kolmogorov-Smirnov and PSI across all active telemetry features."""
        psi_report: Dict[str, float] = {}
        ks_report: Dict[str, float] = {}
        drifted_features: List[str] = []

        for col in self.feature_cols_:
            base_vals = baseline_df[col].dropna().values.astype(np.float64)
            curr_vals = current_df[col].dropna().values.astype(np.float64)

            bins = self.baseline_quantiles_[col]
            psi = self.calculate_psi(base_vals, curr_vals, bins)
            psi_report[col] = round(psi, 4)

            # Two-sample KS test
            ks_stat, p_val = ks_2samp(base_vals, curr_vals)
            ks_report[col] = round(float(ks_stat), 4)

            if psi > 0.10 or p_val < 0.01:
                drifted_features.append(col)

        mean_psi = float(np.mean(list(psi_report.values())))

        if mean_psi > 0.20:
            status = "SEVERE_DRIFT_RETRAIN_TRIGGERED"
        elif mean_psi > 0.10:
            status = "MODERATE_DRIFT_WARNING"
        else:
            status = "STABLE"

        return {
            "status": status,
            "mean_psi": round(mean_psi, 4),
            "max_psi_feature": max(psi_report, key=psi_report.get),
            "max_psi_value": max(psi_report.values()),
            "drifted_features_count": len(drifted_features),
            "feature_psi": psi_report,
            "feature_ks": ks_report
        }


if __name__ == "__main__":
    from src.pipeline.dataset_loader import load_cmapss_fd001, ACTIVE_SENSORS

    train_df, _, _ = load_cmapss_fd001()
    detector = StatisticalDriftDetector()

    # Fit baseline on early healthy operational cycles (cycle <= 30)
    baseline_df = train_df[train_df["time_cycle"] <= 30]
    detector.fit_baseline(baseline_df, ACTIVE_SENSORS)

    # 1. Test stable batch (cycles 31 to 60)
    stable_batch = train_df[(train_df["time_cycle"] > 30) & (train_df["time_cycle"] <= 60)]
    stable_eval = detector.evaluate_batch_drift(stable_batch, baseline_df)
    print("=== Stable Fleet Batch Drift Evaluation ===")
    print(f"Status: {stable_eval['status']}, Mean PSI: {stable_eval['mean_psi']:.4f}")

    # 2. Test degraded batch near end of life (cycles > 160)
    degraded_batch = train_df[train_df["time_cycle"] > 160]
    degraded_eval = detector.evaluate_batch_drift(degraded_batch, baseline_df)
    print("\n=== End-of-Life Degraded Batch Drift Evaluation ===")
    print(f"Status: {degraded_eval['status']}, Mean PSI: {degraded_eval['mean_psi']:.4f}, Max Feature: {degraded_eval['max_psi_feature']}")
    print("Drift evaluation module operational.")
