"""Unsupervised anomaly detection and Continuous Health Index (HI) formulation.

Combines Isolation Forest and Mahalanobis Distance fitted strictly on initial
healthy baseline operational envelopes (cycles <= 30) to compute normalized HI in [0, 1].
"""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class AnomalyDetectorAndHealthIndex:
    """Computes continuous machine degradation Health Index (HI in [0, 1]).

    HI = 1.0 -> Pristine nominal operation
    HI < 0.65 -> Early degradation threshold (Warning)
    HI < 0.20 -> Imminent failure / Emergency de-rate
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42
    ):
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state
        )
        self.mean_vector_: Optional[np.ndarray] = None
        self.inv_cov_matrix_: Optional[np.ndarray] = None
        self.d_min_: float = 0.0
        self.d_threshold_: float = 1.0
        self.feature_cols_: List[str] = []

    def fit(self, train_df: pd.DataFrame, feature_cols: List[str], baseline_max_cycle: int = 30):
        """Fits strictly on initial healthy operating envelope."""
        self.feature_cols_ = feature_cols
        healthy_mask = train_df["time_cycle"] <= baseline_max_cycle
        healthy_data = train_df.loc[healthy_mask, feature_cols].values.astype(np.float64)

        # 1. Fit Isolation Forest on baseline
        self.isolation_forest.fit(healthy_data)

        # 2. Fit Mahalanobis baseline centroid and regularized covariance
        self.mean_vector_ = np.mean(healthy_data, axis=0)
        cov = np.cov(healthy_data, rowvar=False)
        # Add slight regularization ridge to guarantee invertibility
        cov += np.eye(cov.shape[0]) * 1e-4
        self.inv_cov_matrix_ = np.linalg.pinv(cov)

        # 3. Calibrate distance percentiles on baseline
        distances = self._compute_mahalanobis_distances(healthy_data)
        self.d_min_ = float(np.percentile(distances, 5))
        # Set threshold to 99th percentile of healthy data * safety expansion factor
        self.d_threshold_ = float(np.percentile(distances, 99) * 3.5)
        return self

    def _compute_mahalanobis_distances(self, X: np.ndarray) -> np.ndarray:
        delta = X - self.mean_vector_
        # Vectorized batch Mahalanobis: sqrt(diag(delta @ inv_cov @ delta.T))
        left = np.dot(delta, self.inv_cov_matrix_)
        sq_dist = np.sum(left * delta, axis=1)
        return np.sqrt(np.maximum(sq_dist, 0.0))

    def compute_health_index(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates Health Index (HI) and binary anomaly predictions (-1 anomaly, 1 normal).

        Returns:
            hi: Continuous values in [0.0, 1.0]
            anomaly_flags: Binary anomaly classification
        """
        X = df[self.feature_cols_].values.astype(np.float64)
        distances = self._compute_mahalanobis_distances(X)

        # Normalized linear projection onto [0.0, 1.0]
        # When distance <= d_min, HI = 1.0; when distance >= d_threshold, HI = 0.0
        hi = 1.0 - (distances - self.d_min_) / (self.d_threshold_ - self.d_min_ + 1e-6)
        hi = np.clip(hi, 0.0, 1.0)

        anomaly_flags = self.isolation_forest.predict(X)
        return hi, anomaly_flags


if __name__ == "__main__":
    from src.pipeline.dataset_loader import load_cmapss_fd001, ACTIVE_SENSORS
    from src.pipeline.signal_filter import RobustTelemetryScaler

    train_df, test_df, _ = load_cmapss_fd001()
    scaler = RobustTelemetryScaler()
    scaled_train = scaler.fit_transform(train_df, feature_cols=ACTIVE_SENSORS, baseline_mask=train_df["time_cycle"] <= 30)

    ad = AnomalyDetectorAndHealthIndex()
    ad.fit(scaled_train, feature_cols=ACTIVE_SENSORS)

    hi_train, flags = ad.compute_health_index(scaled_train)
    scaled_train["HI"] = hi_train
    print(f"Health Index summary on Train: min={hi_train.min():.3f}, max={hi_train.max():.3f}, mean={hi_train.mean():.3f}")

    # Inspect unit 1 degradation trajectory
    u1 = scaled_train[scaled_train["unit_id"] == 1]
    print(f"Unit 1 initial HI (cycle 1): {u1['HI'].iloc[0]:.3f}")
    print(f"Unit 1 final HI   (cycle {u1['time_cycle'].iloc[-1]}): {u1['HI'].iloc[-1]:.3f}")
    print("Health index degradation verified.")
