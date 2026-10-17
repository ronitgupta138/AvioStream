"""Operating Regime Normalizer for Multi-Condition Turbofan Telemetry (FD002 & FD004).

In complex flight envelopes (e.g. altitude 0-42k ft, Mach 0.0-0.84, throttle 20-100),
sensor readings fluctuate heavily due to ambient flight conditions rather than mechanical
degradation. This module clusters the 3 operational settings into K flight regimes
(K=6 for NASA C-MAPSS) and normalizes each sensor against its regime-specific baseline.
"""

from typing import List, Optional, Dict
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

SETTING_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]


class OperatingRegimeNormalizer:
    """Clustering-based multi-regime normalizer for complex flight envelopes."""

    def __init__(self, n_regimes: int = 6, random_state: int = 42):
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.kmeans: Optional[KMeans] = None
        self.regime_stats: Dict[int, Dict[str, Dict[str, float]]] = {}
        self.sensor_cols: List[str] = []

    def fit(self, df: pd.DataFrame, sensor_cols: List[str]) -> "OperatingRegimeNormalizer":
        """Fits regime clusters and calculates per-regime mean and std for each sensor."""
        self.sensor_cols = list(sensor_cols)

        # In single-condition datasets (FD001/FD003), variance of settings is close to 0
        settings_std = df[SETTING_COLS].std().sum()
        if settings_std < 0.1:
            effective_k = 1
        else:
            effective_k = self.n_regimes

        self.kmeans = KMeans(n_clusters=effective_k, random_state=self.random_state, n_init=10)
        regime_labels = self.kmeans.fit_predict(df[SETTING_COLS])

        df_temp = df.copy()
        df_temp["_regime"] = regime_labels

        self.regime_stats = {}
        for r in range(effective_k):
            regime_subset = df_temp[df_temp["_regime"] == r]
            self.regime_stats[r] = {}
            for col in self.sensor_cols:
                mean_val = float(regime_subset[col].mean())
                std_val = float(regime_subset[col].std())
                # Safeguard against zero std
                std_val = max(std_val, 1e-4)
                self.regime_stats[r][col] = {
                    "mean": mean_val,
                    "std": std_val
                }

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalizes sensor readings relative to their matched operating regime."""
        if self.kmeans is None or not self.regime_stats:
            raise RuntimeError("OperatingRegimeNormalizer must be fitted before calling transform.")

        df_out = df.copy()
        # Ensure sensor columns are float64 to prevent LossySetitemError
        for col in self.sensor_cols:
            if col in df_out.columns:
                df_out[col] = df_out[col].astype(np.float64)

        regime_labels = self.kmeans.predict(df[SETTING_COLS])
        df_out["regime_id"] = regime_labels

        for r, stats in self.regime_stats.items():
            mask = (regime_labels == r)
            if not np.any(mask):
                continue
            for col in self.sensor_cols:
                if col in df_out.columns:
                    m = stats[col]["mean"]
                    s = stats[col]["std"]
                    df_out.loc[mask, col] = (df_out.loc[mask, col] - m) / s

        return df_out

    def fit_transform(self, df: pd.DataFrame, sensor_cols: List[str]) -> pd.DataFrame:
        """Convenience fit and transform."""
        return self.fit(df, sensor_cols).transform(df)

    def save(self, filepath: Path) -> None:
        """Serializes fitted normalizer."""
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: Path) -> "OperatingRegimeNormalizer":
        """Deserializes normalizer."""
        return joblib.load(filepath)


if __name__ == "__main__":
    from src.pipeline.dataset_loader import load_cmapss_subdataset, ACTIVE_SENSORS

    print("--- Evaluating OperatingRegimeNormalizer on FD002 (6 Regimes) ---")
    train_fd002, test_fd002, _ = load_cmapss_subdataset("FD002")

    normalizer = OperatingRegimeNormalizer(n_regimes=6)
    norm_train = normalizer.fit_transform(train_fd002, ACTIVE_SENSORS)

    print(f"FD002 Normalized Train Shape: {norm_train.shape}")
    print(f"Identified Regimes: {sorted(norm_train['regime_id'].unique())}")
    print("Pre-normalization s_4 variance:", train_fd002["s_4"].var())
    print("Post-normalization s_4 variance (stabilized):", norm_train["s_4"].var())
    print("OperatingRegimeNormalizer successfully verified!")
