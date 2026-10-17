"""Multi-scale temporal rolling feature extractor for turbofan sensor telemetry.

Generates rolling statistical moments (mean, std, min-max spread) and
thermodynamic degradation slopes across multi-scale sliding windows W in {5, 15, 30}.
Optimized for zero fragmentation using batched concatenation.
"""

from typing import List, Tuple
import pandas as pd
import numpy as np


class MultiScaleRollingFeatureExtractor:
    """Extracts temporal rolling features per engine unit to avoid cross-unit leakage.

    Windows: W in {5, 15, 30} cycles.
    Moments:
      - mean_{W}: Rolling average (low-pass filter trend)
      - std_{W}: Rolling volatility / mechanical vibration variance
      - spread_{W}: max_{W} - min_{W} (envelope amplitude)
      - slope_{W}: (s_t - s_{t-W+1}) / W (thermodynamic divergence rate)
    """

    def __init__(self, windows: List[int] = [5, 15, 30]):
        self.windows = sorted(windows)

    def extract_features(
        self,
        df: pd.DataFrame,
        sensor_cols: List[str],
        group_col: str = "unit_id",
        time_col: str = "time_cycle"
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Extracts rolling features grouped by engine unit.

        Batches all new columns into a dictionary to avoid DataFrame fragmentation.
        """
        df_sorted = df.sort_values([group_col, time_col]).copy()
        feature_dict = {}
        generated_features: List[str] = []

        # Convert sensors to float64
        for col in sensor_cols:
            df_sorted[col] = df_sorted[col].astype(np.float64)

        for w in self.windows:
            grouped = df_sorted.groupby(group_col)[sensor_cols]

            # 1. Rolling Mean
            rolling_mean = grouped.rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True)
            for c in sensor_cols:
                col_name = f"{c}_mean_w{w}"
                feature_dict[col_name] = rolling_mean[c]
                generated_features.append(col_name)

            # 2. Rolling Std Dev
            rolling_std = grouped.rolling(window=w, min_periods=1).std().reset_index(level=0, drop=True).fillna(0.0)
            for c in sensor_cols:
                col_name = f"{c}_std_w{w}"
                feature_dict[col_name] = rolling_std[c]
                generated_features.append(col_name)

            # 3. Rolling Spread (Max - Min)
            rolling_max = grouped.rolling(window=w, min_periods=1).max().reset_index(level=0, drop=True)
            rolling_min = grouped.rolling(window=w, min_periods=1).min().reset_index(level=0, drop=True)
            for c in sensor_cols:
                col_name = f"{c}_spread_w{w}"
                feature_dict[col_name] = rolling_max[c] - rolling_min[c]
                generated_features.append(col_name)

            # 4. Degradation Slope
            for c in sensor_cols:
                col_name = f"{c}_slope_w{w}"
                lagged = df_sorted.groupby(group_col)[c].shift(w - 1)
                fallback_lagged = df_sorted.groupby(group_col)[c].transform("first")
                effective_lagged = lagged.fillna(fallback_lagged)
                effective_w = np.maximum(df_sorted[time_col] - 1, 1).clip(upper=w)
                feature_dict[col_name] = (df_sorted[c] - effective_lagged) / effective_w
                generated_features.append(col_name)

        # Single batched concatenation - zero fragmentation
        new_features_df = pd.DataFrame(feature_dict, index=df_sorted.index)
        df_out = pd.concat([df_sorted, new_features_df], axis=1)

        return df_out, generated_features


if __name__ == "__main__":
    from src.pipeline.dataset_loader import load_cmapss_fd001, ACTIVE_SENSORS

    train_df, test_df, _ = load_cmapss_fd001()
    extractor = MultiScaleRollingFeatureExtractor(windows=[5, 15, 30])
    train_feat, feats = extractor.extract_features(train_df, sensor_cols=ACTIVE_SENSORS)

    print(f"Original shape: {train_df.shape}")
    print(f"Extracted shape: {train_feat.shape}")
    print(f"Generated {len(feats)} rolling features across 14 active sensors.")
    print(f"Null count in feature matrix: {train_feat[feats].isna().sum().sum()}")
    print("Sample feature columns:", feats[:8])
