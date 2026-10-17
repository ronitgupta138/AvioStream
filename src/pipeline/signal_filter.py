"""Signal noise filtering, exponential smoothing, and robust scaling

for turbofan fleet telemetry.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin


class SavitzkyGolayFilter:
    """Applies Savitzky-Golay polynomial filtering to smooth high-frequency

    telemetry fluctuations while preserving true mechanical inflection points.
    """

    def __init__(self, window_length: int = 7, polyorder: int = 2):
        if window_length % 2 == 0:
            raise ValueError(f"window_length must be odd, got {window_length}")
        self.window_length = window_length
        self.polyorder = polyorder

    def filter_series(self, series: pd.Series) -> pd.Series:
        """Filters a single 1D trajectory."""
        n = len(series)
        if n < self.window_length:
            # Fall back to raw values or smaller window if trajectory is shorter than window
            return series.copy()
        smoothed = savgol_filter(series.values, window_length=self.window_length, polyorder=self.polyorder)
        return pd.Series(smoothed, index=series.index)

    def filter_dataframe(self, df: pd.DataFrame, feature_cols: List[str], group_col: str = "unit_id") -> pd.DataFrame:
        """Applies smoothing per engine trajectory to avoid inter-engine contamination."""
        df_out = df.copy()
        # Cast feature columns to float64 to allow smooth non-integer filtered values
        for col in feature_cols:
            df_out[col] = df_out[col].astype(np.float64)

        for unit_id, group in df.groupby(group_col):
            if len(group) >= self.window_length:
                for col in feature_cols:
                    df_out.loc[group.index, col] = savgol_filter(
                        group[col].values.astype(np.float64),
                        window_length=self.window_length,
                        polyorder=self.polyorder
                    )
        return df_out


class StreamingEMAFilter:
    """Real-time Exponential Moving Average filter for stream processing:

    y_t = alpha * x_t + (1 - alpha) * y_{t-1}
    """

    def __init__(self, alpha: float = 0.20):
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha

    def filter_series(self, series: pd.Series) -> pd.Series:
        return series.ewm(alpha=self.alpha, adjust=False).mean()

    def filter_dataframe(self, df: pd.DataFrame, feature_cols: List[str], group_col: str = "unit_id") -> pd.DataFrame:
        df_out = df.copy()
        for col in feature_cols:
            df_out[col] = df_out[col].astype(np.float64)

        for unit_id, group in df.groupby(group_col):
            for col in feature_cols:
                df_out.loc[group.index, col] = group[col].astype(np.float64).ewm(alpha=self.alpha, adjust=False).mean()
        return df_out


class RobustTelemetryScaler(BaseEstimator, TransformerMixin):
    """Interquartile Range (IQR) robust normalization:

    z = (x - median) / (Q3 - Q1)
    Fitted strictly on baseline healthy operational data to avoid outlier distortion.
    """

    def __init__(self, clip_range: Optional[float] = 5.0):
        self.clip_range = clip_range
        self.medians_: Optional[pd.Series] = None
        self.iqrs_: Optional[pd.Series] = None
        self.feature_cols_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, feature_cols: List[str], baseline_mask: Optional[pd.Series] = None):
        """Fits median and IQR on healthy baseline cycles."""
        self.feature_cols_ = feature_cols
        data = X[baseline_mask] if baseline_mask is not None else X

        self.medians_ = data[feature_cols].median()
        q25 = data[feature_cols].quantile(0.25)
        q75 = data[feature_cols].quantile(0.75)
        iqr = q75 - q25

        # Prevent division by zero for near-constant features
        iqr = iqr.replace(0.0, 1.0)
        self.iqrs_ = iqr
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.medians_ is None or self.iqrs_ is None or self.feature_cols_ is None:
            raise RuntimeError("Scaler must be fitted before transform")

        X_out = X.copy()
        for col in self.feature_cols_:
            scaled = (X_out[col] - self.medians_[col]) / self.iqrs_[col]
            if self.clip_range is not None:
                scaled = scaled.clip(lower=-self.clip_range, upper=self.clip_range)
            X_out[col] = scaled
        return X_out

    def fit_transform(self, X: pd.DataFrame, feature_cols: List[str], baseline_mask: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(X, feature_cols, baseline_mask).transform(X)


if __name__ == "__main__":
    from src.pipeline.dataset_loader import load_cmapss_fd001, ACTIVE_SENSORS

    train_df, test_df, _ = load_cmapss_fd001()
    print(f"Original train shape: {train_df.shape}")

    # 1. Test Savitzky-Golay
    sg = SavitzkyGolayFilter(window_length=7, polyorder=2)
    train_sg = sg.filter_dataframe(train_df, feature_cols=ACTIVE_SENSORS)
    print("Savitzky-Golay smoothed sample (s_9):", train_sg['s_9'].head().values)

    # 2. Test EMA
    ema = StreamingEMAFilter(alpha=0.20)
    train_ema = ema.filter_dataframe(train_df, feature_cols=ACTIVE_SENSORS)
    print("EMA smoothed sample (s_9):", train_ema['s_9'].head().values)

    # 3. Test Robust Scaler fitted on healthy baseline (cycles <= 30)
    baseline_mask = train_df['time_cycle'] <= 30
    scaler = RobustTelemetryScaler(clip_range=5.0)
    train_scaled = scaler.fit_transform(train_df, feature_cols=ACTIVE_SENSORS, baseline_mask=baseline_mask)
    print("Scaled feature sample (s_9):", train_scaled['s_9'].head().values)
    print("All signal conditioning modules executed successfully.")
