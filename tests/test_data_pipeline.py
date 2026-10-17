"""Comprehensive unit test suite for StrataSense Phase 1: Data Engineering Pipeline."""

import pytest
import numpy as np
import pandas as pd

from src.pipeline.dataset_loader import (
    load_cmapss_fd001,
    ACTIVE_SENSORS,
    INVARIANT_SENSORS,
    BASE_COLS
)
from src.pipeline.signal_filter import (
    SavitzkyGolayFilter,
    StreamingEMAFilter,
    RobustTelemetryScaler
)
from src.pipeline.feature_extractor import MultiScaleRollingFeatureExtractor
from src.pipeline.stream_buffer import UnitRingBuffer, FleetStreamBufferManager


class TestDatasetLoader:
    def test_load_cmapss_fd001_dimensions(self):
        train_df, test_df, true_rul_df = load_cmapss_fd001()
        assert train_df.shape[0] == 20631
        assert test_df.shape[0] == 13096
        assert len(true_rul_df) == 100
        assert train_df["unit_id"].nunique() == 100
        assert test_df["unit_id"].nunique() == 100

    def test_invariant_sensors_pruned(self):
        train_df, test_df, _ = load_cmapss_fd001(prune_invariant=True)
        for s in INVARIANT_SENSORS:
            assert s not in train_df.columns
            assert s not in test_df.columns
        for s in ACTIVE_SENSORS:
            assert s in train_df.columns
            assert s in test_df.columns

    def test_rul_clipped_target(self):
        train_df, _, _ = load_cmapss_fd001(rul_clip=125)
        assert train_df["RUL_clipped"].max() == 125
        assert (train_df["RUL_clipped"] <= train_df["RUL_actual"]).all()
        assert (train_df["RUL_clipped"] >= 0).all()


class TestSignalFilters:
    def test_savitzky_golay_filter(self):
        sg = SavitzkyGolayFilter(window_length=7, polyorder=2)
        # Synthetic noisy sine wave
        t = np.linspace(0, 10, 50)
        clean = np.sin(t)
        noisy = clean + np.random.normal(0, 0.2, size=50)
        s = pd.Series(noisy)
        filtered = sg.filter_series(s)
        assert len(filtered) == 50
        assert not filtered.isna().any()
        # Filtered variance should be lower than noisy variance
        assert (filtered - clean).var() < (noisy - clean).var()

    def test_streaming_ema_filter(self):
        ema = StreamingEMAFilter(alpha=0.2)
        s = pd.Series([10.0, 10.0, 10.0, 20.0, 20.0])
        filtered = ema.filter_series(s)
        assert filtered.iloc[0] == 10.0
        assert filtered.iloc[-1] > 10.0
        assert not filtered.isna().any()

    def test_robust_scaler(self):
        scaler = RobustTelemetryScaler(clip_range=3.0)
        df = pd.DataFrame({
            "unit_id": [1] * 10,
            "s_2": [10.0, 11.0, 10.5, 10.2, 10.8, 11.2, 10.1, 10.9, 10.4, 50.0] # 50 is outlier
        })
        scaled = scaler.fit_transform(df, feature_cols=["s_2"])
        assert scaled["s_2"].max() <= 3.0
        assert scaled["s_2"].min() >= -3.0
        assert not scaled["s_2"].isna().any()


class TestFeatureExtractor:
    def test_multi_scale_rolling_extractor(self):
        df = pd.DataFrame({
            "unit_id": [1] * 20 + [2] * 20,
            "time_cycle": list(range(1, 21)) * 2,
            "s_2": list(range(20)) + list(range(20, 40))
        })
        extractor = MultiScaleRollingFeatureExtractor(windows=[5, 10])
        df_out, feats = extractor.extract_features(df, sensor_cols=["s_2"])
        
        # 1 sensor * 4 features * 2 windows = 8 features
        assert len(feats) == 8
        assert df_out[feats].isna().sum().sum() == 0
        
        # Check slope for unit 1 with linear growth of 1/cycle
        assert np.isclose(df_out[df_out["unit_id"] == 1]["s_2_slope_w5"].iloc[10], 1.0, atol=0.2)


class TestStreamBuffer:
    def test_unit_ring_buffer_wrapping(self):
        buf = UnitRingBuffer(capacity=5, num_features=2)
        assert not buf.is_full
        
        for i in range(1, 8):
            buf.push(cycle=i, features=np.array([float(i), float(i * 10)]))
            
        assert buf.is_full
        assert buf.latest_cycle == 7
        data, cycles = buf.get_chronological_window()
        assert len(cycles) == 5
        assert list(cycles) == [3, 4, 5, 6, 7]
        assert data[-1, 0] == 7.0

    def test_fleet_manager_memory_budget(self):
        manager = FleetStreamBufferManager(capacity=30, num_features=14)
        for unit in range(1, 101):
            for c in range(1, 35):
                manager.ingest_snapshot(unit, c, np.ones(14))
                
        assert len(manager.fleet_buffers) == 100
        mem_bytes = manager.total_memory_bytes()
        # Must be well under 1.8 MB (1,887,436 bytes)
        assert mem_bytes < 1_887_436
