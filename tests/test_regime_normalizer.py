"""Unit tests for the multi-regime telemetry normalizer and multi-dataset loader."""

import pytest
import numpy as np
import pandas as pd

from src.pipeline.regime_normalizer import OperatingRegimeNormalizer
from src.pipeline.dataset_loader import load_cmapss_subdataset, ACTIVE_SENSORS


class TestOperatingRegimeNormalizer:
    """Verifies operating regime clustering and sensor variance stabilization."""

    @pytest.fixture
    def synthetic_multiregime_data(self):
        """Generates synthetic dataset with 3 clear operational regimes."""
        np.random.seed(42)
        n_samples = 300
        # 3 regimes with distinct altitude and mach settings
        regimes = [
            (0.0, 0.0, 100.0, 1400.0),    # Sea Level
            (20.0, 0.7, 100.0, 1350.0),   # Mid altitude
            (40.0, 0.84, 100.0, 1300.0)   # High altitude
        ]

        rows = []
        for i in range(n_samples):
            r_idx = i % 3
            alt, mach, tra, base_temp = regimes[r_idx]
            rows.append({
                "unit_id": 1,
                "time_cycle": i + 1,
                "op_setting_1": alt + np.random.normal(0, 0.05),
                "op_setting_2": mach + np.random.normal(0, 0.005),
                "op_setting_3": tra,
                "s_2": 640.0 + r_idx * 5.0 + np.random.normal(0, 0.2),
                "s_4": base_temp + np.random.normal(0, 1.0)
            })
        return pd.DataFrame(rows)

    def test_regime_clustering_and_stabilization(self, synthetic_multiregime_data):
        """Tests that regime normalization eliminates ambient operational shift."""
        df = synthetic_multiregime_data
        sensors = ["s_2", "s_4"]

        raw_s4_var = df["s_4"].var()

        normalizer = OperatingRegimeNormalizer(n_regimes=3)
        norm_df = normalizer.fit_transform(df, sensors)

        # Post-normalization variance should be dramatically reduced toward 1.0
        norm_s4_var = norm_df["s_4"].var()
        assert norm_s4_var < raw_s4_var * 0.1, "Normalization should remove regime-dependent variance"
        assert "regime_id" in norm_df.columns
        assert len(norm_df["regime_id"].unique()) == 3

    def test_multi_dataset_loader_dimensions(self):
        """Verifies loading of FD002 and FD003 datasets."""
        tr2, te2, rul2 = load_cmapss_subdataset("FD002")
        assert tr2["unit_id"].nunique() == 260
        assert len(rul2) == 259

        tr3, te3, rul3 = load_cmapss_subdataset("FD003")
        assert tr3["unit_id"].nunique() == 100
        assert len(rul3) == 100
