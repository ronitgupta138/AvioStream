"""Comprehensive unit test suite for StrataSense Phase 2: Machine Learning Engine."""

from pathlib import Path
import json
import joblib
import pytest
import numpy as np
import pandas as pd
import onnxruntime as ort

from src.pipeline.dataset_loader import load_cmapss_fd001, ACTIVE_SENSORS
from src.pipeline.signal_filter import RobustTelemetryScaler
from src.ml.anomaly_detector import AnomalyDetectorAndHealthIndex
from src.ml.metrics import rmse_score, mae_score, nasa_phm_score
from src.ml.explainability import ModelExplainabilityEngine

from src.config import MODELS_DIR


class TestMetrics:
    def test_asymmetric_nasa_penalty(self):
        y_true = np.array([50.0, 50.0])
        # Prediction 1 is 5 cycles early (d = -5)
        early_score = nasa_phm_score(y_true[:1], y_true[:1] - 5.0)
        # Prediction 2 is 5 cycles late (d = +5)
        late_score = nasa_phm_score(y_true[1:], y_true[1:] + 5.0)
        assert late_score > early_score
        assert np.isclose(late_score / early_score, 1.38, atol=0.05)


class TestAnomalyDetector:
    def test_health_index_bounds_and_degradation(self):
        train_df, _, _ = load_cmapss_fd001()
        scaler = RobustTelemetryScaler()
        scaled = scaler.fit_transform(train_df, feature_cols=ACTIVE_SENSORS, baseline_mask=train_df["time_cycle"] <= 30)

        detector = AnomalyDetectorAndHealthIndex()
        detector.fit(scaled, feature_cols=ACTIVE_SENSORS)

        u1 = scaled[scaled["unit_id"] == 1]
        hi, flags = detector.compute_health_index(u1)

        assert (hi >= 0.0).all() and (hi <= 1.0).all()
        # Initial health must be higher than end-of-life health
        assert hi[0] > hi[-1]
        assert len(flags) == len(u1)


class TestModelArtifacts:
    def test_artifacts_exist(self):
        assert (MODELS_DIR / "rul_lightgbm_best.joblib").exists()
        assert (MODELS_DIR / "stratasense_rul.onnx").exists()
        assert (MODELS_DIR / "feature_names.json").exists()
        assert (MODELS_DIR / "test_benchmark_metrics.json").exists()

    def test_onnx_parity_and_non_negative_predictions(self):
        model = joblib.load(MODELS_DIR / "rul_lightgbm_best.joblib")
        with open(MODELS_DIR / "feature_names.json") as f:
            features = json.load(f)

        session = ort.InferenceSession(str(MODELS_DIR / "stratasense_rul.onnx"), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name

        X_dummy = np.random.normal(0, 1, size=(5, len(features))).astype(np.float32)
        native_pred = model.predict(X_dummy)
        onnx_pred = session.run(None, {input_name: X_dummy})[0].flatten()

        assert np.allclose(native_pred, onnx_pred, atol=1e-4)

    def test_explainability_attribution(self):
        engine = ModelExplainabilityEngine(models_dir=MODELS_DIR)
        top_drivers = engine.get_top_global_features(5)
        assert len(top_drivers) == 5

        # Check attribution structure
        dummy_row = pd.Series({f: 1.0 for f in engine.feature_names})
        attr = engine.attribute_root_cause(dummy_row)
        assert "primary_subsystem" in attr
        assert "critical_sensors" in attr
        assert len(attr["critical_sensors"]) <= 3
