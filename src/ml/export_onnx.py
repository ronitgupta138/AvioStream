"""Portable ONNX model serialization and runtime benchmarking.

Converts trained LightGBM model to ONNX format, validates graph structure,
and verifies zero-loss numerical parity against native Python inference.
"""

from pathlib import Path
import time
import json
import joblib
import numpy as np
import onnx
import onnxruntime as ort
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

from src.config import MODELS_DIR


def export_lightgbm_to_onnx(
    models_dir: Path = MODELS_DIR,
    onnx_filename: str = "aviostream_rul.onnx"
) -> Path:
    """Exports trained LightGBM model to standard ONNX binary."""
    model = joblib.load(models_dir / "rul_lightgbm_best.joblib")
    with open(models_dir / "feature_names.json") as f:
        feature_names = json.load(f)

    num_features = len(feature_names)
    initial_type = [("float_input", FloatTensorType([None, num_features]))]

    print(f"Exporting LightGBM model ({num_features} input features) to ONNX...")
    onnx_model = onnxmltools.convert_lightgbm(
        model,
        initial_types=initial_type,
        target_opset=14
    )

    out_path = models_dir / onnx_filename
    onnxmltools.utils.save_model(onnx_model, str(out_path))

    # Verify model with official ONNX checker
    checker_model = onnx.load(str(out_path))
    onnx.checker.check_model(checker_model)
    print(f"ONNX graph verified and saved to: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    return out_path


def benchmark_onnx_parity_and_latency(
    onnx_path: Path,
    models_dir: Path = MODELS_DIR,
    num_runs: int = 1000
) -> dict:
    """Checks numerical parity against native LightGBM and benchmarks CPU inference latency."""
    model = joblib.load(models_dir / "rul_lightgbm_best.joblib")
    with open(models_dir / "feature_names.json") as f:
        feature_names = json.load(f)

    # Synthetic test batch (10 samples)
    np.random.seed(42)
    X_sample = np.random.normal(0, 1, size=(10, len(feature_names))).astype(np.float32)

    # 1. Native LightGBM prediction
    pred_native = model.predict(X_sample)

    # 2. ONNX Runtime inference
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    pred_onnx = session.run(None, {input_name: X_sample})[0].flatten()

    # 3. Check numerical parity
    max_abs_diff = float(np.max(np.abs(pred_native - pred_onnx)))
    assert max_abs_diff < 1e-4, f"Parity check failed: max delta = {max_abs_diff}"

    # 4. Latency benchmark (single sample inference)
    x_single = X_sample[:1]
    # Warmup
    for _ in range(50):
        _ = session.run(None, {input_name: x_single})

    start = time.perf_counter()
    for _ in range(num_runs):
        _ = session.run(None, {input_name: x_single})
    elapsed = time.perf_counter() - start
    avg_latency_ms = (elapsed / num_runs) * 1000.0

    print("\n--- ONNX Runtime Benchmarks ---")
    print(f"Max absolute discrepancy: {max_abs_diff:.6e} (PASS)")
    print(f"Single-cycle latency:      {avg_latency_ms:.3f} ms across {num_runs} forward passes")
    print(f"Edge Latency Target (< 2.0 ms): {'PASSED' if avg_latency_ms < 2.0 else 'NEEDS_OPTIMIZATION'}")

    results = {
        "max_abs_diff": max_abs_diff,
        "avg_latency_ms": avg_latency_ms,
        "num_features": len(feature_names),
        "target_met": avg_latency_ms < 2.0
    }
    return results


if __name__ == "__main__":
    out_file = export_lightgbm_to_onnx()
    benchmark_onnx_parity_and_latency(out_file)
