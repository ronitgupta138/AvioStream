"""Unified Master Pipeline CLI Runner for StrataSense.

Commands:
  --train           Executes full data prep + LightGBM training + ONNX export
  --evaluate        Runs NASA C-MAPSS FD001 test set evaluation
  --drift-test      Simulates statistical telemetry drift evaluation
  --simulate-stream Replays streaming telemetry through FIFO buffer & triage engine
  --all             Executes end-to-end verification pipeline
"""

import sys
import argparse
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.dataset_loader import load_cmapss_fd001, ACTIVE_SENSORS
from src.ml.train_rul_regressor import prepare_training_features, train_lightgbm_model, save_artifacts
from src.ml.evaluate_models import evaluate_on_test_benchmark
from src.ml.export_onnx import export_lightgbm_to_onnx, benchmark_onnx_parity_and_latency
from src.ml.anomaly_detector import AnomalyDetectorAndHealthIndex
from src.mlops.drift_detector import StatisticalDriftDetector
from src.triage.decision_engine import DeterministicTriageEngine
from src.triage.work_order_generator import WorkOrderPayloadGenerator


def run_training():
    print("\n[1/3] Loading NASA C-MAPSS dataset...")
    train_df, _, _ = load_cmapss_fd001()
    print("[2/3] Extracting multi-scale rolling features...")
    df_feat, scaler, features = prepare_training_features(train_df)
    print("[3/3] Training LightGBM regressor with GroupKFold...")
    model, metrics = train_lightgbm_model(df_feat, features)
    save_artifacts(model, scaler, features, metrics)
    print("Training phase completed.")


def run_export_onnx():
    print("\nExporting model to portable ONNX format...")
    onnx_path = export_lightgbm_to_onnx()
    benchmark_onnx_parity_and_latency(onnx_path)
    print("ONNX export and latency benchmarking completed.")


def run_evaluation():
    print("\nRunning official test set benchmark evaluation...")
    metrics = evaluate_on_test_benchmark()
    return metrics


def run_drift_test():
    print("\nRunning statistical telemetry drift simulation...")
    train_df, _, _ = load_cmapss_fd001()
    detector = StatisticalDriftDetector()
    baseline = train_df[train_df["time_cycle"] <= 30]
    detector.fit_baseline(baseline, ACTIVE_SENSORS)

    stable_batch = train_df[(train_df["time_cycle"] > 30) & (train_df["time_cycle"] <= 60)]
    eval_stable = detector.evaluate_batch_drift(stable_batch, baseline)
    print(f"  Stable Batch PSI:   {eval_stable['mean_psi']:.4f} ({eval_stable['status']})")

    degraded_batch = train_df[train_df["time_cycle"] > 160]
    eval_degraded = detector.evaluate_batch_drift(degraded_batch, baseline)
    print(f"  Degraded Batch PSI: {eval_degraded['mean_psi']:.4f} ({eval_degraded['status']})")
    print("Drift evaluation completed.")


def run_streaming_simulation():
    print("\nSimulating real-time telemetry stream and auto-triage dispatch...")
    triage = DeterministicTriageEngine()
    wo_gen = WorkOrderPayloadGenerator()

    # Simulate 3 critical engine steps
    for cycle in [185, 190, 195]:
        rul = max(195 - cycle, 5.0)
        hi = 0.15
        decision = triage.evaluate_engine_state(42, cycle, rul, hi)
        if decision.severity.value == "CRITICAL":
            wo = wo_gen.generate_work_order(
                42, cycle, decision.severity.value, rul, hi,
                decision.action_required, decision.sla_hours
            )
            print(f"  Cycle {cycle}: Dispatched {wo['work_order_id']} | Action: {wo['action_required']} | SLA: {wo['sla_hours']}h")
    print("Streaming simulation completed.")


def main():
    parser = argparse.ArgumentParser(description="StrataSense Master Execution CLI")
    parser.add_argument("--train", action="store_true", help="Train LightGBM RUL regressor")
    parser.add_argument("--export-onnx", action="store_true", help="Export model to ONNX")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate test set benchmark")
    parser.add_argument("--drift-test", action="store_true", help="Simulate telemetry drift")
    parser.add_argument("--simulate-stream", action="store_true", help="Simulate streaming & auto triage")
    parser.add_argument("--all", action="store_true", help="Run complete end-to-end pipeline")

    args = parser.parse_args()

    if len(sys.argv) == 1 or args.all:
        print("=========================================================")
        print("         STRATASENSE MASTER VERIFICATION PIPELINE        ")
        print("=========================================================")
        start_time = time.time()
        run_training()
        run_export_onnx()
        run_evaluation()
        run_drift_test()
        run_streaming_simulation()
        elapsed = time.time() - start_time
        print(f"\nFULL PIPELINE EXECUTED SUCCESSFULLY IN {elapsed:.2f} SECONDS!")
        print("=========================================================")
    else:
        if args.train:
            run_training()
        if args.export_onnx:
            run_export_onnx()
        if args.evaluate:
            run_evaluation()
        if args.drift_test:
            run_drift_test()
        if args.simulate_stream:
            run_streaming_simulation()


if __name__ == "__main__":
    main()
