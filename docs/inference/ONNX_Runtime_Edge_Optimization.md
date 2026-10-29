# ONNX Runtime Edge Optimization

## 1. Engine Architecture
AvioStream compiles LightGBM gradient-boosted decision trees and neural regressors into standard ONNX graph models with FP16/INT8 quantizations.

## 2. Edge Execution Characteristics
- Inference Latency: 0.014 ms per telemetry vector
- Memory Footprint: < 12 MB resident set size (RSS)
- Zero runtime Python dependencies on embedded telemetry node.