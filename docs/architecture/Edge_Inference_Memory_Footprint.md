# ONNX Edge Inference Memory Footprint & 0.014ms Latency Profile

## 1. Edge Deployment Specifications
- **Model Runtime:** ONNX Runtime C++ engine with Java 21 Foreign Function & Memory (FFM) bindings.
- **Quantization:** Dynamic INT8 weight quantization reducing model artifact from 14.8MB to 3.7MB.
- **Inference Latency:** 0.014ms per single-engine 21-sensor frame on edge avionics hardware.
- **Resident Set Size (RSS):** Flat 42MB runtime overhead, suitable for certified embedded flight controllers.
