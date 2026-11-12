# Dynamic INT8 Quantization Specification for Edge Telemetry Gateways

## 1. Target Hardware Constraints
Airborne IoT telemetry gateways operate within constrained thermal and memory envelopes, requiring optimized ML inference runtimes.

## 2. Quantization Scale & Zero-Point Mapping
Floating point weights $W_{fp32}$ and inputs $X_{fp32}$ are mapped to 8-bit integers:

$$q = \text{clamp}\left(\text{round}\left(\frac{x}{S}\right) + Z, -128, 127\right)$$

Where scale factor $S = \frac{x_{\max} - x_{\min}}{255}$ and zero-point $Z = -\text{round}(x_{\min} / S) - 128$.

## 3. Latency & Memory Reduction
- **Memory Footprint:** 74% reduction compared to standard FP32 model weights.
- **Edge Inference Latency:** Sub-0.02ms execution time per telemetry frame using ONNX Runtime INT8 SIMD kernels.
