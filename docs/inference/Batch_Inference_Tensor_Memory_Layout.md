# Batch Inference Tensor Memory Alignment & Vector Packing

## 1. Memory Alignment
- Allocates contiguous row-major memory buffers aligned to 64-byte boundaries (cache line size).
- Zero-copy tensor wrapping directly into ONNX Runtime input bindings.
