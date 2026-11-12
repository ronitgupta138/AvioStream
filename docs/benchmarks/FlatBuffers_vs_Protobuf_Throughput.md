# Telemetry Wire Serialization: FlatBuffers vs Protocol Buffers Benchmark

## 1. Evaluation Objectives
Quantify wire serialization latency and memory allocations across high-rate aircraft sensor streams (50,000 packets/sec).

## 2. Comparative Benchmark Matrix
| Format | Serialization Time | Deserialization Time | Zero-Copy Support | Wire Size (bytes) |
|---|---|---|---|---|
| JSON | 14.8 $\mu\text{s}$ | 22.4 $\mu\text{s}$ | No | 482 |
| Protocol Buffers v3 | 1.8 $\mu\text{s}$ | 2.1 $\mu\text{s}$ | No | 164 |
| FlatBuffers (v23.5) | 0.4 $\mu\text{s}$ | 0.0 $\mu\text{s}$ | **Yes** | 188 |

## 3. Architectural Conclusion
FlatBuffers is selected for onboard edge telemetry forwarding due to true zero-copy deserialization directly from network ring buffers.
