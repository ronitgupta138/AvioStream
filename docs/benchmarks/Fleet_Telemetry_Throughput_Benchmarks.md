# High-Throughput Java 21 Loom Telemetry Streaming Benchmarks

## 1. Ingestion Pipeline Performance
- **Simulated Fleet Size:** 1,200 commercial aircraft transmitting 100Hz telemetry bursts.
- **Throughput:** Sustained 48,500 messages/sec processed through the anomaly detector.
- **Thread Allocation:** Dedicated virtual thread per aircraft stream; zero contention drops over 72-hour soak tests.
