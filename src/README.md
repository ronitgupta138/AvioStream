# AvioStream — Core Architecture & Source Modules

Modular pipeline, high-throughput Project Loom ingestion engine, and deterministic CMMS work-order dispatch.

---

## Directory Index

| Module | Stack | Purpose |
|---|---|---|
| `pipeline/` | Python / NumPy | NASA C-MAPSS dataset loader, sensor invariant filtering, Savitzky-Golay & EMA signal conditioning, ring buffer |
| `ml/` | LightGBM / ONNX | Dual-stage prognostics core: unsupervised Anomaly/Health Index + supervised RUL regression with asymmetric NASA loss |
| `mlops/` | MLflow / SciPy | Statistical drift monitoring (Kolmogorov-Smirnov & PSI) and model registry lifecycle |
| `triage/` | Python | Deterministic 4-tier decision state machine and automated SAP PM / IBM Maximo JSON work-order generator |
| `java_engine/` | Java 21 Loom / JNI | High-throughput concurrent streaming ingestion buffer (>48k msgs/s) with embedded ONNX runtime |
| `dashboard/` | Streamlit | Real-time aerospace turbofan fleet command center, digital twin simulator, and live CMMS queue |
