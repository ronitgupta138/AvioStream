# System Architecture Artifacts

Complete architectural schemas, dataflows, and protocol definitions for the StrataSense Predictive Maintenance Platform.

---

## High-Level Topology

```
+--------------------------------------------------------------------------------------------------+
|                                     STRATASENSE ARCHITECTURE                                     |
+--------------------------------------------------------------------------------------------------+
|                                                                                                  |
|   OFFLINE EXPERIMENTATION (Python 3.11)                  ONLINE EDGE INGESTION & TRIAGE (Java 21) |
|   +------------------------------------+                 +-----------------------------------+   |
|   | NASA C-MAPSS Telemetry Dataset     |                 | Live Fleet Sensor Telemetry       |   |
|   +------------------------------------+                 +-----------------------------------+   |
|                     |                                                      |                     |
|                     v                                                      v                     |
|   +------------------------------------+                 +-----------------------------------+   |
|   | Preprocessing & Feature Extraction |                 | Concurrent Ingestion Engine       |   |
|   | • 7 Flatline Sensors Pruned        |                 | • Virtual Threads (Project Loom)  |   |
|   | • Savitzky-Golay + EMA Jitter Cut  |                 | • Throughput: > 10,000 msgs/sec   |   |
|   | • Rolling Windows W={5, 15, 30}    |                 | • FIFO Ring Buffer (< 1.8 MB/unit)|   |
|   +------------------------------------+                 +-----------------------------------+   |
|                     |                                                      |                     |
|                     v                                                      v                     |
|   +------------------------------------+    ONNX Export  +-----------------------------------+   |
|   | Dual-Stage Model Training          | --------------->| Embedded ONNX Runtime Engine      |   |
|   | • Branch 1: Isolation Forest (HI)  |   (.onnx graph) | • onnxruntime-java (JNI)          |   |
|   | • Branch 2: LightGBM Regressor     |                 | • Inference Latency: < 2ms / cyc  |   |
|   | • Asymmetric NASA Loss (3x penalty)|                 | • Zero Python GIL Bottlenecks     |   |
|   +------------------------------------+                 +-----------------------------------+   |
|                     |                                                      |                     |
|                     v                                                      v                     |
|   +------------------------------------+                 +-----------------------------------+   |
|   | MLOps Lifecycle (MLflow)           |                 | Deterministic Triage Agent        |   |
|   | • Parameter & Metric Logging       |                 | • 🟢 NORMAL   (>80 cyc | 10s poll)|   |
|   | • Model Registry Staging Gates     |                 | • 🟡 MONITOR  (41-80 cyc | 1s poll|   |
|   | • Drift Engine (K-S & PSI tests)   |                 | • 🟠 SCHEDULE (16-40 cyc | 72h SLA|   |
|   | • Automated Retraining Loop        |                 | • 🔴 CRITICAL (≤15 cyc | De-rate) |   |
|   +------------------------------------+                 +-----------------------------------+   |
|                                                                            |                     |
|                                                                            v                     |
|                                                          +-----------------------------------+   |
|                                                          | Automated CMMS Work Order Dispatch|   |
|                                                          | • Sub-15ms Programmatic JSON      |   |
|                                                          | • SAP PM / IBM Maximo API Targets |   |
|                                                          +-----------------------------------+   |
+--------------------------------------------------------------------------------------------------+
```

---

## Automated CMMS Work-Order Payload Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "StrataSenseMaintenanceWorkOrder",
  "type": "object",
  "required": [
    "work_order_id",
    "timestamp_utc",
    "unit_id",
    "operational_cycle",
    "severity",
    "rul_prediction",
    "health_index",
    "root_cause",
    "action_required",
    "sla_hours"
  ],
  "properties": {
    "work_order_id": { "type": "string", "example": "WO-20261006-0042" },
    "timestamp_utc": { "type": "string", "format": "date-time" },
    "unit_id": { "type": "integer", "example": 42 },
    "operational_cycle": { "type": "integer", "example": 192 },
    "severity": {
      "type": "string",
      "enum": ["NORMAL", "MONITOR", "SCHEDULED", "CRITICAL"]
    },
    "rul_prediction": { "type": "number", "example": 14.2 },
    "health_index": { "type": "number", "minimum": 0.0, "maximum": 1.0, "example": 0.18 },
    "root_cause": {
      "type": "object",
      "properties": {
        "primary_subsystem": { "type": "string", "example": "High_Pressure_Compressor" },
        "critical_sensors": {
          "type": "array",
          "items": { "type": "string" },
          "example": ["s11", "s12", "s9"]
        },
        "shap_importance_score": { "type": "number", "example": 0.42 }
      }
    },
    "action_required": {
      "type": "string",
      "enum": ["POLL_TELEMETRY", "LOG_ADVISORY", "STAGE_2_INSPECTION", "AUTO_DISPATCH_P1", "EMERGENCY_DE_RATE"]
    },
    "recommended_parts": {
      "type": "array",
      "items": { "type": "string" },
      "example": ["HPC_Blades_Kit_B", "Compressor_Seal_42"]
    },
    "sla_hours": { "type": "integer", "example": 24 }
  }
}
```
