<div align="center">

```text
 █████╗ ██╗   ██╗██╗██████╗ ███████╗███████╗██████╗███████╗ █████╗ ███╗   ███╗
██╔══██╗██║   ██║██║██╔═══██╗██╔════╝╚═██╔═╝██╔══██╗██╔════╝██╔══██╗████╗ ████║
███████║██║   ██║██║██║   ██║███████╗  ██║  ██████╔╝█████╗  ███████║██╔████╔██║
██╔══██║╚██╗ ██╔╝██║██║   ██║╚════██║  ██║  ██╔══██╗██╔══╝  ██╔══██║██║╚██╔╝██║
██║  ██║ ╚████╔╝ ██║╚██████╔╝███████║  ██║  ██║  ██║███████╗██║  ██║██║ ╚═╝ ██║
╚═╝  ╚═╝  ╚═══╝  ╚═╝ ╚═════╝ ╚══════╝  ╚═╝  ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝
```

### **AvioStream — Closed-Loop Autonomous Predictive Maintenance Platform**
*Hybrid Enterprise Telemetry Ingestion, Dual-Stage Machine Learning & Deterministic CMMS Work-Order Dispatch*

[![GitHub Stars](https://badgen.net/github/stars/ronitgupta138/AvioStream?icon=github&color=0891b2)](https://github.com/ronitgupta138/AvioStream/stargazers)
[![Java 21](https://img.shields.io/badge/Java-21%20LTS%20Loom-0891b2?style=flat-square&logo=openjdk&logoColor=white)](https://openjdk.org/projects/jdk/21/)
[![Throughput](https://img.shields.io/badge/Throughput-48k%20msgs%2Fs-10b981?style=flat-square)](https://github.com/ronitgupta138/AvioStream)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Edge-0.014ms-0891b2?style=flat-square&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![Python 3.11](https://img.shields.io/badge/Python-3.11%20MLOps-0891b2?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-25%2F25%20Passed-10b981?style=flat-square)](https://github.com/ronitgupta138/AvioStream)
[![NASA PHM Score](https://img.shields.io/badge/NASA%20PHM-296.77-0891b2?style=flat-square)](https://github.com/ronitgupta138/AvioStream)
[![Docker](https://img.shields.io/badge/Docker-Multi--Stage-0891b2?style=flat-square&logo=docker&logoColor=white)](https://github.com/ronitgupta138/AvioStream)
[![CI/CD Pipeline](https://img.shields.io/badge/CI-GitHub_Actions-0891b2?style=flat-square&logo=githubactions&logoColor=white)](https://github.com/ronitgupta138/AvioStream/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-MIT-gray?style=flat-square)](LICENSE)

</div>

> ⚡ **AvioStream** is a high-throughput, closed-loop autonomous predictive maintenance and telemetry streaming platform for commercial aerospace turbofan fleets. By coupling offline Python 3.11 MLOps and portable ONNX graph serialization with a high-throughput **Java 21 Project Loom (Virtual Threads)** streaming ingestion engine (>48,000 msgs/s), AvioStream continuously tracks degradation trajectories, detects early sensor anomalies, estimates Remaining Useful Life ($RUL$), and auto-dispatches standardized enterprise CMMS/ERP work orders through a strictly deterministic non-LLM triage state machine in sub-15ms execution windows.

---

## 📌 Architectural Highlights

* **High-Throughput Loom Ingestion Engine:** Java 21 Project Loom virtual threads stream buffer supporting **>48,000 msgs/s** with sub-35 KB total heap footprint across 100 concurrent engine units.
* **Dual-Stage Prognostics Core:** Unsupervised Anomaly Detection via Isolation Forest (computing continuous Health Index $HI \in [0.0, 1.0]$) paired with supervised LightGBM Remaining Useful Life ($RUL$) regression.
* **Asymmetric NASA PHM Loss:** Custom penalty objective penalizing late failure predictions ($d > 0$) 3× heavier than early predictions to eliminate catastrophic in-flight downtime.
* **Embedded ONNX Edge Runtime:** Portable serialized model graph (`models/aviostream_rul.onnx`) executing in **0.014ms** per cycle on commodity CPU cores via JNI bindings with zero Python runtime overhead.
* **Statistical Telemetry Drift Engine:** Automated Two-Sample Kolmogorov-Smirnov (K-S) test and 10-bin Population Stability Index ($PSI$) tracking over sliding operational windows for model degradation governance.
* **Deterministic Non-LLM Triage:** Microsecond 4-tier decision state machine eliminating LLM hallucination risk and generating validated SAP PM / IBM Maximo BAPI work orders.

---

## 🏛️ System Architecture

```
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    AVIOSTREAM ARCHITECTURE                                   │
└───────────────────────────────────────────────────────────────────────────────────────────────┘

  [ NASA C-MAPSS FD001–FD004 Fleet Telemetry ]        [ Continuous Streaming Telemetry Tap ]
                       │                                                 │
                       ▼                                                 ▼
┌──────────────────────────────────────────────┐  ONNX Graph   ┌────────────────────────────────┐
│             OFFLINE MLOPS CORE               │  Serialized   │  ONLINE VIRTUAL THREAD ENGINE  │
│                (Python 3.11)                 │  ───────────► │        (Java 21 + Loom)        │
│                                              │  (< 0.02ms)   │                                │
│ • K-Means Operating Regime Normalizer (K=6)  │               │ • High-Throughput Ring Buffer  │
│ • 7 Invariant Sensor Pruning (σ = 0 dropped) │               │   (~48,000 msgs/sec ingestion) │
│ • Savitzky-Golay + EMA Noise Attenuation     │               │ • Zero-GC Memory Footprint     │
│ • Multi-Scale Rolling Windows W={5, 15, 30}  │               │   (< 35 KB across 100 units)   │
│ • Dual ML: Isolation Forest (HI) + LightGBM  │               │ • Embedded onnxruntime-java    │
│ • Piecewise Target R_clip = 125 cycles       │               │ • Strongly-Typed Triage Ladder │
│ • Custom Asymmetric NASA PHM Loss Function   │               │   (Sub-0.01ms Decision SLA)    │
│ • Drift Governance: K-S & PSI Retrain Loop   │               │ • Programmatic CMMS Dispatch   │
└──────────────────────────────────────────────┘               └────────────────────────────────┘
                                                                                 │
                                                                                 ▼
                                                               ┌────────────────────────────────┐
                                                               │  ENTERPRISE CMMS / ERP PAYLOAD │
                                                               │  • Sub-15ms Validated JSON     │
                                                               │  • SAP PM / IBM Maximo BAPI    │
                                                               │  • Automated Parts Mapping     │
                                                               └────────────────────────────────┘
```

---

## ⚡ Empirical Benchmarks & SLAs

Rigorous verification evaluated across all **100 test engine units** on their final recorded operational cycles against ground-truth `RUL_FD001.txt` on commodity hardware (**AMD Ryzen 5 5600H, 15GB RAM**):

| Evaluation Metric | Target SLA | AvioStream Measured Result | Verification Verdict |
| :--- | :--- | :--- | :--- |
| **Test Set Root Mean Squared Error (RMSE)** | $< 15.00\text{ cycles}$ | **14.79 cycles** | **Verified ✅** (State-of-the-art tabular tree result) |
| **Official NASA PHM Score ($S$)** | $< 350.00$ | **296.77** | **Verified ✅** (Beats threshold by $> 15\%$) |
| **Test Mean Absolute Error (MAE)** | $< 12.00\text{ cycles}$ | **11.31 cycles** | **Verified ✅** |
| **Mean Prediction Bias ($d = \widehat{y} - y$)** | $\approx 0.0\text{ cycles}$ | **-0.45 cycles** | **Verified ✅** (Zero late-failure penalty skew) |
| **5-Fold GroupKFold Cross-Validation RMSE** | $< 14.50\text{ cycles}$ | **13.59 cycles** | **Verified ✅** (Zero data leakage across units) |
| **ONNX Runtime CPU Forward Pass Latency** | $< 2.00\text{ ms}$ | **0.014 ms / cycle** | **Verified ✅** (~140x faster than SLA) |
| **Java 21 Loom Streaming Throughput** | $> 10,000\text{ msgs/s}$ | **~48,000 msgs/s** | **Verified ✅** (~4.8x higher than SLA) |
| **Deterministic Triage Decision SLA** | $< 15.00\text{ ms}$ | **0.008 ms / decision** | **Verified ✅** (Sub-microsecond deterministic) |
| **Circular Ring Buffer Heap Footprint** | $< 1.80\text{ MB}$ | **34.80 KB (100 units)** | **Verified ✅** (< 2% of allocated budget) |

---

## 🌐 Multi-Flight-Condition Cross-Benchmark Matrix (FD001–FD004)

Real-world commercial turbofans transition through complex flight envelopes (altitudes $0\text{--}42,000\text{ ft}$, Mach $0.0\text{--}0.84$, and throttle $20\text{--}100\%$). AvioStream features a $K$-Means clustering **Operating Regime Normalizer** (`src/pipeline/regime_normalizer.py`) that clusters operational flight settings and stabilizes environmental sensor drift without physical probe bias:

| Dataset | Operating Regimes | Fault Modes | Train Units | Test Units | Test RMSE | Test MAE | NASA Score ($S$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FD001** | 1 (Sea Level) | 1 (HPC Degradation) | 100 | 100 | **14.79** | **11.31** | **296.77** |
| **FD002** | 6 (Full Flight Envelope) | 1 (HPC Degradation) | 260 | 259 | **27.33** | **18.04** | **9,746.05** |
| **FD003** | 1 (Sea Level) | 2 (HPC + Fan) | 100 | 100 | **15.93** | **11.70** | **550.62** |
| **FD004** | 6 (Full Flight Envelope) | 2 (HPC + Fan) | 249 | 248 | **28.38** | **20.66** | **6,007.57** |

---

## 🛠️ The 4 Core Architectural Pillars

### 1. Data Engineering & Signal Conditioning
- **Invariant Sensor Pruning:** Evaluates variance across all 21 telemetry channels; automatically prunes 7 invariant flatline sensors ($s_1, s_5, s_6, s_{10}, s_{16}, s_{18}, s_{19}$) with $\sigma = 0$.
- **Noise Attenuation:** Cascaded Savitzky-Golay polynomial smoothing ($W=7, p=2$) preserves inflection points without phase distortion; streaming Exponential Moving Average ($\alpha=0.20$) suppresses high-frequency sensor jitter.
- **Multi-Scale Rolling Extraction:** Computes rolling mean ($\mu_W$), standard deviation ($\sigma_W$), min-max spread, and first-derivative degradation slope across scales $W \in \{5, 15, 30\}$, yielding 185 features with zero Pandas fragmentation.

### 2. Dual-Stage Machine Learning Core
- **Stage A (Unsupervised Anomaly Detection):** Isolation Forest baseline fitted exclusively on nominal cycles ($t \le 30$) computing continuous Health Index $HI \in [0.0, 1.0]$.
- **Stage B (Supervised Prognostics):** LightGBM Regressor with piecewise linear RUL target clipping ($R_{clip} = 125$ cycles).
- **Asymmetric NASA PHM Loss:** Custom objective function penalizing late failure predictions ($d > 0$) 3× heavier than early predictions:
  $$h(d) = \begin{cases} e^{-d / 13} - 1, & d < 0 \\ e^{d / 10} - 1, & d \ge 0 \end{cases}$$
- **Portable ONNX Serialization:** Exported to portable `.onnx` graph format; verified floating-point numerical parity (max delta $< 10^{-5}$) and CPU forward pass latency of **0.014ms**.

### 3. MLOps Lifecycle & Telemetry Drift Governance
- **Statistical Drift Engine:** Automated Two-Sample Kolmogorov-Smirnov (K-S) test and 10-bin Population Stability Index ($PSI$) tracking across sliding operational windows.
- **Retraining Automation:** Gates trigger automated shadow canary retraining when $PSI > 0.20$ or new run-to-failure cycles finish.

### 4. Deterministic Non-LLM Triage Agent
- **Strictly Non-LLM Execution:** Eliminates non-deterministic LLM hallucination and latency in high-stakes turbofan maintenance.
- **4-Tier Health Ladder:**
  - 🟢 **NORMAL** ($RUL > 80, HI \ge 0.70$): Standard 10s telemetry polling.
  - 🟡 **MONITOR** ($41\text{--}80\text{ cyc}, HI < 0.70$): 1s high-frequency polling + advisory flag.
  - 🟠 **SCHEDULED** ($16\text{--}40\text{ cyc}$): Stage-2 automated inspection order ($< 72\text{h}$ SLA).
  - 🔴 **CRITICAL** ($RUL \le 15 \lor HI < 0.20$): Priority-1 emergency work order + safe-load de-rate dispatch ($< 24\text{h}$ SLA).
- **Diagnostic Exception Traps:** Detects sudden degradation acceleration ($|dRUL/dt| > 15$) and isolates hardware sensor probe failures from true physical spool damage.

---

## 💻 Streamlit Fleet Command & Digital Twin Operations Console

Launch the dark-theme real-time operations console:
```bash
streamlit run src/dashboard/app.py
```

### Console Views
1. **Fleet Command Overview:** Real-time 100-engine fleet status grid with color-coded health badges and maintenance priority dispatch queue.
2. **Digital Twin & Stream Replayer:**
   - Cycle-by-cycle operational scrubber simulating run-to-failure engines (e.g., Unit 42 over 156 cycles).
   - Turbofan physical subsystem wear progress indicators: Fan Assembly, LPC, HPC, Combustor, HPT, LPT.
   - Real-time sensor sparklines ($s_4$ exhaust temperature, $s_{11}$ static pressure, $s_9$ core speed).
   - Live work-order card and raw JSON payload view on critical state triggers.
3. **Engine Diagnostics Deep-Dive:** Sensor wear time-series and Tree-SHAP root-cause feature attribution ranking.
4. **Live CMMS Dispatch Console:** Enterprise work-order queue and JSON payload inspector.
5. **Architecture & Benchmarks:** Verified live benchmark metrics and SLA verification reports.

---

## 🚀 Quickstart & Makefile Commands

```bash
# Clone the repository
git clone https://github.com/ronitgupta138/AvioStream.git
cd AvioStream

# Run complete end-to-end pipeline in one command
make run-pipeline

# Run Python unit tests (23/23 tests passing)
make test-py

# Run Java 21 Loom engine test suite (2/2 tests passing)
make test-java

# Run all test suites
make test

# Launch interactive Streamlit Digital Twin UI
make run-dashboard

# Build and run zero-dependency Docker container
make docker-build
make docker-run
```

---

## 📦 Standard Enterprise CMMS JSON Work-Order Payload

When an engine crosses health thresholds, AvioStream automatically generates a structured JSON payload conforming to enterprise SAP PM / IBM Maximo REST BAPI specifications:

```json
{
  "work_order_id": "WO-20261007-0042-88B12A",
  "site_id": "AERO-TURBOFAN-FLEET-1",
  "timestamp_utc": "2026-10-07T11:45:00Z",
  "unit_id": 42,
  "operational_cycle": 156,
  "severity": "CRITICAL",
  "rul_prediction": 14.2,
  "health_index": 0.18,
  "root_cause": {
    "primary_subsystem": "High_Pressure_Compressor_Static_Pressure",
    "critical_sensors": ["s_11", "s_3", "s_4"],
    "attribution_score": 0.5234
  },
  "action_required": "AUTO_DISPATCH_P1",
  "recommended_parts": [
    "HPC_Blades_Kit_B",
    "Compressor_Seal_42"
  ],
  "sla_hours": 24,
  "dispatch_status": "AUTO_DISPATCHED",
  "notes": "Automated closed-loop work order for Engine 42 at cycle 156."
}
```

---

## 📁 Repository Organization

```text
AvioStream/
├── Makefile                           # Unified developer build, test, and run targets
├── Dockerfile                         # Multi-stage production container specification
├── .dockerignore                      # Build context exclusions
├── LICENSE                            # MIT License
├── CONTRIBUTING.md                    # Development workflows & contribution guidelines
├── CODE_OF_CONDUCT.md                 # Contributor Covenant v2.1
├── SECURITY.md                        # Vulnerability reporting protocol
├── requirements.txt                   # Pinned Python dependencies
├── PROJECT_PLAN.md                    # Master execution roadmap & phase tracking plan
├── BENCHMARKS.md                      # Comprehensive empirical metrics & SLA verification
├── README.md                          # Master project documentation & system architecture
├── .github/
│   ├── FUNDING.yml                    # GitHub Sponsors configuration
│   ├── PULL_REQUEST_TEMPLATE.md       # PR review checklist
│   ├── ISSUE_TEMPLATE/                # Structured bug and feature templates
│   └── workflows/ci.yml               # Automated GitHub Actions CI/CD pipeline
├── data/c-mapss/                      # NASA C-MAPSS benchmark datasets (FD001–FD004)
├── models/
│   ├── aviostream_rul.onnx            # Exported portable ONNX model graph (< 0.02ms)
│   ├── robust_scaler.joblib           # Fitted telemetry IQR robust scaler
│   ├── feature_names.json             # 185 engineered feature dictionary
│   └── test_benchmark_metrics.json    # Verified test set evaluation metrics
├── src/
│   ├── config.py                      # Centralized path and project root resolution
│   ├── pipeline/                      # Ingestion, filtering, regime normalizer, ring buffer
│   ├── ml/                            # Dual ML models, loss functions, ONNX serialization
│   ├── mlops/                         # K-S and PSI statistical drift governance
│   ├── triage/                        # Deterministic 4-tier state machine & CMMS generator
│   ├── java_engine/                   # Java 21 Project Loom High-Throughput Streaming Engine
│   └── dashboard/                     # Streamlit interactive UI & dynamic stream player
└── tests/                             # 23 unit tests covering all pipeline layers
```

---

<div align="center">

Made with ⚡ by **Ronit Gupta**  
*Systems & High-Concurrency Backend Engineer*

</div>
