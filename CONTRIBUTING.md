# Contributing to AvioStream

Thank you for your interest in contributing to **AvioStream**! This document outlines guidelines for issues, branching, and pull requests.

---

## 🛠️ Development Workflow

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ronitgupta138/AvioStream.git
   cd AvioStream
   ```

2. **Branching Conventions**:
   - `feat/<feature-name>`: New features (pipelines, models, telemetry extractors)
   - `fix/<bug-name>`: Bug fixes and edge-case handling
   - `docs/<doc-topic>`: Documentation and architectural schema updates
   - `test/<test-suite>`: Unit, integration, or concurrency test additions
   - `perf/<subsystem>`: Performance benchmarks and latency optimizations

3. **Running Test Suites**:
   Ensure both the Python and Java test suites pass before submitting changes:
   ```bash
   # Run Python pipeline & ML tests (23 tests)
   make test-py

   # Run Java 21 Project Loom streaming engine tests (2 tests)
   make test-java

   # Run all tests in one shot
   make test
   ```

---

## 📋 Architectural Invariants

When contributing to AvioStream, preserve these core engineering invariants:
1. **Deterministic Non-LLM Triage:** Never introduce LLM API calls or non-deterministic generators into the critical triage state machine (`src/triage/`). All work orders must execute within predictable sub-15ms windows.
2. **Zero-GC Loom Ingestion:** Ensure high-throughput stream buffers in `src/java_engine/` maintain low memory allocation profiles to avoid JVM garbage collection pauses under high telemetry load (>48k msgs/s).
3. **Reproducible Numerical Parity:** All trained models exported to ONNX format must maintain floating-point delta parity $< 10^{-5}$ compared to native Python inference.
4. **Clean Commit Messages:** Use Conventional Commits (`feat(...)`, `fix(...)`, `docs(...)`, `test(...)`).

---

## 📬 Submitting a Pull Request

1. Push your branch to your fork.
2. Open a Pull Request against `main`.
3. Verify that all automated GitHub Actions CI/CD workflows pass.
4. Clearly explain the motivation, architectural trade-offs, and verification steps in the PR description.
