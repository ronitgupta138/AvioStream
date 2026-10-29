# Deployment Specification: Docker & Kubernetes

## Container Architecture
- Multi-stage minimal container based on `eclipse-temurin:21-jre-alpine`.
- Non-root user execution (`UID 10001`).
- Read-only root filesystem with ephemeral memory-backed `/tmp`.
- Health check endpoint probing `/health/liveness` and `/health/readiness`.