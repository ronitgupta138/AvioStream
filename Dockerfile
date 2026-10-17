# =========================================================================
# StrataSense — Production Multi-Stage Container Specification
# Autonomous Aerospace Fleet Predictive Maintenance Engine
# =========================================================================

# -------------------------------------------------------------------------
# STAGE 1: Java 21 Project Loom Engine Builder
# -------------------------------------------------------------------------
FROM maven:3.9-eclipse-temurin-21 AS java-builder

WORKDIR /build/java_engine
COPY src/java_engine/pom.xml .
# Pre-fetch Maven dependencies for caching
RUN mvn dependency:go-offline -B || true

COPY src/java_engine/src ./src
RUN mvn clean package -DskipTests=false -B

# -------------------------------------------------------------------------
# STAGE 2: Python Runtime & Streamlit Fleet Command Platform
# -------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    JAVA_HOME=/usr/lib/jvm/default-jvm \
    PATH="/usr/lib/jvm/default-jvm/bin:${PATH}"

WORKDIR /app

# Install OpenJDK 21 headless runtime and curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python ML and dashboard dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source, models, and benchmark telemetry
COPY data/c-mapss /app/data/c-mapss
COPY models /app/models
COPY src /app/src
COPY tests /app/tests
COPY BENCHMARKS.md PROJECT_PLAN.md README.md /app/

# Copy compiled Java artifacts from Stage 1
COPY --from=java-builder /build/java_engine/target /app/src/java_engine/target

# Healthcheck configuration for Streamlit web server
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

EXPOSE 8501

# Default execution: Launch Streamlit Fleet Command Dashboard
CMD ["streamlit", "run", "src/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
