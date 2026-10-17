"""Centralized path and configuration resolution for StrataSense.

Resolves repository root dynamically across local dev, CI/CD runners, and Docker containers.
"""

from pathlib import Path
import os

# Priority:
# 1. Environment variable PROJECT_ROOT if set
# 2. Path relative to this config file: src/config.py -> parent is src -> parent is repo root
PROJECT_ROOT = Path(
    os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent)
).resolve()

DATA_DIR = PROJECT_ROOT / "data" / "c-mapss"
MODELS_DIR = PROJECT_ROOT / "models"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
LOGS_DIR = PROJECT_ROOT / "logs"
