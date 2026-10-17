"""Model registry and experiment tracking metadata manager.

Tracks model versions, hyperparameters, and manages promotion gates
(Development -> Staging -> Canary Validation -> Production -> Archived).
"""

from pathlib import Path
from typing import Dict, List, Optional
import json
import time

from src.config import MODELS_DIR

REGISTRY_FILE = MODELS_DIR / "model_registry.json"


class ModelRegistryManager:
    """Simulates production MLOps Model Registry governance."""

    STAGES = ["Development", "Staging", "Canary", "Production", "Archived"]

    def __init__(self, registry_path: Path = REGISTRY_FILE):
        self.registry_path = registry_path
        self.registry: Dict = self._load_registry()

    def _load_registry(self) -> Dict:
        if self.registry_path.exists():
            with open(self.registry_path, "r") as f:
                return json.load(f)
        return {
            "models": {},
            "production_version": None,
            "audit_trail": []
        }

    def _save_registry(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=2)

    def register_model_version(
        self,
        version: str,
        model_type: str,
        hyperparameters: dict,
        metrics: dict,
        artifact_path: str
    ) -> dict:
        """Registers a newly trained model version into Staging."""
        record = {
            "version": version,
            "stage": "Staging",
            "model_type": model_type,
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hyperparameters": hyperparameters,
            "metrics": metrics,
            "artifact_path": artifact_path
        }
        self.registry["models"][version] = record
        self._log_audit(version, "REGISTERED", "Registered to Staging")
        self._save_registry()
        return record

    def promote_to_production(self, version: str, validation_rmse: float) -> bool:
        """Promotes a candidate version if it meets the strict quality gate."""
        if version not in self.registry["models"]:
            raise KeyError(f"Version {version} not found in registry")

        # Strict Quality Gate: RMSE must be < 15.0 cycles
        if validation_rmse >= 15.0:
            self._log_audit(version, "PROMOTION_REJECTED", f"RMSE {validation_rmse:.2f} failed threshold (< 15.0)")
            self._save_registry()
            return False

        current_prod = self.registry["production_version"]
        if current_prod and current_prod in self.registry["models"]:
            self.registry["models"][current_prod]["stage"] = "Archived"
            self._log_audit(current_prod, "DEMOTED", "Demoted to Archived")

        self.registry["models"][version]["stage"] = "Production"
        self.registry["production_version"] = version
        self._log_audit(version, "PROMOTED", f"Promoted to Production with RMSE {validation_rmse:.2f}")
        self._save_registry()
        return True

    def _log_audit(self, version: str, action: str, details: str):
        self.registry["audit_trail"].append({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "version": version,
            "action": action,
            "details": details
        })

    def get_production_model(self) -> Optional[dict]:
        prod_v = self.registry.get("production_version")
        if prod_v:
            return self.registry["models"].get(prod_v)
        return None


if __name__ == "__main__":
    manager = ModelRegistryManager()

    # Register initial production model
    with open(MODELS_DIR / "test_benchmark_metrics.json") as f:
        bench_metrics = json.load(f)

    reg = manager.register_model_version(
        version="v1.0.0",
        model_type="LightGBM_C_MAPSS_Regressor",
        hyperparameters={"n_estimators": 450, "learning_rate": 0.025, "cap": 125},
        metrics=bench_metrics,
        artifact_path="models/aviostream_rul.onnx"
    )
    promoted = manager.promote_to_production("v1.0.0", validation_rmse=bench_metrics["rmse"])

    print(f"Registered Version: {reg['version']}")
    print(f"Promotion to Production: {'SUCCESS' if promoted else 'FAILED'}")
    print(f"Active Production Model: {manager.get_production_model()['version']}")
    print(f"Audit events logged: {len(manager.registry['audit_trail'])}")
