"""Feature importance and root-cause failure attribution engine.

Analyzes trained LightGBM models to extract global sensor importances
and generates per-engine root-cause failure attribution for work orders.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import json
import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR

# Mapping physical sensor IDs to turbofan mechanical subsystems
SENSOR_SUBSYSTEM_MAP = {
    "s_2": "Low_Pressure_Compressor_Thermal",
    "s_3": "High_Pressure_Compressor_Inlet",
    "s_4": "Low_Pressure_Turbine_Exhaust",
    "s_7": "High_Pressure_Compressor_Discharge",
    "s_8": "Fan_Spool_Shaft",
    "s_9": "Core_High_Pressure_Spool",
    "s_11": "High_Pressure_Compressor_Static_Pressure",
    "s_12": "Fuel_Flow_Combustor_Ratio",
    "s_13": "Corrected_Fan_Speed_Dynamics",
    "s_14": "Corrected_Core_Speed_Dynamics",
    "s_15": "Bypass_Duct_Aerodynamics",
    "s_17": "Turbine_Bleed_Enthalpy",
    "s_20": "High_Pressure_Turbine_Coolant_Bleed",
    "s_21": "Low_Pressure_Turbine_Coolant_Bleed"
}


class ModelExplainabilityEngine:
    """Extracts global feature weights and computes localized root cause."""

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir
        self.model = joblib.load(models_dir / "rul_lightgbm_best.joblib")
        with open(models_dir / "feature_names.json") as f:
            self.feature_names = json.load(f)

        # Gain-based importance from LightGBM booster
        importances = self.model.booster_.feature_importance(importance_type="gain")
        total_gain = float(np.sum(importances))
        self.importance_series = pd.Series(
            importances / (total_gain + 1e-9),
            index=self.feature_names
        ).sort_values(ascending=False)

    def get_top_global_features(self, top_n: int = 10) -> List[Tuple[str, float]]:
        return list(self.importance_series.head(top_n).items())

    def attribute_root_cause(
        self,
        engine_snapshot: pd.Series,
        top_k: int = 3
    ) -> Dict:
        """Determines top degraded sensors and failure subsystem for an engine cycle."""
        # Find which sensors are deviating most from baseline weighted by importance
        sensor_weights: Dict[str, float] = {}

        for feat, imp in self.importance_series.items():
            base_sensor = feat.split("_")[0] + "_" + feat.split("_")[1] if feat.startswith("s_") else None
            if base_sensor in SENSOR_SUBSYSTEM_MAP:
                val = abs(float(engine_snapshot.get(feat, 0.0)))
                sensor_weights[base_sensor] = sensor_weights.get(base_sensor, 0.0) + (val * imp)

        sorted_sensors = sorted(sensor_weights.items(), key=lambda x: x[1], reverse=True)
        top_sensor, top_weight = sorted_sensors[0] if sorted_sensors else ("s_11", 1.0)
        top_sensors_list = [s[0] for s in sorted_sensors[:top_k]]

        primary_subsystem = SENSOR_SUBSYSTEM_MAP.get(top_sensor, "General_Turbomachinery_Degradation")

        return {
            "primary_subsystem": primary_subsystem,
            "critical_sensors": top_sensors_list,
            "top_degraded_sensor": top_sensor,
            "attribution_score": round(float(top_weight), 4),
            "subsystem_description": f"Abnormal thermodynamic wear in {primary_subsystem} driven by {top_sensor}"
        }


if __name__ == "__main__":
    engine = ModelExplainabilityEngine()
    print("=== Top Global Degradation Drivers (by Split Gain) ===")
    for feat, gain in engine.get_top_global_features(8):
        print(f"  {feat:<25} {gain * 100:.2f}%")

    # Sample root-cause attribution
    sample_snapshot = pd.Series({f: 1.5 for f in engine.feature_names})
    attribution = engine.attribute_root_cause(sample_snapshot)
    print("\n=== Sample Root-Cause Attribution ===")
    print(json.dumps(attribution, indent=2))
