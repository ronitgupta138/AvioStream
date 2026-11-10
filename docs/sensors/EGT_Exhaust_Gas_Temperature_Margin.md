# Exhaust Gas Temperature (EGT) Margin Degradation Telemetry Specification

## 1. Sensor Overview
- **Sensor Channel:** `T50` (Low Pressure Turbine Outlet Total Temperature).
- **Physical Significance:** As turbine seals erode and compressor efficiency decays, more fuel must be injected to maintain target thrust, driving EGT towards certified redline limits.
- **EGT Margin:** Defined as $\text{EGT}_{\text{redline}} - \text{EGT}_{\text{takeoff}}$.

## 2. Anomaly Drift Thresholds
- Baseline healthy takeoff margin: $\ge 45^{\circ}\text{C}$.
- Warning threshold: $\le 18^{\circ}\text{C}$ indicates imminent turbine distress and required combustor borescope inspection.
