# Low Pressure Turbine (LPT) Tip Clearance Capacitive Probe Specification

## 1. Physics & Sensor Operating Principles
High-temperature capacitive probes mounted on the turbine casing measure the physical gap between rotating blade tips and stationary shroud surfaces during dynamic flight maneuvers.

## 2. Capacitance-to-Distance Transfer Function
The measured capacitance $C$ varies inversely with blade clearance distance $d$:

$$C = \frac{\varepsilon_0 \varepsilon_r A}{d}$$

- **Operational Clearance Window:** $0.5\,\text{mm}$ to $3.5\,\text{mm}$.
- **Thermal Expansion Compensation:** Casing temperature signals ($T_{case}$) dynamically calibrate nominal baseline capacitance.

## 3. Anomaly Criteria
A sudden decrease in tip clearance ($d < 0.3\,\text{mm}$) signals casing ovalization or severe rotor vibration, triggering immediate blade rub warnings.
