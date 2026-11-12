# Engine Trend Monitoring (ETM) Alert Hysteresis & Debouncing Logic

## 1. Motivation
Transient pilot throttle changes, aerodynamic turbulence, and climb phase transitions produce momentary sensor spikes that must not trigger false engine maintenance alerts.

## 2. Hysteresis Band Definition
- **Upper Activation Threshold ($H_{\text{on}}$):** EGT margin degradation exceeds $+12^\circ\text{C}$ over smoothed baseline.
- **Lower Deactivation Threshold ($H_{\text{off}}$):** EGT margin returns below $+6^\circ\text{C}$.

## 3. Multi-Cycle Persistence Counter
An alert is dispatched to airline dispatch engineering only after the signal condition persists for $\ge 5$ consecutive steady-state cruise monitoring cycles ($t_{\text{persist}} \ge 15\,\text{minutes}$).
