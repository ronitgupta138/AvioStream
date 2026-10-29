# Triage Protocol: Nominal Engine State

## Criteria
- Predicted Remaining Useful Life (RUL) > 50 flight cycles
- Anomaly score < 0.25 (Isolation Forest output)
- All sensor z-scores within $\pm 1.96\sigma$ boundary.

## Dispatch Action
Engine cleared for standard scheduled line maintenance; no operational restrictions.