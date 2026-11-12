# Combustor Liner Low-Cycle Fatigue (LCF) Life Prediction Model

## 1. Damage Mechanism
Repeated thermal expansion and mechanical stress cycles during takeoff throttle advances induce plastic strain deformation in gas turbine combustor liners.

## 2. Manson-Coffin Plastic Strain Relationship
Total fatigue cycle life $N_f$ is estimated via:

$$\frac{\Delta \varepsilon}{2} = \frac{\sigma'_f}{E} (2N_f)^b + \varepsilon'_f (2N_f)^c$$

Where:
- $\Delta \varepsilon$: Total strain amplitude per flight mission.
- $\sigma'_f, b$: Fatigue strength coefficient and exponent.
- $\varepsilon'_f, c$: Fatigue ductility coefficient and exponent.

## 3. Remaining Useful Life (RUL) Deduction
Accumulated fatigue damage index $D = \sum \frac{n_i}{N_{fi}}$ updates the engine RUL prognostic model after every flight mission cycle.
