# NASA C-MAPSS FD004 Multi-Fault Degradation Model

## 1. Flight Envelope Complexity
The FD004 subset of the NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) represents the most complex turbofan dataset:
- **Operating Conditions:** 6 distinct flight modes (altitude 0–42,000 ft, Mach number 0.0–0.84, throttle resolver angle 20–100°).
- **Fault Modes:** Dual simultaneous failure mechanisms — High-Pressure Compressor (HPC) degradation and Fan blade erosion.

## 2. Feature Engineering Pipeline
- Flight condition clustering via k-means into discrete operating regimes.
- Operational regime normalization (Z-score scaling per cluster) across 21 sensor channels to isolate true thermo-mechanical wear from altitude-induced pressure drops.
