# Asymmetric Penalty Scoring Function for Turbofan RUL Estimation

## 1. Aerospace Risk Asymmetry
In aviation maintenance, an early prediction ($d = \hat{y} - y \le 0$) prompts scheduled precautionary overhaul with minimal safety risk. Conversely, a late prediction ($d > 0$) causes unexpected in-flight shutdown or catastrophic engine failure.

## 2. Mathematical Scoring Function
$$S = \sum_{i=1}^{N} s_i \quad \text{where } s_i = \begin{cases} e^{-d_i / 13} - 1, & d_i \le 0 \\ e^{d_i / 10} - 1, & d_i > 0 \end{cases}$$
The exponential penalty factor penalizes positive errors ($d_i > 0$) substantially more aggressively, guiding the custom LightGBM loss gradient toward fail-safe early warnings.
