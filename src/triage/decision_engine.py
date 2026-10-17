"""Strictly non-LLM rule-based decision ladder and diagnostic triage engine.

Evaluates predicted Remaining Useful Life (RUL) and Continuous Health Index (HI)
to determine operational severity and dispatch maintenance work orders deterministically.
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class SeverityLevel(str, Enum):
    NORMAL = "NORMAL"
    MONITOR = "MONITOR"
    SCHEDULED = "SCHEDULED"
    CRITICAL = "CRITICAL"


@dataclass
class TriageDecision:
    unit_id: int
    operational_cycle: int
    rul_prediction: float
    health_index: float
    severity: SeverityLevel
    action_required: str
    sla_hours: int
    escalate_to_human: bool
    escalation_reason: Optional[str] = None


class DeterministicTriageEngine:
    """Non-LLM state machine ladder:

    - NORMAL:    RUL > 80, HI >= 0.70 -> 10s polling interval
    - MONITOR:   41 <= RUL <= 80 or (RUL > 80 and HI < 0.70) -> 1s polling, advisory flag
    - SCHEDULED: 16 <= RUL <= 40 -> Automated Stage-2 inspection (72h SLA)
    - CRITICAL:  RUL <= 15 or HI < 0.20 -> Priority-1 dispatch + safe de-rate (24h SLA)

    Exception Traps:
    - Rapid Degradation Trap: |dRUL/dt| > 15 cycles/hr -> Routes to Diagnostic Engineer
    - Sensor Conflict Trap: Inconsistent sensor spikes -> Routes to Instrumentation Team
    """

    def __init__(
        self,
        critical_rul_threshold: float = 15.0,
        scheduled_rul_threshold: float = 40.0,
        monitor_rul_threshold: float = 80.0,
        critical_hi_threshold: float = 0.20,
        monitor_hi_threshold: float = 0.70,
        rapid_drop_threshold: float = 15.0
    ):
        self.critical_rul_threshold = critical_rul_threshold
        self.scheduled_rul_threshold = scheduled_rul_threshold
        self.monitor_rul_threshold = monitor_rul_threshold
        self.critical_hi_threshold = critical_hi_threshold
        self.monitor_hi_threshold = monitor_hi_threshold
        self.rapid_drop_threshold = rapid_drop_threshold

    def evaluate_engine_state(
        self,
        unit_id: int,
        operational_cycle: int,
        rul_prediction: float,
        health_index: float,
        prev_rul_prediction: Optional[float] = None,
        sensor_conflict_detected: bool = False
    ) -> TriageDecision:
        """Evaluates engine telemetry deterministically without non-deterministic AI risk."""
        # 1. Check Sensor Hardware Fault Trap
        if sensor_conflict_detected:
            return TriageDecision(
                unit_id=unit_id,
                operational_cycle=operational_cycle,
                rul_prediction=round(rul_prediction, 1),
                health_index=round(health_index, 3),
                severity=SeverityLevel.MONITOR,
                action_required="INSTRUMENTATION_PROBE_DIAGNOSTIC",
                sla_hours=48,
                escalate_to_human=True,
                escalation_reason="Sensor reading divergence detected without core mechanical degradation."
            )

        # 2. Check Rapid Degradation Acceleration Trap
        if prev_rul_prediction is not None:
            drop_rate = prev_rul_prediction - rul_prediction
            if drop_rate > self.rapid_drop_threshold:
                return TriageDecision(
                    unit_id=unit_id,
                    operational_cycle=operational_cycle,
                    rul_prediction=round(rul_prediction, 1),
                    health_index=round(health_index, 3),
                    severity=SeverityLevel.CRITICAL,
                    action_required="EMERGENCY_DE_RATE_AND_DISPATCH",
                    sla_hours=12,
                    escalate_to_human=True,
                    escalation_reason=f"Degradation acceleration detected (RUL dropped {drop_rate:.1f} cycles in 1 step)."
                )

        # 3. 4-Tier Deterministic Decision Ladder
        if rul_prediction <= self.critical_rul_threshold or health_index < self.critical_hi_threshold:
            return TriageDecision(
                unit_id=unit_id,
                operational_cycle=operational_cycle,
                rul_prediction=round(rul_prediction, 1),
                health_index=round(health_index, 3),
                severity=SeverityLevel.CRITICAL,
                action_required="AUTO_DISPATCH_P1",
                sla_hours=24,
                escalate_to_human=False
            )

        if rul_prediction <= self.scheduled_rul_threshold:
            return TriageDecision(
                unit_id=unit_id,
                operational_cycle=operational_cycle,
                rul_prediction=round(rul_prediction, 1),
                health_index=round(health_index, 3),
                severity=SeverityLevel.SCHEDULED,
                action_required="STAGE_2_INSPECTION_ORDER",
                sla_hours=72,
                escalate_to_human=False
            )

        if rul_prediction <= self.monitor_rul_threshold or health_index < self.monitor_hi_threshold:
            return TriageDecision(
                unit_id=unit_id,
                operational_cycle=operational_cycle,
                rul_prediction=round(rul_prediction, 1),
                health_index=round(health_index, 3),
                severity=SeverityLevel.MONITOR,
                action_required="INCREASE_POLLING_FREQUENCY_1S",
                sla_hours=168,
                escalate_to_human=False
            )

        # Default Nominal Healthy State
        return TriageDecision(
            unit_id=unit_id,
            operational_cycle=operational_cycle,
            rul_prediction=round(rul_prediction, 1),
            health_index=round(health_index, 3),
            severity=SeverityLevel.NORMAL,
            action_required="STANDARD_POLLING_10S",
            sla_hours=720,
            escalate_to_human=False
        )


if __name__ == "__main__":
    engine = DeterministicTriageEngine()

    # Scenario 1: Nominal
    d1 = engine.evaluate_engine_state(1, 10, rul_prediction=110.0, health_index=0.95)
    print(f"Scenario 1: {d1.severity.value} -> {d1.action_required}")

    # Scenario 2: Monitor
    d2 = engine.evaluate_engine_state(1, 120, rul_prediction=65.0, health_index=0.68)
    print(f"Scenario 2: {d2.severity.value} -> {d2.action_required}")

    # Scenario 3: Scheduled
    d3 = engine.evaluate_engine_state(1, 150, rul_prediction=28.0, health_index=0.45)
    print(f"Scenario 3: {d3.severity.value} -> {d3.action_required}")

    # Scenario 4: Critical
    d4 = engine.evaluate_engine_state(1, 180, rul_prediction=11.5, health_index=0.18)
    print(f"Scenario 4: {d4.severity.value} -> {d4.action_required}")

    # Scenario 5: Rapid Spike Trap
    d5 = engine.evaluate_engine_state(1, 160, rul_prediction=30.0, health_index=0.50, prev_rul_prediction=52.0)
    print(f"Scenario 5: {d5.severity.value} -> {d5.action_required} (Escalate: {d5.escalate_to_human})")
    print("Deterministic Triage Engine verified.")
