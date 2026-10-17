"""Comprehensive unit test suite for StrataSense Phase 4: Deterministic Triage Agent."""

import json
import pytest

from src.triage.decision_engine import DeterministicTriageEngine, SeverityLevel
from src.triage.work_order_generator import WorkOrderPayloadGenerator


class TestDeterministicTriageEngine:
    def setup_method(self):
        self.engine = DeterministicTriageEngine()

    def test_nominal_state(self):
        d = self.engine.evaluate_engine_state(1, 10, rul_prediction=110.0, health_index=0.92)
        assert d.severity == SeverityLevel.NORMAL
        assert d.action_required == "STANDARD_POLLING_10S"
        assert not d.escalate_to_human

    def test_monitor_state(self):
        d = self.engine.evaluate_engine_state(1, 120, rul_prediction=65.0, health_index=0.68)
        assert d.severity == SeverityLevel.MONITOR
        assert d.action_required == "INCREASE_POLLING_FREQUENCY_1S"
        assert not d.escalate_to_human

    def test_scheduled_state(self):
        d = self.engine.evaluate_engine_state(1, 150, rul_prediction=30.0, health_index=0.50)
        assert d.severity == SeverityLevel.SCHEDULED
        assert d.action_required == "STAGE_2_INSPECTION_ORDER"
        assert d.sla_hours == 72
        assert not d.escalate_to_human

    def test_critical_state(self):
        d = self.engine.evaluate_engine_state(1, 185, rul_prediction=12.0, health_index=0.15)
        assert d.severity == SeverityLevel.CRITICAL
        assert d.action_required == "AUTO_DISPATCH_P1"
        assert d.sla_hours == 24
        assert not d.escalate_to_human

    def test_sensor_conflict_trap(self):
        d = self.engine.evaluate_engine_state(
            unit_id=1,
            operational_cycle=100,
            rul_prediction=90.0,
            health_index=0.85,
            sensor_conflict_detected=True
        )
        assert d.escalate_to_human
        assert "divergence" in d.escalation_reason.lower()
        assert d.action_required == "INSTRUMENTATION_PROBE_DIAGNOSTIC"

    def test_rapid_degradation_trap(self):
        d = self.engine.evaluate_engine_state(
            unit_id=1,
            operational_cycle=140,
            rul_prediction=35.0,
            health_index=0.55,
            prev_rul_prediction=55.0  # Dropped 20 cycles in 1 step!
        )
        assert d.severity == SeverityLevel.CRITICAL
        assert d.escalate_to_human
        assert d.action_required == "EMERGENCY_DE_RATE_AND_DISPATCH"


class TestWorkOrderPayloadGenerator:
    def setup_method(self):
        self.generator = WorkOrderPayloadGenerator()

    def test_payload_structure_and_completeness(self):
        wo = self.generator.generate_work_order(
            unit_id=42,
            operational_cycle=192,
            severity="CRITICAL",
            rul_prediction=14.2,
            health_index=0.18,
            action_required="AUTO_DISPATCH_P1",
            sla_hours=24,
            root_cause_info={
                "primary_subsystem": "High_Pressure_Compressor_Static_Pressure",
                "critical_sensors": ["s_11", "s_3"],
                "attribution_score": 0.45
            }
        )

        assert wo["unit_id"] == 42
        assert wo["operational_cycle"] == 192
        assert wo["severity"] == "CRITICAL"
        assert wo["sla_hours"] == 24
        assert "HPC_Blades_Kit_B" in wo["recommended_parts"]
        assert wo["root_cause"]["primary_subsystem"] == "High_Pressure_Compressor_Static_Pressure"
        assert wo["dispatch_status"] == "AUTO_DISPATCHED"
