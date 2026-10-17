package io.aviostream;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * Deterministic Java 21 rule-based triage runner and JSON CMMS work-order emitter.
 */
public class TriageRuleEngine {

    public enum Severity {
        NORMAL, MONITOR, SCHEDULED, CRITICAL
    }

    public record RootCause(String primarySubsystem, List<String> criticalSensors, double shapScore) {}

    public record WorkOrder(
        String workOrderId,
        String siteId,
        String timestampUtc,
        int unitId,
        int operationalCycle,
        String severity,
        double rulPrediction,
        double healthIndex,
        RootCause rootCause,
        String actionRequired,
        List<String> recommendedParts,
        int slaHours,
        String dispatchStatus
    ) {}

    private final ObjectMapper mapper = new ObjectMapper();

    public WorkOrder evaluateAndDispatch(
        int unitId,
        int cycle,
        double rul,
        double healthIndex,
        String failingSubsystem,
        List<String> sensors
    ) {
        Severity severity;
        String action;
        int sla;

        if (rul <= 15.0 || healthIndex < 0.20) {
            severity = Severity.CRITICAL;
            action = "AUTO_DISPATCH_P1";
            sla = 24;
        } else if (rul <= 40.0) {
            severity = Severity.SCHEDULED;
            action = "STAGE_2_INSPECTION_ORDER";
            sla = 72;
        } else if (rul <= 80.0 || healthIndex < 0.70) {
            severity = Severity.MONITOR;
            action = "INCREASE_POLLING_FREQUENCY_1S";
            sla = 168;
        } else {
            severity = Severity.NORMAL;
            action = "STANDARD_POLLING_10S";
            sla = 720;
        }

        List<String> parts = switch (failingSubsystem) {
            case "High_Pressure_Compressor_Static_Pressure" -> List.of("HPC_Blades_Kit_B", "Compressor_Seal_42");
            case "Low_Pressure_Turbine_Exhaust" -> List.of("LPT_Rotor_Blades_Stage2", "Exhaust_Thermocouple_Array");
            default -> List.of("General_Inspection_Seal_Kit", "Fastener_Replacement_Set");
        };

        String woId = "WO-JAVA-" + unitId + "-" + UUID.randomUUID().toString().substring(0, 6).toUpperCase();
        return new WorkOrder(
            woId,
            "AERO-TURBOFAN-FLEET-1",
            Instant.now().toString(),
            unitId,
            cycle,
            severity.name(),
            Math.round(rul * 100.0) / 100.0,
            Math.round(healthIndex * 1000.0) / 1000.0,
            new RootCause(failingSubsystem, sensors, 0.48),
            action,
            parts,
            sla,
            "AUTO_DISPATCHED"
        );
    }

    public String toJson(WorkOrder order) {
        try {
            return mapper.writerWithDefaultPrettyPrinter().writeValueAsString(order);
        } catch (Exception e) {
            throw new RuntimeException("Serialization failure", e);
        }
    }
}
