package io.stratasense;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class EngineTest {

    @Test
    public void testTelemetryBufferVirtualThreadsConcurrency() throws InterruptedException {
        TelemetryBuffer buffer = new TelemetryBuffer(30, 14);

        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            for (int unit = 1; unit <= 50; unit++) {
                final int unitId = unit;
                executor.submit(() -> {
                    for (int cycle = 1; cycle <= 40; cycle++) {
                        float[] sensors = new float[14];
                        sensors[0] = (float) cycle;
                        buffer.ingest(unitId, cycle, sensors);
                    }
                });
            }
            executor.shutdown();
            assertTrue(executor.awaitTermination(5, TimeUnit.SECONDS));
        }

        assertEquals(50, buffer.getFleetUnitCount());
        assertEquals(2000, buffer.getTotalIngested());
        assertEquals(40, buffer.getUnitBuffer(1).getLatestCycle());
    }

    @Test
    public void testTriageRuleEngineStateTransitions() {
        TriageRuleEngine engine = new TriageRuleEngine();

        // 1. Critical state
        TriageRuleEngine.WorkOrder wo1 = engine.evaluateAndDispatch(
            42, 190, 12.5, 0.18, "High_Pressure_Compressor_Static_Pressure", List.of("s_11")
        );
        assertEquals("CRITICAL", wo1.severity());
        assertEquals("AUTO_DISPATCH_P1", wo1.actionRequired());
        assertEquals(24, wo1.slaHours());
        assertTrue(wo1.recommendedParts().contains("HPC_Blades_Kit_B"));

        // 2. Normal state
        TriageRuleEngine.WorkOrder wo2 = engine.evaluateAndDispatch(
            1, 20, 110.0, 0.95, "None", List.of()
        );
        assertEquals("NORMAL", wo2.severity());
        assertEquals("STANDARD_POLLING_10S", wo2.actionRequired());

        // Check JSON generation
        String json = engine.toJson(wo1);
        assertNotNull(json);
        assertTrue(json.contains("WO-JAVA-42"));
        assertTrue(json.contains("AUTO_DISPATCHED"));
    }
}
