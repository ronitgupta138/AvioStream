package io.aviostream.model;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class EngineVibrationPacketTest {
    @Test
    void testVibrationThresholdCheck() {
        EngineVibrationPacket pkt = new EngineVibrationPacket("ENG-01", 1.8, 2.2, System.currentTimeMillis());
        assertFalse(pkt.isVibrationExceeded(2.5));
        assertTrue(pkt.isVibrationExceeded(2.0));
        assertEquals("ENG-01", pkt.engineId());
    }
}
