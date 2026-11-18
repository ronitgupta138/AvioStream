package io.aviostream.telemetry;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class Arinc717DecoderTest {
    @Test
    void testSyncWordDetection() {
        assertTrue(Arinc717Decoder.isSyncWord(0x247));
        assertTrue(Arinc717Decoder.isSyncWord(0x5B8));
        assertTrue(Arinc717Decoder.isSyncWord(0xA47));
        assertTrue(Arinc717Decoder.isSyncWord(0xDB8));
        assertFalse(Arinc717Decoder.isSyncWord(0x123));
    }

    @Test
    void testSensorScaling() {
        double mid = Arinc717Decoder.decodeSensorValue(2048, 0.0, 100.0);
        assertTrue(mid > 49.0 && mid < 51.0);

        double zero = Arinc717Decoder.decodeSensorValue(0, 10.0, 20.0);
        assertEquals(10.0, zero, 1e-4);

        double max = Arinc717Decoder.decodeSensorValue(4095, 10.0, 20.0);
        assertEquals(20.0, max, 1e-4);
    }
}
