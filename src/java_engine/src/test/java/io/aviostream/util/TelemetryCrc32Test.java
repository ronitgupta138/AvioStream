package io.aviostream.util;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TelemetryCrc32Test {
    @Test
    void testCrc32Computation() {
        String frame = "ENG_TELEMETRY:altitude=35000;mach=0.82;egt=650";
        long crc = TelemetryCrc32.compute(frame);
        assertTrue(crc > 0);
        assertTrue(TelemetryCrc32.verify(frame, crc));
        assertFalse(TelemetryCrc32.verify(frame + "_corrupted", crc));
    }
}
