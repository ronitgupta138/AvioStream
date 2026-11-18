package io.aviostream.buffer;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class CircularTelemetryBufferTest {
    @Test
    void testBufferWrapAround() {
        CircularTelemetryBuffer buf = new CircularTelemetryBuffer(4);
        buf.put(10.0);
        buf.put(20.0);
        buf.put(30.0);
        buf.put(40.0);
        assertEquals(40.0, buf.getLatest());

        // Overwrite oldest slot
        buf.put(50.0);
        assertEquals(50.0, buf.getLatest());
        assertEquals(5, buf.getCount());
    }

    @Test
    void testInvalidCapacityThrows() {
        assertThrows(IllegalArgumentException.class, () -> new CircularTelemetryBuffer(5));
    }
}
