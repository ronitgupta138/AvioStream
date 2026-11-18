package io.aviostream.buffer;

import java.util.concurrent.atomic.AtomicLong;

/**
 * High-throughput circular telemetry ring buffer.
 */
public class CircularTelemetryBuffer {
    private final double[] buffer;
    private final int capacity;
    private final AtomicLong writeSequence = new AtomicLong(0);

    public CircularTelemetryBuffer(int capacity) {
        if ((capacity & (capacity - 1)) != 0 || capacity <= 0) {
            throw new IllegalArgumentException("Capacity must be a power of two");
        }
        this.capacity = capacity;
        this.buffer = new double[capacity];
    }

    public void put(double value) {
        long seq = writeSequence.getAndIncrement();
        int index = (int) (seq & (capacity - 1));
        buffer[index] = value;
    }

    public double getLatest() {
        long seq = writeSequence.get();
        if (seq == 0) return 0.0;
        int index = (int) ((seq - 1) & (capacity - 1));
        return buffer[index];
    }

    public long getCount() {
        return writeSequence.get();
    }
}
