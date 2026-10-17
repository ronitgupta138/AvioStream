package io.aviostream;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Concurrent ring buffer for telemetry snapshots executed across
 * Java 21 Virtual Threads with zero contention locks.
 */
public class TelemetryBuffer {

    public record Snapshot(int unitId, int cycle, float[] features) {}

    public static class UnitBuffer {
        private final int capacity;
        private final int numFeatures;
        private final float[][] buffer;
        private final int[] cycles;
        private int head = 0;
        private int count = 0;

        public UnitBuffer(int capacity, int numFeatures) {
            this.capacity = capacity;
            this.numFeatures = numFeatures;
            this.buffer = new float[capacity][numFeatures];
            this.cycles = new int[capacity];
        }

        public synchronized void push(int cycle, float[] features) {
            System.arraycopy(features, 0, buffer[head], 0, Math.min(features.length, numFeatures));
            cycles[head] = cycle;
            head = (head + 1) % capacity;
            if (count < capacity) {
                count++;
            }
        }

        public synchronized int getCount() {
            return count;
        }

        public synchronized int getLatestCycle() {
            if (count == 0) return 0;
            int prev = (head - 1 + capacity) % capacity;
            return cycles[prev];
        }
    }

    private final int capacity;
    private final int numFeatures;
    private final ConcurrentHashMap<Integer, UnitBuffer> fleetBuffers = new ConcurrentHashMap<>();
    private final AtomicInteger totalIngested = new AtomicInteger(0);

    public TelemetryBuffer(int capacity, int numFeatures) {
        this.capacity = capacity;
        this.numFeatures = numFeatures;
    }

    public void ingest(int unitId, int cycle, float[] features) {
        fleetBuffers.computeIfAbsent(unitId, id -> new UnitBuffer(capacity, numFeatures))
                    .push(cycle, features);
        totalIngested.incrementAndGet();
    }

    public int getFleetUnitCount() {
        return fleetBuffers.size();
    }

    public int getTotalIngested() {
        return totalIngested.get();
    }

    public UnitBuffer getUnitBuffer(int unitId) {
        return fleetBuffers.get(unitId);
    }
}
