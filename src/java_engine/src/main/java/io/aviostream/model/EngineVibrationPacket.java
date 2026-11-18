package io.aviostream.model;

import java.io.Serializable;

/**
 * Immutable aircraft turbine vibration telemetry packet.
 */
public record EngineVibrationPacket(
    String engineId,
    double n1FanVibrationMmPerSec,
    double n2CoreVibrationMmPerSec,
    long timestampMs
) implements Serializable {
    public boolean isVibrationExceeded(double maxAllowed) {
        return n1FanVibrationMmPerSec > maxAllowed || n2CoreVibrationMmPerSec > maxAllowed;
    }
}
