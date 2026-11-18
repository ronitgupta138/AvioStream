package io.aviostream.dynamics;

/**
 * 1D Kalman filter for smoothing noisy turbine sensor signals.
 */
public class KalmanFilter1D {
    private double x; // state estimate
    private double p; // error covariance
    private final double q; // process noise
    private final double r; // measurement noise

    public KalmanFilter1D(double initialEstimate, double initialError, double processNoise, double measurementNoise) {
        this.x = initialEstimate;
        this.p = initialError;
        this.q = processNoise;
        this.r = measurementNoise;
    }

    public double update(double measurement) {
        // Prediction
        p = p + q;
        // Measurement update
        double k = p / (p + r); // Kalman gain
        x = x + k * (measurement - x);
        p = (1.0 - k) * p;
        return x;
    }

    public double getState() {
        return x;
    }
}
