package io.aviostream.dynamics;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class KalmanFilter1DTest {
    @Test
    void testFilterConvergence() {
        KalmanFilter1D filter = new KalmanFilter1D(500.0, 10.0, 0.01, 2.0);
        double smoothed = 0.0;
        // Feed noisy measurements oscillating around 520.0
        for (int i = 0; i < 50; i++) {
            double noise = (i % 2 == 0) ? 2.5 : -2.5;
            smoothed = filter.update(520.0 + noise);
        }
        assertEquals(520.0, smoothed, 1.5);
    }
}
