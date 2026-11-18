package io.aviostream.alert;

/**
 * Remaining Useful Life (RUL) hysteresis alert state machine.
 */
public class RulHysteresisAlert {
    public enum AlertSeverity {
        NORMAL,
        CAUTION,
        CRITICAL
    }

    private final double criticalThreshold;
    private final double cautionThreshold;
    private final double resetHysteresis;

    private AlertSeverity currentSeverity = AlertSeverity.NORMAL;

    public RulHysteresisAlert(double criticalThreshold, double cautionThreshold, double resetHysteresis) {
        this.criticalThreshold = criticalThreshold;
        this.cautionThreshold = cautionThreshold;
        this.resetHysteresis = resetHysteresis;
    }

    public AlertSeverity evaluate(double predictedRulCycles) {
        if (predictedRulCycles <= criticalThreshold) {
            currentSeverity = AlertSeverity.CRITICAL;
        } else if (currentSeverity == AlertSeverity.CRITICAL && predictedRulCycles < criticalThreshold + resetHysteresis) {
            // Stay critical due to hysteresis
            currentSeverity = AlertSeverity.CRITICAL;
        } else if (predictedRulCycles <= cautionThreshold) {
            currentSeverity = AlertSeverity.CAUTION;
        } else if (currentSeverity == AlertSeverity.CAUTION && predictedRulCycles < cautionThreshold + resetHysteresis) {
            // Stay caution due to hysteresis
            currentSeverity = AlertSeverity.CAUTION;
        } else {
            currentSeverity = AlertSeverity.NORMAL;
        }
        return currentSeverity;
    }

    public AlertSeverity getCurrentSeverity() {
        return currentSeverity;
    }
}
