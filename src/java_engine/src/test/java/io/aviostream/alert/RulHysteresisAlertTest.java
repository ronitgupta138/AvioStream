package io.aviostream.alert;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class RulHysteresisAlertTest {
    @Test
    void testAlertTransitionsAndHysteresis() {
        RulHysteresisAlert alert = new RulHysteresisAlert(15.0, 30.0, 5.0);

        assertEquals(RulHysteresisAlert.AlertSeverity.NORMAL, alert.evaluate(50.0));
        assertEquals(RulHysteresisAlert.AlertSeverity.CAUTION, alert.evaluate(25.0));
        assertEquals(RulHysteresisAlert.AlertSeverity.CRITICAL, alert.evaluate(12.0));

        // When RUL recovers to 18 (below 15 + 5), it should stay critical
        assertEquals(RulHysteresisAlert.AlertSeverity.CRITICAL, alert.evaluate(18.0));

        // When RUL recovers past 20, drops to caution
        assertEquals(RulHysteresisAlert.AlertSeverity.CAUTION, alert.evaluate(22.0));

        // Above 35 (30 + 5), returns to normal
        assertEquals(RulHysteresisAlert.AlertSeverity.NORMAL, alert.evaluate(36.0));
    }
}
