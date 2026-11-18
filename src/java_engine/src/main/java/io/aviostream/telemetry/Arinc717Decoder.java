package io.aviostream.telemetry;

/**
 * ARINC-717 flight data acquisition subframe word decoder (12-bit words).
 */
public class Arinc717Decoder {
    public static final int SYNC_SUBFRAME_1 = 0x247; // 0010 0100 0111
    public static final int SYNC_SUBFRAME_2 = 0x5B8; // 0101 1011 1000
    public static final int SYNC_SUBFRAME_3 = 0xA47; // 1010 0100 0111
    public static final int SYNC_SUBFRAME_4 = 0xDB8; // 1101 1011 1000

    public static boolean isSyncWord(int word12Bit) {
        int masked = word12Bit & 0x0FFF;
        return masked == SYNC_SUBFRAME_1 || masked == SYNC_SUBFRAME_2 ||
               masked == SYNC_SUBFRAME_3 || masked == SYNC_SUBFRAME_4;
    }

    public static double decodeSensorValue(int rawWord12Bit, double minScale, double maxScale) {
        int val = rawWord12Bit & 0x0FFF;
        double ratio = val / 4095.0;
        return minScale + ratio * (maxScale - minScale);
    }
}
