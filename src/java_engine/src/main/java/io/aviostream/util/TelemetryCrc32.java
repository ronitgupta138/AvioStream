package io.aviostream.util;

import java.nio.charset.StandardCharsets;
import java.util.zip.CRC32;

/**
 * CRC32 checksum validator for telemetry data frames.
 */
public class TelemetryCrc32 {
    public static long compute(String data) {
        if (data == null) return 0L;
        CRC32 crc = new CRC32();
        crc.update(data.getBytes(StandardCharsets.UTF_8));
        return crc.getValue();
    }

    public static boolean verify(String data, long expectedChecksum) {
        return compute(data) == expectedChecksum;
    }
}
