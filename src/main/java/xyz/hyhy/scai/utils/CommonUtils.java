package xyz.hyhy.scai.utils;

import java.text.SimpleDateFormat;
import java.util.Date;

public class CommonUtils {
    private CommonUtils() {

    }

    private static SimpleDateFormat simpledateformat = new SimpleDateFormat("yyyyMMddhhmmss");

    public static String getCurrentDateString() {
        return simpledateformat.format(new Date());
    }

    public static String getRandomIDWithDate() {
        return getCurrentDateString() + String.format("09d", (int) (Math.random() * 999999999));
    }

    public static int trimInt(int value, int min, int max) {
        return Math.max(min, Math.min(value, max));
    }

}
