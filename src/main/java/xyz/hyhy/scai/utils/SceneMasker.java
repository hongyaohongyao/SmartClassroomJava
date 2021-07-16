package xyz.hyhy.scai.utils;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

public class SceneMasker {
    protected int[][] sceneMask;
    protected Mat originMask;
    protected final static ColorEncoder colorEncoder = ColorEncoder.newInstance(
            new int[][]{
                    {0, 0, 0},
                    {255, 255, 255},
                    {0, 0, 255}
            });
    public static final int AREA_DEFAULT = 0;
    public static final int AREA_SEAT = 1;
    public static final int AREA_WALL = 2;

    private static class ColorEncoder {
        public int b;
        public int g;
        public int r;
        public int flag;
        public ColorEncoder nextEncoder;
        public final static int DEFAULT_FLAG = 0;


        private ColorEncoder(int[][] colorSet, int idx) {
            this.b = colorSet[idx][0];
            this.g = colorSet[idx][1];
            this.r = colorSet[idx][2];
            this.flag = idx;
            if (++idx < colorSet.length) {
                this.nextEncoder = new ColorEncoder(colorSet, idx);
            }
        }

        public static ColorEncoder newInstance(int[][] colorSet) {
            if (colorSet.length == 0)
                return null;
            return new ColorEncoder(colorSet, 0);
        }

        public int color2flag(int b, int g, int r) {
            if (b == this.b && g == this.g && r == this.r)
                return this.flag;
            if (nextEncoder != null)
                return nextEncoder.color2flag(b, g, r);
            return DEFAULT_FLAG;
        }

    }

    public interface JudgementByMask {
        boolean judge(int[][] mask);
    }

    public SceneMasker(String filename) {
        originMask = Imgcodecs.imread(filename);
        this.sceneMask = new int[originMask.rows()][originMask.cols()];
        for (int x = 0; x < originMask.cols(); x++) {
            for (int y = 0; y < originMask.rows(); y++) {
                this.sceneMask[y][x] = channel2Int(originMask.get(y, x));
            }
        }
    }

    public Mat fixWithMask(Mat img) {
        double transparency = 0.3;
        Core.addWeighted(originMask, transparency, img, 1 - transparency, 0, img);
        return img;
    }

    public boolean judge(JudgementByMask judgementByMask) {
        return judgementByMask.judge(sceneMask);
    }

    public static int channel2Int(double[] ch) {
        return colorEncoder.color2flag((int) ch[0], (int) ch[1], (int) ch[2]);
    }
}
