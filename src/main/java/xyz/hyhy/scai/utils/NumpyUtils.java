package xyz.hyhy.scai.utils;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

public class NumpyUtils {

    public static final NDManager ndManager = NDManager.newBaseManager(Device.cpu());


    public static double normalDistributionFunction(double miu, double sigma, double x) {
        return (1 / (sigma * Math.sqrt(2 * Math.PI))) * Math.exp(-Math.pow((x - miu) / sigma, 2) / 2);
    }

    public static void main(String[] args) {
        float[] num = new float[5];
        double miu = 4.2;
        double sigma = 1.4;
        for (int i = 0; i < num.length; i++) {
            num[i] = (float) normalDistributionFunction(miu, sigma, i + 0.5);
        }
        NDArray numVec = ndManager.create(num);
        System.out.println(numVec.div(numVec.sum()));
    }
}
