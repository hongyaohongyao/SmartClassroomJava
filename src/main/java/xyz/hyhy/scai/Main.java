package xyz.hyhy.scai;

import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import xyz.hyhy.scai.utils.CVUtils;
import xyz.hyhy.scai.utils.ImageUtils;
import xyz.hyhy.scai.utils.NumpyUtils;

public class Main extends MainClass {
    public static int[] nums;
    public static double count = 0;


    public static void main(String[] args) {
        NDManager ndManager = NumpyUtils.ndManager;
        NDArray x1 = ndManager.zeros(new Shape(1, 136, 2));
        NDArray x2 = ndManager.zeros(new Shape(2, 3));
        NDArray ones = ndManager.ones(new Shape(1, 136, 1));
        NDArray concat = x1.concat(ones, 2).reshape(new Shape(136, -1));
        System.out.println(concat.getShape());
        System.out.println();
    }
}
