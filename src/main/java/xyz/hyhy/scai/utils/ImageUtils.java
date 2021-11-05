package xyz.hyhy.scai.utils;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.ndarray.NDManager;
import org.opencv.core.CvType;
import org.opencv.core.Mat;


public class ImageUtils {

    static {
        ImageFactory.setImageFactory(new CVImageFactory());
    }

    private ImageUtils() {

    }

    public static Image mat2Image(Mat mat) {
        return ImageFactory.getInstance().fromImage(mat);
    }


    public static Mat image2Mat(Image img) {
        byte[] buf = img.toNDArray(NDManager.newBaseManager()).toByteArray();
        Mat mat = new Mat(img.getHeight(), img.getWidth(), CvType.CV_8UC3);
        mat.put(0, 0, buf);
        return mat;
    }

    public static Image getSubImage(Image img, Rectangle rect) {
        int x = (int) Math.max(0, rect.getX());
        int y = (int) Math.max(0, rect.getY());
        int w = Math.min(img.getWidth(), (int) (x + rect.getWidth())) - x;
        int h = Math.min(img.getHeight(), (int) (y + rect.getHeight())) - y;
        return img.getSubimage(x, y, w, h);
    }

    public static Image getSubImage(Image img, int x, int y, int w, int h) {
        return img.getSubimage(x, y, w, h);
    }

    public static Mat getSubImage(Mat mat, int x, int y, int w, int h) {
//        System.out.println(Arrays.asList(x, y, w, h, mat.cols(), mat.rows()));
        return mat.submat(y, y + h, x, x + w);
    }


}
