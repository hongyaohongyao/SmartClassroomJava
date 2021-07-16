package xyz.hyhy.scai;

import ai.djl.MalformedModelException;
import ai.djl.ModelException;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.DetectedObjects.DetectedObject;
import ai.djl.modality.cv.output.Joints;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.translate.TranslateException;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import xyz.hyhy.scai.constant.ColorConst;
import xyz.hyhy.scai.ml.AlphaPoseEstimator;
import xyz.hyhy.scai.ml.ParallelPoseEstimator;
import xyz.hyhy.scai.ml.YoloV5Detector;
import xyz.hyhy.scai.utils.CVUtils;
import xyz.hyhy.scai.utils.ImageUtils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class AlphaPose extends MainClass {

    public static void main(String[] args) throws Exception {
        String source = AlphaPose.class.getResource("/front_cheat.mp4").getPath();
//        int source = 0;

        int par = 3, mlts = 10;
        boolean gpu = true;
        try (YoloV5Detector detector = new YoloV5Detector();
             ParallelPoseEstimator ppe = new AlphaPoseEstimator(par, mlts, gpu)) {
            VideoCapture cap = new VideoCapture(source);
            if (!cap.isOpened()) {//isOpened函数用来判断摄像头调用是否成功
                System.out.println("Camera Error");//如果摄像头调用失败，输出错误信息
            } else {
                Mat frame = new Mat();//创建一个输出帧
                boolean flag = cap.read(frame);//read方法读取摄像头的当前帧
                while (flag) {
                    detect(frame, detector, ppe);
                    HighGui.imshow("yolov5", frame);
                    HighGui.waitKey(100);
                    flag = cap.read(frame);
                }
            }

        } catch (RuntimeException | ModelException | TranslateException | IOException e) {
            e.printStackTrace();
        }
    }

    static Rect rect = new Rect();
    static Scalar color = new Scalar(0, 255, 0);

    static void detect(Mat frame,
                       YoloV5Detector detector,
                       ParallelPoseEstimator ppe) throws IOException, ModelNotFoundException, MalformedModelException, TranslateException {
        Image img = ImageUtils.mat2Image(frame);
        long startTime = System.currentTimeMillis();
        try {
            DetectedObjects results = detector.detect(img);
            List<DetectedObject> detectedObjects = new ArrayList<>(results.getNumberOfObjects());
            List<Rectangle> jointsInput = new ArrayList<>(results.getNumberOfObjects());
            for (DetectedObject obj : results.<DetectedObject>items()) {
                if ("person".equals(obj.getClassName())) {
                    detectedObjects.add(obj);
                    jointsInput.add(obj.getBoundingBox().getBounds());
                }
            }
            List<Joints> joints = ppe.infer(frame, jointsInput);
            for (DetectedObject obj : detectedObjects) {
                BoundingBox bbox = obj.getBoundingBox();
                Rectangle rectangle = bbox.getBounds();
                String showText = String.format("%s: %.2f", obj.getClassName(), obj.getProbability());
                rect.x = (int) rectangle.getX();
                rect.y = (int) rectangle.getY();
                rect.width = (int) rectangle.getWidth();
                rect.height = (int) rectangle.getHeight();
                // 画框
                Imgproc.rectangle(frame, rect, color, 2);
                //画名字
                Imgproc.putText(frame, showText,
                        new Point(rect.x, rect.y),
                        Imgproc.FONT_HERSHEY_COMPLEX,
                        rectangle.getWidth() / 200,
                        color);
            }
            for (Joints jointsItem : joints) {
                CVUtils.draw136KeypointsLight(frame, jointsItem);
            }
        } finally {

        }
        boolean showFPS = true;
        if (showFPS) {
            double fps = 1000.0 / (System.currentTimeMillis() - startTime);
            System.out.println(String.format("%.2f", fps));
            Imgproc.putText(frame, String.format("FPS: %.2f", fps),
                    new Point(0, 52),
                    Imgproc.FONT_HERSHEY_COMPLEX,
                    0.5,
                    ColorConst.COLOR_RED);
        }

    }

}
