package xyz.hyhy.scai;

import ai.djl.MalformedModelException;
import ai.djl.ModelException;
import ai.djl.modality.Classifications;
import ai.djl.modality.Classifications.Classification;
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
import xyz.hyhy.scai.ml.*;
import xyz.hyhy.scai.utils.CVUtils;
import xyz.hyhy.scai.utils.ImageUtils;
import xyz.hyhy.scai.utils.PoseEstimator;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class ClassAction extends MainClass {
    public static final int MODEL_TYPE = ClassroomActionClassifier.TYPE_HALPE;

    public static void main(String[] args) throws Exception {
        String source = AlphaPose.class.getResource("/front_cheat.mp4").getPath();
//        int source = 0;
        try (YoloV5Detector detector = new YoloV5Detector();
             ParallelPoseEstimator ppe = getParallelPoseEstimator();
             ClassroomActionClassifier classifier =
                     new ClassroomActionClassifier(MODEL_TYPE)) {
            PoseEstimator pe = new PoseEstimator(640, 480);
            VideoCapture cap = new VideoCapture(source);
            if (!cap.isOpened()) {//isOpened函数用来判断摄像头调用是否成功
                System.out.println("Camera Error");//如果摄像头调用失败，输出错误信息
            } else {
                Mat frame = new Mat();//创建一个输出帧
                boolean flag = cap.read(frame);//read方法读取摄像头的当前帧
                while (flag) {
                    detect(frame, detector, classifier, ppe, pe);
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
    static Scalar COLOR_GREEN = new Scalar(0, 255, 0);
    static Scalar COLOR_RED = new Scalar(0, 0, 255);

    static void detect(Mat frame,
                       YoloV5Detector detector,
                       ClassroomActionClassifier classifier,
                       ParallelPoseEstimator ppe,
                       PoseEstimator pe) throws IOException, ModelNotFoundException, MalformedModelException, TranslateException {
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
            // 姿态估计
            List<Joints> joints = ppe.infer(frame, jointsInput);
            // 姿态分类
            List<Classifications> classifications = classifier.classify(joints);
            int i = 0;
            for (DetectedObject obj : detectedObjects) {
                BoundingBox bbox = obj.getBoundingBox();
                Rectangle rectangle = bbox.getBounds();
                String showText = String.format("%s: %.2f", obj.getClassName(), obj.getProbability());
                rect.x = (int) rectangle.getX();
                rect.y = (int) rectangle.getY();
                rect.width = (int) rectangle.getWidth();
                rect.height = (int) rectangle.getHeight();
                // 画框
                Imgproc.rectangle(frame, rect, COLOR_GREEN, 2);
                //画名字
                Imgproc.putText(frame, showText,
                        new Point(rect.x, rect.y),
                        Imgproc.FONT_HERSHEY_COMPLEX,
                        rectangle.getWidth() / 200,
                        COLOR_GREEN);
                int textPos = 52;

//                for (Classification classification : classifications.get(i).items()) {
//                    String showClass = String.format("%s: %.2f", classification.getClassName(),
//                            classification.getProbability());
//                    Imgproc.putText(frame, showClass,
//                            new Point(rect.x, rect.y + textPos),
//                            Imgproc.FONT_HERSHEY_COMPLEX,
//                            0.5,
//                            COLOR_RED);
//                    textPos += 20;
//                }
                Classification classification = classifications.get(i).best();
                String showClass = String.format("%s: %.2f", classification.getClassName(),
                        classification.getProbability());
                Imgproc.putText(frame, showClass,
                        new Point(rect.x, rect.y + textPos),
                        Imgproc.FONT_HERSHEY_COMPLEX,
                        0.5,
                        COLOR_RED);
                textPos += 20;

                if (MODEL_TYPE == ClassroomActionClassifier.TYPE_COCO) {
                    CVUtils.draw86KeypointsLight(frame, joints.get(i));
                } else if (MODEL_TYPE == ClassroomActionClassifier.TYPE_HALPE) {
                    CVUtils.draw136KeypointsLight(frame, joints.get(i));
                } else {
                    CVUtils.draw86KeypointsLight(frame, joints.get(i));
                }

                int startIndex, endIndex;
                // 姿态估计
                if (MODEL_TYPE == ClassroomActionClassifier.TYPE_COCO) {
                    startIndex = 18;
                    endIndex = 86;
                } else if (MODEL_TYPE == ClassroomActionClassifier.TYPE_HALPE) {
                    startIndex = 26;
                    endIndex = 94;
                } else {
                    startIndex = 18;
                    endIndex = 86;
                }
                Mat[] rVecNtVec = pe.solvePose(joints.get(i).getJoints().subList(startIndex, endIndex));
                pe.drawAxis(frame, rVecNtVec[0], rVecNtVec[1]);
                i++;
            }
        } finally {

        }
        boolean showFPS = true;
        if (showFPS)
            System.out.println(String.format("%.2f", 1000.0 / (System.currentTimeMillis() - startTime)));
    }

    public static ParallelPoseEstimator getParallelPoseEstimator() throws Exception {
        int par = 3, mlts = 10;
        boolean gpu = true;
        switch (MODEL_TYPE) {
            case ClassroomActionClassifier.TYPE_HALPE: {
                return new AlphaPoseEstimator(par, mlts, gpu);
            }
            case ClassroomActionClassifier.TYPE_COCO: {
                return new HolisticEstimator(par, mlts, gpu);
            }
            default: {
                System.out.println("default_model");
                return new HolisticEstimator();
            }
        }
    }

}
