package xyz.hyhy.scai;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.DetectedObjects.DetectedObject;
import ai.djl.modality.cv.output.Joints;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.translator.YoloV5Translator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.util.Pair;
import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import xyz.hyhy.scai.translator.SPPETranslator2;
import xyz.hyhy.scai.utils.CVUtils;
import xyz.hyhy.scai.utils.ImageUtils;

import java.io.IOException;

public class MediapipePose {

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
//        String source = AlphaPose.class.getResource("/front_cheat.mp4").getPath();
        int source = 0;
        Translator<Image, DetectedObjects> translator = YoloV5Translator.builder().optSynsetArtifactName("coco.names").build();
//        Criteria<Image, DetectedObjects> criteria =
//                Criteria.builder()
//                        .setTypes(Image.class, DetectedObjects.class)
//                        .optDevice(Device.cpu())
//                        .optModelUrls(Main.class.getResource("/yolov5s").getPath())
//                        .optModelName("yolov5s.torchscript.pt")
//                        .optTranslator(translator)
//                        .optEngine("PyTorch")
//                        .build();
        Criteria<Image, DetectedObjects> criteria =
                Criteria.builder()
                        .setTypes(Image.class, DetectedObjects.class)
                        .optDevice(Device.cpu())
                        .optModelUrls(MediapipePose.class.getResource("/yolov5").getPath())
                        .optModelName("yolov5s.onnx")
                        .optTranslator(translator)
                        .optEngine("OnnxRuntime")
                        .build();


        SPPETranslator2 poseTranslator = SPPETranslator2.builder().build();
        Criteria<Pair<Mat, Rectangle>, Joints> poseCriteria =
                Criteria.builder()
                        .setTypes(poseTranslator.getPairClass(), Joints.class)
                        .optDevice(Device.cpu())
                        .optModelUrls(MediapipePose.class.getResource("/sppe").getPath())
                        .optModelName("pose_landmark_full.onnx")
                        .optTranslator(poseTranslator)
                        .optEngine("OnnxRuntime")
                        .build();
        try (ZooModel<Image, DetectedObjects> model = ModelZoo.loadModel(criteria);
             ZooModel<Pair<Mat, Rectangle>, Joints> poseModel = ModelZoo.loadModel(poseCriteria)) {
            VideoCapture cap = new VideoCapture(source);
            if (!cap.isOpened()) {//isOpened函数用来判断摄像头调用是否成功
                System.out.println("Camera Error");//如果摄像头调用失败，输出错误信息
            } else {
                Mat frame = new Mat();//创建一个输出帧
                boolean flag = cap.read(frame);//read方法读取摄像头的当前帧
                while (flag) {
                    detect(frame, model, poseModel);
                    HighGui.imshow("yolov5", frame);
                    HighGui.waitKey(20);
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
                       ZooModel<Image, DetectedObjects> model,
                       ZooModel<Pair<Mat, Rectangle>, Joints> poseModel) throws IOException, ModelNotFoundException, MalformedModelException, TranslateException {
        Image img = ImageUtils.mat2Image(frame);
        long startTime = System.currentTimeMillis();
        try (Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
            DetectedObjects results = predictor.predict(img);

            for (DetectedObject obj : results.<DetectedObject>items()) {
                if ("person".equals(obj.getClassName())) {
                    BoundingBox bbox = obj.getBoundingBox();
                    Rectangle rectangle = bbox.getBounds();
                    String showText = String.format("%s: %.2f", obj.getClassName(), obj.getProbability());
                    rect.x = (int) rectangle.getX();
                    rect.y = (int) rectangle.getY();
                    rect.width = (int) rectangle.getWidth();
                    rect.height = (int) rectangle.getHeight();
                    Joints joints = predictJointsInPerson(frame, rectangle, poseModel);
                    // 画框
                    Imgproc.rectangle(frame, rect, color, 2);
                    //画名字
                    Imgproc.putText(frame, showText,
                            new Point(rect.x, rect.y),
                            Imgproc.FONT_HERSHEY_COMPLEX,
                            rectangle.getWidth() / 200,
                            color);
                    //pose
                    CVUtils.drawLandMarks(frame, joints);
                }
            }
        }
        boolean showFPS = true;
        if (showFPS)
            System.out.println(String.format("%.2f", 1000.0 / (System.currentTimeMillis() - startTime)));
    }

    private static Joints predictJointsInPerson(Mat frame, Rectangle bbox,
                                                ZooModel<Pair<Mat, Rectangle>, Joints> poseModel) throws TranslateException {

        try (Predictor<Pair<Mat, Rectangle>, Joints> predictor = poseModel.newPredictor()) {
            Joints joints = predictor.predict(new Pair<>(frame, bbox));
            return joints;
        }
    }
}
