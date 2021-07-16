package xyz.hyhy.scai;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.DetectedObjects.DetectedObject;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.translator.YoloV5Translator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import xyz.hyhy.scai.utils.CommonUtils;
import xyz.hyhy.scai.utils.ImageUtils;
import xyz.hyhy.scai.utils.SceneMasker;

import java.io.IOException;

import static org.opencv.videoio.Videoio.CAP_ANY;

public class Yolov5WithMask extends MainClass {
    public static final SceneMasker masker;

    static {
        masker = new SceneMasker(Yolov5WithMask.class.getResource("/sceneMask.png").getPath().substring(1));
    }

    public static void main(String[] args) {
        Translator<Image, DetectedObjects> translator = YoloV5Translator
                .builder()
                .optSynsetArtifactName("coco.names")
                .build();
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
                        .optModelUrls(Yolov5WithMask.class.getResource("/yolov5").getPath())
                        .optModelName("yolov5s.onnx")
                        .optTranslator(translator)
                        .optEngine("OnnxRuntime")
                        .build();
        try (ZooModel<Image, DetectedObjects> model = ModelZoo.loadModel(criteria)) {
            VideoCapture cap = new VideoCapture(CAP_ANY);
            if (!cap.isOpened()) {//isOpened函数用来判断摄像头调用是否成功
                System.out.println("Camera Error");//如果摄像头调用失败，输出错误信息
            } else {
                Mat frame = new Mat();//创建一个输出帧
                boolean flag = cap.read(frame);//read方法读取摄像头的当前帧
                while (flag) {
                    detect(frame, model);
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

    static void detect(Mat frame, ZooModel<Image, DetectedObjects> model) throws IOException, ModelNotFoundException, MalformedModelException, TranslateException {
        Image img = ImageUtils.mat2Image(frame);
        long startTime = System.currentTimeMillis();
        try (Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
            DetectedObjects results = predictor.predict(img);
//            System.out.println(results);
            for (DetectedObject obj : results.<DetectedObject>items()) {
                if (!"person".equals(obj.getClassName()))
                    continue;
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
                System.out.println(isInSeat(rectangle, 639, 479));
            }
            boolean showMask = true;
            if (showMask)
                masker.fixWithMask(frame);
        }
        boolean showFPS = false;
        if (showFPS)
            System.out.println(String.format("%.2f", 1000.0 / (System.currentTimeMillis() - startTime)));
    }

    public static boolean isInSeat(Rectangle rectangle, int width, int height) {
        return masker.judge(mask -> {
            int x = (int) (rectangle.getX() + rectangle.getWidth() / 2);
            int y = (int) (rectangle.getY() + rectangle.getHeight() / 2);
            int topY = (int) rectangle.getY();
            x = CommonUtils.trimInt(x, 0, width);
            y = CommonUtils.trimInt(y, 0, height);
            topY = CommonUtils.trimInt(topY, 0, height);
            return mask[y][x] == SceneMasker.AREA_SEAT && mask[topY][x] != SceneMasker.AREA_WALL;
        });
    }
}
