package xyz.hyhy.scai;

import ai.djl.Device;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Joints;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.translator.YoloV5Translator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.util.Pair;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import xyz.hyhy.scai.ml.AlphaPoseEstimator;
import xyz.hyhy.scai.ml.HolisticEstimator;
import xyz.hyhy.scai.ml.LPNPoseEstimator;
import xyz.hyhy.scai.ml.ParallelPoseEstimator;
import xyz.hyhy.scai.translator.SPPETranslator;
import xyz.hyhy.scai.translator.SPPETranslator2;
import xyz.hyhy.scai.utils.ImageUtils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class ModuleTest extends MainClass {


    public static void main(String[] args) {
        testFastPose();
    }

    public static void testYolov5() {
        Translator<Image, DetectedObjects> translator = YoloV5Translator.builder().optSynsetArtifactName("coco.names").build();
        Criteria<Image, DetectedObjects> criteria =
                Criteria.builder()
                        .setTypes(Image.class, DetectedObjects.class)
                        .optDevice(Device.gpu())
                        .optModelUrls(AlphaPose.class.getResource("/yolov5").getPath())
                        .optModelName("yolov5s.torchscript.pt")
                        .optTranslator(translator)
                        .optEngine("PyTorch")
                        .build();
        int height = 480, width = 640;
        // 开始测试
        long sum = 0;
        int testTimes = 8;
        try (ZooModel<Image, DetectedObjects> model = ModelZoo.loadModel(criteria)) {
            Scalar white = new Scalar(255, 255, 255);
            int n = 10;
            try (Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
                while (n-- > 0) {
                    List<Image> imgs = new ArrayList<>();
                    for (int i = 0; i < testTimes; i++) {
                        imgs.add(ImageUtils.mat2Image(new Mat(height, width, CvType.CV_8UC3, white)));
                    }
                    predictor.batchPredict(imgs);
                    long startTime = System.currentTimeMillis();
                    predictor.batchPredict(imgs);
                    sum += System.currentTimeMillis() - startTime;

//                    Mat mat = new Mat(height, width, CvType.CV_8UC3, white);
//                    long startTime = System.currentTimeMillis();
//                    predictor.predict(ImageUtils.mat2Image(mat));
//                    sum += System.currentTimeMillis() - startTime;
                    System.out.println("剩余测试次数: " + n);
                }
            }
        } catch (RuntimeException | ModelException | TranslateException | IOException e) {
            e.printStackTrace();
        }
        System.out.println(String.format("yolov5 on cpu 平均FPS: %.2f fps", 1000.0 * testTimes * 10 / sum));
    }


    public static void testFastPose2() {
        SPPETranslator2 poseTranslator = SPPETranslator2.builder().build();
        Criteria<Pair<Mat, Rectangle>, Joints> criteria =
                Criteria.builder()
                        .setTypes(poseTranslator.getPairClass(), Joints.class)
                        .optDevice(Device.cpu())
                        .optModelUrls(AlphaPose.class.getResource("/sppe").getPath())
                        .optModelName("model2.onnx")
                        .optTranslator(poseTranslator)
                        .optEngine("OnnxRuntime")
                        .build();
        int height = 256, width = 256;
        // 开始测试
        long sum = 0;
        int testTimes = 200;
        try (ZooModel<Pair<Mat, Rectangle>, Joints> poseModel = ModelZoo.loadModel(criteria)) {
            Scalar white = new Scalar(255, 255, 255);
            int n = testTimes;
            while (n-- > 0) {
                try (Predictor<Pair<Mat, Rectangle>, Joints> predictor = poseModel.newPredictor()) {
                    Mat mat = new Mat(height, width, CvType.CV_8UC3, white);
                    Rectangle rectangle = new Rectangle(0, 0, width, height);
                    long startTime = System.currentTimeMillis();
                    predictor.predict(new Pair<>(mat, rectangle));
                    sum += System.currentTimeMillis() - startTime;
                }
                System.out.println("剩余测试次数: " + n);
            }
        } catch (RuntimeException | ModelException | TranslateException | IOException e) {
            e.printStackTrace();
        }
        System.out.println(String.format("平均FPS: %.2f fps", 1000.0 * testTimes / sum));
    }

    public static void testFastPose() {
        int par = 15, mlts = 1;
        boolean gpu = false;
        int testTimes = 200;
        int testNum = 10;
        try (ParallelPoseEstimator ppe = new AlphaPoseEstimator(par, mlts, gpu)) {
            for (int i = 0; i < testNum; i++) {

                int height = 256, width = 192;
                Scalar white = new Scalar(255, 255, 255);
                Mat frame = new Mat(height, width, CvType.CV_8UC3, white);

                // 开始测试
                List<Rectangle> bboxes = new ArrayList<>(testTimes);
                int n = testTimes;
                while (n-- > 0) {
                    bboxes.add(new Rectangle(0, 0, width, height));
                }
                ppe.infer(frame, bboxes);//预热
                System.out.println(String.format("测试条件 人次:%d, 并行度: %d, 最小工作划分: %d ", testTimes, par, mlts));
                System.out.println("测试中...");
                long startTime = System.currentTimeMillis();
                List<Joints> joints = ppe.infer(frame, bboxes);
                long spendTime0 = System.currentTimeMillis() - startTime;
                System.out.println(String.format("ForkJoin平均FPS: %.2f fps, output_size: %d, 花费时间: %.2fs",
                        1000.0 * testTimes / spendTime0, joints.size(), spendTime0 / 1000.0));
                System.out.println(joints.contains(null) ? "处理结果错误" : "处理结果正确");
                // 线性推理测试
                System.out.println("测试中...");
                startTime = System.currentTimeMillis();
                List<Joints> joints2 = ppe.inferOneByOne(frame, bboxes);
                long spendTime1 = System.currentTimeMillis() - startTime;
                System.out.println(String.format("OneByOne平均FPS: %.2f fps, output_size: %d, 花费时间: %.2fs",
                        1000.0 * testTimes / spendTime1, joints.size(), spendTime1 / 1000.0));
                System.out.println(String.format("ForkJoin速度提升%.2f倍", spendTime1 * 1.0 / spendTime0));

            }
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    public static void testLPN() {
        int par = 15, mlts = 1;
        boolean gpu = false;
        int testTimes = 200;
        try (ParallelPoseEstimator ppe = new LPNPoseEstimator(par, mlts, gpu)) {
            int height = 256, width = 192;
            Scalar white = new Scalar(255, 255, 255);
            Mat frame = new Mat(height, width, CvType.CV_8UC3, white);

            // 开始测试
            List<Rectangle> bboxes = new ArrayList<>(testTimes);
            int n = testTimes;
            while (n-- > 0) {
                bboxes.add(new Rectangle(0, 0, width, height));
            }
            ppe.infer(frame, bboxes);
            System.out.println(String.format("测试条件 人次:%d, 并行度: %d, 最小工作划分: %d ", testTimes, par, mlts));
            System.out.println("测试中...");
            long startTime = System.currentTimeMillis();
            List<Joints> joints = ppe.infer(frame, bboxes);
            long spendTime0 = System.currentTimeMillis() - startTime;
            System.out.println(String.format("ForkJoin平均FPS: %.2f fps, output_size: %d, 花费时间: %.2fs",
                    1000.0 * testTimes / spendTime0, joints.size(), spendTime0 / 1000.0));
            System.out.println(joints.contains(null) ? "处理结果错误" : "处理结果正确");
            System.out.println("测试中...");
            startTime = System.currentTimeMillis();
            List<Joints> joints2 = ppe.inferOneByOne(frame, bboxes);
            long spendTime1 = System.currentTimeMillis() - startTime;
            System.out.println(String.format("OneByOne平均FPS: %.2f fps, output_size: %d, 花费时间: %.2fs",
                    1000.0 * testTimes / spendTime1, joints.size(), spendTime1 / 1000.0));
            System.out.println(String.format("ForkJoin速度提升%.2f倍", spendTime1 * 1.0 / spendTime0));
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    public static void testHolistic() {
        int par = 15, mlts = 1;
        boolean gpu = false;
        int testTimes = 200;
        int testNum = 10;
        try (ParallelPoseEstimator ppe = new HolisticEstimator(par, mlts, gpu)) {
            for (int i = 0; i < testNum; i++) {
                int height = 256, width = 192;
                Scalar white = new Scalar(255, 255, 255);
                Mat frame = new Mat(height, width, CvType.CV_8UC3, white);

                // 开始测试
                List<Rectangle> bboxes = new ArrayList<>(testTimes);
                int n = testTimes;
                while (n-- > 0) {
                    bboxes.add(new Rectangle(0, 0, width, height));
                }
                ppe.infer(frame, bboxes);
                System.out.println(String.format("测试条件 人次:%d, 并行度: %d, 最小工作划分: %d ", testTimes, par, mlts));
                System.out.println("测试中...");
                long startTime = System.currentTimeMillis();
                List<Joints> joints = ppe.infer(frame, bboxes);
                long spendTime0 = System.currentTimeMillis() - startTime;
                System.out.println(String.format("ForkJoin平均FPS: %.2f fps, output_size: %d, 花费时间: %.2fs",
                        1000.0 * testTimes / spendTime0, joints.size(), spendTime0 / 1000.0));
                System.out.println(joints.contains(null) ? "处理结果错误" : "处理结果正确");
                System.out.println("测试中...");
                startTime = System.currentTimeMillis();
                List<Joints> joints2 = ppe.inferOneByOne(frame, bboxes);
                long spendTime1 = System.currentTimeMillis() - startTime;
                System.out.println(String.format("OneByOne平均FPS: %.2f fps, output_size: %d, 花费时间: %.2fs",
                        1000.0 * testTimes / spendTime1, joints.size(), spendTime1 / 1000.0));
                System.out.println(String.format("ForkJoin速度提升%.2f倍", spendTime1 * 1.0 / spendTime0));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    public static void testClassroomAction() {
        Translator<Image, DetectedObjects> translator = YoloV5Translator.builder().optSynsetArtifactName("coco.names").build();
        Criteria<Image, DetectedObjects> criteria =
                Criteria.builder()
                        .setTypes(Image.class, DetectedObjects.class)
                        .optDevice(Device.cpu())
                        .optModelUrls(AlphaPose.class.getResource("/yolov5").getPath())
                        .optModelName("yolov5s.onnx")
                        .optTranslator(translator)
                        .optEngine("OnnxRuntime")
                        .build();
        int height = 480, width = 640;
        // 开始测试
        long sum = 0;
        int testTimes = 200;
        try (ZooModel<Image, DetectedObjects> model = ModelZoo.loadModel(criteria)) {
            Scalar white = new Scalar(255, 255, 255);
            int n = testTimes;
            while (n-- > 0) {
                try (Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
                    Mat mat = new Mat(height, width, CvType.CV_8UC3, white);
                    long startTime = System.currentTimeMillis();
                    predictor.predict(ImageUtils.mat2Image(mat));
                    sum += System.currentTimeMillis() - startTime;
                }
                System.out.println("剩余测试次数: " + n);
            }
        } catch (RuntimeException | ModelException | TranslateException | IOException e) {
            e.printStackTrace();
        }
        System.out.println(String.format("Log reg on cpu 平均FPS: %.2f fps", 1000.0 * testTimes / sum));
    }
}
