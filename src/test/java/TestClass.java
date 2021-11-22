import ai.djl.Device;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Joints;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.YoloV5Translator;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Pipeline;
import ai.djl.translate.Translator;
import ai.djl.util.Pair;
import org.junit.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.highgui.HighGui;
import xyz.hyhy.scai.AlphaPose;
import xyz.hyhy.scai.MainClass;
import xyz.hyhy.scai.translator.PureImageTranslator;
import xyz.hyhy.scai.translator.PureTranslator;
import xyz.hyhy.scai.translator.SPPETranslator;
import xyz.hyhy.scai.translator.SimpSPPETranslator;
import xyz.hyhy.scai.utils.CVUtils;
import xyz.hyhy.scai.utils.ImageUtils;

import java.awt.image.BufferedImage;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class TestClass extends MainClass {
    public <IN, OUT> void test(ZooModel<IN, OUT> model,
                               IN inp,
                               int inpNum,
                               int warmupNum) throws Exception {

        try (Predictor<IN, OUT> predictor = model.newPredictor()) {
            if (warmupNum > 0) {
                System.out.printf("预热中(%d次)\n", warmupNum);
                for (int i = 0; i < warmupNum; i++) {
                    predictor.predict(inp);
                }
            }
            System.out.printf("开始测试(%d次)\n", inpNum);
            long startTime = System.currentTimeMillis();
            for (int i = 0; i < inpNum; i++) {
                predictor.predict(inp);
            }
            long endTime = System.currentTimeMillis();
            System.out.printf("处理速度: %f s/img%n\n", (endTime - startTime) / 1000.0 / inpNum);
        }
    }

    public <IN, OUT> void testBatch(ZooModel<IN, OUT> model,
                                    IN inp,
                                    int inpNum,
                                    int warmupNum,
                                    int batchSize) throws Exception {
        List<IN> inputs = new ArrayList<>(batchSize);
        for (int i = 0; i < batchSize; i++) {
            inputs.add(inp);
        }
        try (Predictor<IN, OUT> predictor = model.newPredictor()) {
            if (warmupNum > 0) {
                System.out.printf("预热中(%d次)\n", warmupNum);
                for (int i = 0; i < warmupNum; i++) {
                    predictor.batchPredict(inputs);
                }
            }
            System.out.printf("开始测试(%d次)\n", inpNum);
            long startTime = System.currentTimeMillis();
            for (int i = 0; i < inpNum; i++) {
                predictor.batchPredict(inputs);
            }
            long endTime = System.currentTimeMillis();
            System.out.printf("处理速度: %f s/img%n\n", (endTime - startTime) / 1000.0 / inpNum);
        }
    }

    @Test
    public void yolov5Test() throws Exception {
        int imgsNum = 1000;
        int warmupNum = 50;
        Scalar white = new Scalar(0, 0, 0);
        int height = 640, width = 640;
        Image img = ImageUtils.mat2Image(new Mat(height, width, CvType.CV_8UC3, white));
        Translator<Image, DetectedObjects> translator = YoloV5Translator.builder().optSynsetArtifactName("coco.names").build();
        Criteria<Image, DetectedObjects> criteria = Criteria.builder()
                .setTypes(Image.class, DetectedObjects.class)
                .optDevice(Device.gpu())
                .optModelUrls(AlphaPose.class.getResource("/yolov5").getPath())
                .optModelName("yolov5s.torchscript.pt")
                .optTranslator(translator)
                .optEngine("PyTorch")
                .build();
        test(ModelZoo.loadModel(criteria), img, imgsNum, warmupNum);
    }

    @Test
    public void resnet101Test() throws Exception {
        int imgsNum = 1000;
        int warmupNum = 50;
        Scalar white = new Scalar(0, 0, 0);
        int height = 640, width = 640;
        Image img = ImageUtils.mat2Image(new Mat(height, width, CvType.CV_8UC3, white));
        Translator<Image, Void> translator = new PureImageTranslator.Builder().build(true);
        Criteria<Image, Void> criteria = Criteria.builder()
                .setTypes(Image.class, Void.class)
                .optDevice(Device.gpu())
                .optModelUrls(Objects.requireNonNull(AlphaPose.class.getResource("/others")).getPath())
                .optModelName("resnet101.torchscript.pth")
                .optTranslator(translator)
                .optEngine("PyTorch")
                .build();
        test(ModelZoo.loadModel(criteria), img, imgsNum, warmupNum);
    }

    @Test
    public void resnet101WithPreprocessTest() throws Exception {
        int imgsNum = 1000;
        int warmupNum = 50;
        Scalar white = new Scalar(0, 0, 0);
        int height = 640, width = 640;
        Image img = ImageUtils.mat2Image(new Mat(height, width, CvType.CV_8UC3, white));
        Translator<Image, Void> translator = new PureImageTranslator.Builder().build(height, width);
        Criteria<Image, Void> criteria = Criteria.builder()
                .setTypes(Image.class, Void.class)
                .optDevice(Device.gpu())
                .optModelUrls(Objects.requireNonNull(AlphaPose.class.getResource("/others")).getPath())
                .optModelName("resnet101.torchscript.pth")
                .optTranslator(translator)
                .optEngine("PyTorch")
                .build();
        test(ModelZoo.loadModel(criteria), img, imgsNum, warmupNum);
    }

    @Test
    public void resnet18Test() throws Exception {
        int imgsNum = 1000;
        int warmupNum = 50;
        Scalar white = new Scalar(0, 0, 0);
        int height = 640, width = 640;
        Image img = ImageUtils.mat2Image(new Mat(height, width, CvType.CV_8UC3, white));
        Translator<Image, Void> translator = new PureImageTranslator.Builder().build(height, width);
        Criteria<Image, Void> criteria = Criteria.builder()
                .setTypes(Image.class, Void.class)
                .optDevice(Device.gpu())
                .optModelUrls(Objects.requireNonNull(AlphaPose.class.getResource("/others")).getPath())
                .optModelName("resnet18.torchscript.pth")
                .optTranslator(translator)
                .optEngine("PyTorch")
                .build();
        test(ModelZoo.loadModel(criteria), img, imgsNum, warmupNum);
    }

    @Test
    public void yoloV5WithToTensorTest() throws Exception {
        int imgsNum = 1000;
        int warmupNum = 50;
        Scalar white = new Scalar(0, 0, 0);
        int height = 640, width = 640;
        Image img = ImageUtils.mat2Image(new Mat(height, width, CvType.CV_8UC3, white));
        Translator<Image, Void> translator = new PureImageTranslator.Builder().build(true);
        Criteria<Image, Void> criteria = Criteria.builder()
                .setTypes(Image.class, Void.class)
                .optDevice(Device.gpu())
                .optModelUrls(Objects.requireNonNull(AlphaPose.class.getResource("/yolov5")).getPath())
                .optModelName("yolov5s.torchscript.pt")
                .optTranslator(translator)
                .optEngine("PyTorch")
                .build();
        test(ModelZoo.loadModel(criteria), img, imgsNum, warmupNum);
    }

    @Test
    public void pureYoloV5Test() throws Exception {
        int imgsNum = 1000;
        int warmupNum = 50;
        int height = 640, width = 640;
        Translator<Shape, Void> translator = new PureTranslator();
        Criteria<Shape, Void> criteria = Criteria.builder()
                .setTypes(Shape.class, Void.class)
                .optDevice(Device.gpu())
                .optModelUrls(Objects.requireNonNull(AlphaPose.class.getResource("/yolov5")).getPath())
                .optModelName("yolov5s.torchscript.pt")
                .optTranslator(translator)
                .optEngine("PyTorch")
                .build();
        ZooModel<Shape, Void> model = ModelZoo.loadModel(criteria);
        test(model, new Shape(3, height, width), imgsNum, warmupNum);
    }

    @Test
    public void pureYoloV5FP16Test() throws Exception {
        int imgsNum = 1000;
        int warmupNum = 50;
        int height = 640, width = 640;
        Translator<Shape, Void> translator = new PureTranslator(true);
        Criteria<Shape, Void> criteria = Criteria.builder()
                .setTypes(Shape.class, Void.class)
                .optDevice(Device.gpu())
                .optModelUrls(Objects.requireNonNull(AlphaPose.class.getResource("/yolov5")).getPath())
                .optModelName("yolov5s_h.torchscript.pt")
                .optTranslator(translator)
                .optEngine("PyTorch")
                .build();
        ZooModel<Shape, Void> model = ModelZoo.loadModel(criteria);
        test(model, new Shape(3, height, width), imgsNum, warmupNum);
    }

    @Test
    public void pureYoloV5TestOnOnnx() throws Exception {
        int imgsNum = 1000;
        int warmupNum = 50;
        int height = 480, width = 640;
        Translator<Shape, Void> translator = new PureTranslator();
        Criteria<Shape, Void> criteria = Criteria.builder()
                .setTypes(Shape.class, Void.class)
                .optDevice(Device.gpu())
                .optModelUrls(Objects.requireNonNull(AlphaPose.class.getResource("/yolov5")).getPath())
                .optModelName("yolov5s.onnx")
                .optTranslator(translator)
                .optEngine("OnnxRuntime")
                .build();
        ZooModel<Shape, Void> model = ModelZoo.loadModel(criteria);
        test(model, new Shape(3, height, width), imgsNum, warmupNum);
    }

    @Test
    public void pureResnet18Test() throws Exception {
        int imgsNum = 1000;
        int warmupNum = 50;
        Scalar white = new Scalar(0, 0, 0);
        int height = 640, width = 640;
        Translator<Shape, Void> translator = new PureTranslator();
        Criteria<Shape, Void> criteria = Criteria.builder()
                .setTypes(Shape.class, Void.class)
                .optDevice(Device.gpu())
                .optModelUrls(Objects.requireNonNull(AlphaPose.class.getResource("/others")).getPath())
                .optModelName("resnet18.torchscript.pth")
                .optTranslator(translator)
                .optEngine("PyTorch")
                .build();
        ZooModel<Shape, Void> model = ModelZoo.loadModel(criteria);
        test(model, new Shape(3, height, width), imgsNum, warmupNum);
    }

    @Test
    public void pureResnet101Test() throws Exception {
        int imgsNum = 1000;
        int warmupNum = 50;
        int height = 640, width = 640;
        Translator<Shape, Void> translator = new PureTranslator();
        Criteria<Shape, Void> criteria = Criteria.builder()
                .setTypes(Shape.class, Void.class)
                .optDevice(Device.gpu())
                .optModelUrls(Objects.requireNonNull(AlphaPose.class.getResource("/others")).getPath())
                .optModelName("resnet101.torchscript.pth")
                .optTranslator(translator)
                .optEngine("PyTorch")
                .build();
        ZooModel<Shape, Void> model = ModelZoo.loadModel(criteria);
        test(model, new Shape(3, height, width), imgsNum, warmupNum);
    }

    @Test
    public void toTensorTest() throws Exception {
        int inpNum = 1000;
        int height = 640, width = 640;
        NDManager ndManager = NDManager.newBaseManager(Device.gpu());
        Pipeline pipeline = new Pipeline(new ToTensor());
        Shape shape = new Shape(height, width, 3);

        System.out.printf("开始测试(%d次)\n", inpNum);
        long startTime = System.currentTimeMillis();
        for (int i = 0; i < inpNum; i++) {
            try (NDManager subManager = ndManager.newSubManager()) {
                pipeline.transform(new NDList(subManager.randomNormal(shape)));
            }
        }
        long endTime = System.currentTimeMillis();
        System.out.printf("处理速度: %f s/img%n\n", (endTime - startTime) / 1000.0 / inpNum);
    }

    @Test
    public void normalizeTest() throws Exception {
        int inpNum = 1000;
        int height = 640, width = 640;
        NDManager ndManager = NDManager.newBaseManager(Device.gpu());
        Pipeline pipeline = new Pipeline(new ToTensor(),
                new Normalize(
                        new float[]{0.406f, 0.457f, 0.480f},
                        new float[]{1, 1, 1}
                ));
        Shape shape = new Shape(height, width, 3);

        System.out.printf("开始测试(%d次)\n", inpNum);
        long startTime = System.currentTimeMillis();
        for (int i = 0; i < inpNum; i++) {
            try (NDManager subManager = ndManager.newSubManager()) {
                pipeline.transform(new NDList(subManager.randomNormal(shape)));
            }
        }
        long endTime = System.currentTimeMillis();
        System.out.printf("处理速度: %f s/img%n\n", (endTime - startTime) / 1000.0 / inpNum);
    }

    @Test
    public void Image2NDArrayTest() throws Exception {
        int inpNum = 1000;
        int height = 640, width = 640;

        Scalar white = new Scalar(0, 0, 0);
        Image img = ImageUtils.mat2Image(new Mat(height, width, CvType.CV_8UC3, white));

        NDManager ndManager = NDManager.newBaseManager(Device.gpu());

        System.out.printf("开始测试(%d次)\n", inpNum);
        long startTime = System.currentTimeMillis();
        for (int i = 0; i < inpNum; i++) {
            try (NDManager subManager = ndManager.newSubManager()) {
                img.toNDArray(subManager, Image.Flag.COLOR);
            }
        }
        long endTime = System.currentTimeMillis();
        System.out.printf("处理速度: %f s/img%n\n", (endTime - startTime) / 1000.0 / inpNum);
    }

    @Test
    public void managerTest() throws Exception {
        int inpNum = 1000;
        int height = 640, width = 640;

        NDManager ndManager = NDManager.newBaseManager(Device.gpu());

        System.out.printf("开始测试(%d次)\n", inpNum);
        long startTime = System.currentTimeMillis();
        for (int i = 0; i < inpNum; i++) {
            try (NDManager subManager = ndManager.newSubManager()) {
                ByteBuffer bb = subManager.allocateDirect(3 * height * width);
                subManager.create(bb, new Shape(height, width, 3), DataType.UINT8);
            }
        }
        long endTime = System.currentTimeMillis();
        System.out.printf("处理速度: %f s/img%n\n", (endTime - startTime) / 1000.0 / inpNum);
    }

    @Test
    public void byteBufferTest() throws Exception {
        int inpNum = 1000;
        int height = 640, width = 640;
        int size = height * width;

        System.out.printf("开始测试(%d次)\n", inpNum);
        long startTime = System.currentTimeMillis();
        for (int i = 0; i < inpNum; i++) {
            ByteBuffer bb = ByteBuffer.allocateDirect(size * 3);
            for (int j = 0; j < size; j++) {
                bb.put((byte) 1);
                bb.put((byte) 1);
                bb.put((byte) 1);
            }
        }
        long endTime = System.currentTimeMillis();
        System.out.printf("处理速度: %f s/img%n\n", (endTime - startTime) / 1000.0 / inpNum);
    }

    @Test
    public void bufferedImageTest() throws Exception {
        int inpNum = 1000;
        int height = 640, width = 640;

        Scalar white = new Scalar(0, 0, 0);
        BufferedImage img = (BufferedImage) HighGui.toBufferedImage(new Mat(height, width, CvType.CV_8UC3, white));

        System.out.printf("开始测试(%d次)\n", inpNum);
        long startTime = System.currentTimeMillis();
        for (int i = 0; i < inpNum; i++) {
            img.getRGB(0, 0, width, height, null, 0, width);
        }
        long endTime = System.currentTimeMillis();
        System.out.printf("处理速度: %f s/img%n\n", (endTime - startTime) / 1000.0 / inpNum);
    }


    @Test
    public void copyNDListTest() throws Exception {
        int inpNum = 1000;
        int height = 640, width = 640;

        NDManager ndManager = NDManager.newBaseManager(Device.gpu());
        System.out.printf("开始测试(%d次)\n", inpNum);
        long startTime = System.currentTimeMillis();
        for (int i = 0; i < inpNum; i++) {
            try (NDManager subManager = ndManager.newSubManager()) {
                ByteBuffer bb = subManager.allocateDirect(3 * height * width);
                NDArray ndArray = subManager.create(bb, new Shape(height, width, 3), DataType.UINT8);
                ndManager.create(ndArray.toByteArray(), ndArray.getShape());
            }
        }
        long endTime = System.currentTimeMillis();
        System.out.printf("处理速度: %f s/img%n\n", (endTime - startTime) / 1000.0 / inpNum);
    }

    @Test
    public void testTest() throws Exception {
        int height = 640, width = 640;

        Scalar white = new Scalar(0, 0, 0);
        Image img = ImageUtils.mat2Image(new Mat(height, width, CvType.CV_8UC3, white));

        System.out.println(img.getClass().getName());
    }

    @Test
    public void SPPETest() throws Exception {
        int imgsNum = 1000;
        int warmupNum = 50;
        int height = 256, width = 192;
        Translator<Shape, Void> translator = new PureTranslator();
        Criteria<Shape, Void> criteria = Criteria.builder()
                .setTypes(Shape.class, Void.class)
                .optDevice(Device.gpu())
                .optModelUrls(Objects.requireNonNull(AlphaPose.class.getResource("/sppe")).getPath())
                .optModelName("halpe136_mobile.torchscript.pth")
                .optTranslator(translator)
                .optEngine("PyTorch")
                .build();
        ZooModel<Shape, Void> model = ModelZoo.loadModel(criteria);
        test(model, new Shape(3, height, width), imgsNum, warmupNum);
    }

    @Test
    public void SPPEBatchTest() throws Exception {
        int imgsNum = 500;
        int warmupNum = 10;
        int height = 256, width = 192;
        Translator<Shape, Void> translator = new PureTranslator();
        Criteria<Shape, Void> criteria = Criteria.builder()
                .setTypes(Shape.class, Void.class)
                .optDevice(Device.gpu())
                .optModelUrls(Objects.requireNonNull(AlphaPose.class.getResource("/sppe")).getPath())
                .optModelName("halpe136_mobile.torchscript.pth")
                .optTranslator(translator)
                .optEngine("PyTorch")
                .build();
        ZooModel<Shape, Void> model = ModelZoo.loadModel(criteria);
        testBatch(model, new Shape(3, height, width), imgsNum, warmupNum, 50);
    }

    @Test
    public void SPPETranslatorTest() throws Exception {
        int imgsNum = 1000;
        int warmupNum = 50;
        Scalar white = new Scalar(0, 0, 0);
        int height = 640, width = 640, h = 256, w = 192;
        Mat img = new Mat(height, width, CvType.CV_8UC3, white);
        Rectangle rect = new Rectangle(
                (width - w) / 2.0,
                (height - h) / 2.0,
                w,
                h
        );
        SPPETranslator translator = SPPETranslator.builder().build();
        Criteria<Pair<Mat, Rectangle>, Joints> criteria = Criteria.builder()
                .setTypes(translator.getPairClass(), Joints.class)
                .optDevice(Device.gpu())
                .optModelUrls(AlphaPose.class.getResource("/sppe").getPath())
                .optModelName("halpe136_mobile.torchscript.pth")
                .optTranslator(translator)
                .optEngine("PyTorch")
                .build();
        test(ModelZoo.loadModel(criteria), new Pair<>(img, rect), imgsNum, warmupNum);
    }


    @Test
    public void SPPETranslatorBatchTest() throws Exception {
        int imgsNum = 100;
        int warmupNum = 10;
        Scalar white = new Scalar(0, 0, 0);
        int height = 640, width = 640, h = 256, w = 192;
        Mat img = new Mat(height, width, CvType.CV_8UC3, white);
        Rectangle rect = new Rectangle(
                (width - w) / 2.0,
                (height - h) / 2.0,
                w,
                h
        );
        SPPETranslator translator = SPPETranslator.builder().build();
        Criteria<Pair<Mat, Rectangle>, Joints> criteria = Criteria.builder()
                .setTypes(translator.getPairClass(), Joints.class)
                .optDevice(Device.gpu())
                .optModelUrls(AlphaPose.class.getResource("/sppe").getPath())
                .optModelName("halpe136_mobile.torchscript.pth")
                .optTranslator(translator)
                .optEngine("PyTorch")
                .build();
        testBatch(ModelZoo.loadModel(criteria), new Pair<>(img, rect), imgsNum, warmupNum, 30);
    }

    @Test
    public void SPPETranslatorSimpBatchTest() throws Exception {
        int imgsNum = 100;
        int warmupNum = 10;
        Scalar white = new Scalar(0, 0, 0);
        int height = 640, width = 640, h = 256, w = 192;
        Mat img = new Mat(height, width, CvType.CV_8UC3, white);
        Rectangle rect = new Rectangle(
                (width - w) / 2.0,
                (height - h) / 2.0,
                w,
                h
        );
        SimpSPPETranslator translator = SimpSPPETranslator.builder().build();
        Criteria<Pair<Mat, Rectangle>, Joints> criteria = Criteria.builder()
                .setTypes(translator.getPairClass(), Joints.class)
                .optDevice(Device.gpu())
                .optModelUrls(AlphaPose.class.getResource("/sppe").getPath())
                .optModelName("halpe136_mobile_simp.torchscript.pth")
                .optTranslator(translator)
                .optEngine("PyTorch")
                .build();
        testBatch(ModelZoo.loadModel(criteria), new Pair<>(img, rect), imgsNum, warmupNum, 50);
    }

    @Test
    public void scaleTest() throws Exception {
        int inpNum = 1000;
        int height = 640, width = 640, bh = 256, bw = 192;
        Scalar white = new Scalar(0, 0, 0);
        Mat img = new Mat(height, width, CvType.CV_8UC3, white);

        Rectangle bbox = new Rectangle(
                (width - bw) / 2.0,
                (height - bh) / 2.0,
                bw,
                bh
        );
        int x = (int) Math.max(0, bbox.getX());
        int y = (int) Math.max(0, bbox.getY());
        int w = Math.min(img.width(), (int) (x + bbox.getWidth())) - x;
        int h = Math.min(img.height(), (int) (y + bbox.getHeight())) - y;
        System.out.printf("开始测试(%d次)\n", inpNum);
        long startTime = System.currentTimeMillis();
        for (int i = 0; i < inpNum; i++) {
            Mat frame = img.clone();
            CVUtils.scale(frame, x, y, w, h);
        }
        long endTime = System.currentTimeMillis();
        System.out.printf("处理速度: %f s/img%n\n", (endTime - startTime) / 1000.0 / inpNum);
    }

}
