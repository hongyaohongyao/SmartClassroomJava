package xyz.hyhy.scai.exemodules;

import ai.djl.modality.Classifications;
import ai.djl.modality.Classifications.Classification;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects.DetectedObject;
import ai.djl.modality.cv.output.Joints;
import ai.djl.modality.cv.output.Rectangle;
import org.bytedeco.javacv.FFmpegFrameRecorder;
import org.bytedeco.javacv.FrameRecorder;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import xyz.hyhy.scai.constant.ColorConst;
import xyz.hyhy.scai.core.commons.DataDict;
import xyz.hyhy.scai.core.modules.ExeModule;
import xyz.hyhy.scai.pojo.ClassActionPOJO;
import xyz.hyhy.scai.utils.CVUtils;
import xyz.hyhy.scai.utils.FFmpegUtils;
import xyz.hyhy.scai.utils.PoseEstimator;

import java.util.List;


public class DrawModule extends ExeModule {
    static Rect rect = new Rect();
    static Scalar color = new Scalar(0, 255, 0);
    public long lastTime = System.currentTimeMillis();
    private PoseEstimator pe;
    private static final OpenCVFrameConverter.ToIplImage converter = new OpenCVFrameConverter.ToIplImage();
    //    VideoWriter writer = new VideoWriter("appsrc ! videoconvert ! video/x-raw,format=I420,clock-rate=90000 ! omxh264enc ! video/x-h264,stream-format=(string)byte-stream,alignment=(string)au ! h264parse ! queue ! flvmux ! rtmpsink location=rtmp://localhost:1935/testai",
//            0,
//            25.0, new Size(800, 480), true);
    private FrameRecorder recorder;

    @Override
    protected int processData(DataDict data, DataDict globalData) {
        try {
            ClassActionPOJO pojo = (ClassActionPOJO) data.get("pojo");
            List<DetectedObject> results = pojo.detectedObjects;
            Mat frame = pojo.frame;
            List<Classifications> classifications = pojo.classifications;
            int i = 0;
            for (DetectedObject obj : results) {
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
                // 绘制分类信息
                int textPos = 52;
                if (classifications != null) {
                    Classification classification = classifications.get(i).best();
                    String showClass = String.format("%s: %.2f", classification.getClassName(),
                            classification.getProbability());
                    Imgproc.putText(frame, showClass,
                            new Point(rect.x, rect.y + textPos),
                            Imgproc.FONT_HERSHEY_COMPLEX,
                            0.6,
                            ColorConst.COLOR_RED, 2);
                    textPos += 20;
                }
                i++;
            }
            // 绘制骨骼关键点
            List<Joints> joints = pojo.joints;
            if (joints != null)
                for (Joints jointsItem : joints) {
                    CVUtils.draw136KeypointsLight(frame, jointsItem);
                }
            // 绘制头部姿态
            List<Mat[]> headPoses = pojo.headPoses;
            if (headPoses != null)
                for (Mat[] headPose : headPoses) {
                    pe.drawAxis(frame, headPose[0], headPose[1]);
                }
            // 记录fps
            long currentTime = System.currentTimeMillis();
            long inv = currentTime - lastTime;
            lastTime = currentTime;
            Imgproc.putText(frame, String.format("FPS: %.2f", 1000.0 / inv),
                    new Point(0, 52),
                    Imgproc.FONT_HERSHEY_COMPLEX,
                    0.5,
                    ColorConst.COLOR_RED);
            // 显示图像
            HighGui.imshow("yolov5", frame);
            HighGui.waitKey(40);
            if (recorder != null) {
                int n = (int) (25 * inv / 1000.0);
                while (n-- > 0)
                    recorder.record(converter.convert(frame));
            }
            return OK;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return CLOSE;
    }

    @Override
    public void open() throws Exception {
        pe = PoseEstimator.getInstance();

//        String rtmp = "rtmp://localhost:8080/testai/";
//        try {
//            recorder = FFmpegUtils.getFFmpegFrameRecorder();
//        } catch (Exception e) {
//            e.printStackTrace();
//        }
    }

    public DrawModule() throws FFmpegFrameRecorder.Exception {
        super();
    }

    @Override
    public void close() throws Exception {
        super.close();
        HighGui.destroyAllWindows();
        if (recorder != null)
            recorder.close();
    }
}
