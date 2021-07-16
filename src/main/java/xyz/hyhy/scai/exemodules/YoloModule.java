package xyz.hyhy.scai.exemodules;

import ai.djl.MalformedModelException;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.DetectedObjects.DetectedObject;
import ai.djl.repository.zoo.ModelNotFoundException;
import org.opencv.core.Mat;
import xyz.hyhy.scai.constant.CoreConst;
import xyz.hyhy.scai.core.commons.DataDict;
import xyz.hyhy.scai.core.modules.ExeModule;
import xyz.hyhy.scai.ml.YoloV5Detector;
import xyz.hyhy.scai.pojo.ClassActionPOJO;
import xyz.hyhy.scai.utils.ImageUtils;

import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;

public class YoloModule extends ExeModule {
    private YoloV5Detector detector;


    @Override
    public void open() throws MalformedModelException, ModelNotFoundException, IOException {
        detector = new YoloV5Detector();
    }

    @Override
    protected int processData(DataDict data, DataDict globalData) {
        try {
            ClassActionPOJO pojo = (ClassActionPOJO) data.get("pojo");
            Mat frame = pojo.frame;
            Image img = ImageUtils.mat2Image(frame);
            List<DetectedObject> results = detector.detect(img).items();

            pojo.detectedObjects = results
                    .stream()
                    .filter(YoloModule::filter)
                    .collect(Collectors.toList());
            return OK;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return CLOSE;
    }

    private static boolean filter(DetectedObject detectedObject) {
        return "person".equals(detectedObject.getClassName()) &&
                detectedObject.getProbability() > CoreConst.DETECTION_PROB_THRESHOLD;
    }

    @Override
    public void close() throws Exception {
        super.close();
        if (detector != null)
            detector.close();
    }
}
