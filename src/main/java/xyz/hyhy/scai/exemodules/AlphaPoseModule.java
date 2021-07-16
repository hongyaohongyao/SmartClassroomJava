package xyz.hyhy.scai.exemodules;

import ai.djl.MalformedModelException;
import ai.djl.modality.cv.output.DetectedObjects.DetectedObject;
import ai.djl.modality.cv.output.Joints;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.repository.zoo.ModelNotFoundException;
import org.opencv.core.Mat;
import xyz.hyhy.scai.core.commons.DataDict;
import xyz.hyhy.scai.core.modules.ExeModule;
import xyz.hyhy.scai.ml.AlphaPoseEstimator;
import xyz.hyhy.scai.ml.ParallelPoseEstimator;
import xyz.hyhy.scai.pojo.ClassActionPOJO;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class AlphaPoseModule extends ExeModule {
    private ParallelPoseEstimator ppe;


    @Override
    public void open() throws MalformedModelException, ModelNotFoundException, IOException {
        int par = 3, mlts = 10;
        boolean gpu = true;
        ppe = new AlphaPoseEstimator(par, mlts, gpu);
    }

    @Override
    protected int processData(DataDict data, DataDict globalData) {
        try {
            ClassActionPOJO pojo = (ClassActionPOJO) data.get("pojo");
            Mat frame = pojo.frame;
            List<DetectedObject> detectedObjects = pojo.detectedObjects;
            Rectangle[] rectangles = new Rectangle[detectedObjects.size()];
            for (int i = 0; i < rectangles.length; i++) {
                rectangles[i] = detectedObjects.get(i).getBoundingBox().getBounds();
            }
            // 姿态评估
            List<Joints> joints = ppe.infer(frame, Arrays.asList(rectangles));
            pojo.joints = joints;
            return OK;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return CLOSE;
    }

    @Override
    public void close() throws Exception {
        super.close();
        if (ppe != null)
            ppe.close();
    }
}
