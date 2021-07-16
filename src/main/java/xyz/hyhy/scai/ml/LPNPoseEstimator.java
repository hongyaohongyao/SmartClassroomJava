package xyz.hyhy.scai.ml;

import ai.djl.MalformedModelException;
import ai.djl.modality.cv.output.Joints;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;
import org.opencv.core.Mat;
import xyz.hyhy.scai.translator.BasePairTranslator;
import xyz.hyhy.scai.translator.LPNTranslator;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class LPNPoseEstimator extends ParallelPoseEstimator {


    public LPNPoseEstimator() throws Exception {
        super();
    }


    public LPNPoseEstimator(int parallelism, int minLenToStartWork, boolean gpu) throws MalformedModelException, ModelNotFoundException, IOException {
        super(parallelism, minLenToStartWork, gpu);
    }

    @Override
    public BasePairTranslator<Mat, Rectangle, Joints> getPoseTranslator() {
        return LPNTranslator.builder().build();
    }

    //    @Override
//    public String getPoseModelName() {
//        return "lpn_50_256x192.onnx";
//    }
    @Override
    public String getPoseModelName() {
        return "lpn_50_256x192.onnx";
    }

    @Override
    public String getLibrary() {
        return "OnnxRuntime";
    }

    @Override
    public void singleWork(PoseTask poseTask) throws TranslateException {
        int len = poseTask.end - poseTask.start + 1;
        List<Pair<Mat, Rectangle>> inputObj = new ArrayList<>(len);
        for (int i = poseTask.start; i <= poseTask.end; i++) {
            inputObj.add(new Pair<>(poseTask.frame, poseTask.bboxes.get(i)));
        }
        List<Joints> jts = posePredictor.batchPredict(inputObj);
        System.arraycopy(jts.toArray(), 0,
                poseTask.outputs, poseTask.start, len);
    }
}
