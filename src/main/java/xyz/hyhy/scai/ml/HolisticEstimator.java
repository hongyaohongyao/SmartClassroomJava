package xyz.hyhy.scai.ml;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.output.Joints;
import ai.djl.modality.cv.output.Joints.Joint;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;
import org.opencv.core.Mat;
import xyz.hyhy.scai.AlphaPose;
import xyz.hyhy.scai.translator.BasePairTranslator;
import xyz.hyhy.scai.translator.FaceDetectionTranslator;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class HolisticEstimator extends LPNPoseEstimator {
    protected BasePairTranslator<Mat, Rectangle, Joints> faceTranslator;
    protected Criteria<Pair<Mat, Rectangle>, Joints> faceCriteria;
    protected ZooModel<Pair<Mat, Rectangle>, Joints> faceModel;
    protected Predictor<Pair<Mat, Rectangle>, Joints> facePredictor;

    public HolisticEstimator() throws Exception {
        this(DEFAULT_PARALLELISM, 3, false);
    }

    public HolisticEstimator(int parallelism, int minLenToStartWork, boolean gpu) throws MalformedModelException, ModelNotFoundException, IOException {
        super(parallelism, minLenToStartWork, gpu);

        faceTranslator = FaceDetectionTranslator.builder().build();
        faceCriteria = Criteria.builder()
                .setTypes(poseTranslator.getPairClass(), Joints.class)
                .optDevice(gpu ? Device.gpu() : Device.cpu())
                .optModelUrls(AlphaPose.class.getResource("/face").getPath())
                .optModelName("landmark_detection_56_se_external.onnx")
                .optTranslator(faceTranslator)
                .optEngine("OnnxRuntime")
                .build();
        faceModel = ModelZoo.loadModel(faceCriteria);
        facePredictor = faceModel.newPredictor();
    }

    @Override
    public void singleWork(PoseTask poseTask) throws TranslateException {
        int len = poseTask.end - poseTask.start + 1;
        List<Pair<Mat, Rectangle>> inputObj = new ArrayList<>(len);
        for (int i = poseTask.start; i <= poseTask.end; i++) {
            inputObj.add(new Pair<>(poseTask.frame, poseTask.bboxes.get(i)));
        }
        // LPN姿态估计
        List<Joints> jtsList = posePredictor.batchPredict(inputObj);
        System.arraycopy(jtsList.toArray(), 0,
                poseTask.outputs, poseTask.start, len);
        // 人脸对齐
        for (int i = poseTask.start; i <= poseTask.end; i++) {
            Joints jts = poseTask.outputs[i];
            List<Joint> jointList = jts.getJoints();
            // 获取人脸框
            poseTask.outputs[i] = jts;
            double xMin = Double.MAX_VALUE;
            double xMax = Double.MIN_VALUE;
            double yMin = Double.MAX_VALUE;
            double yMax = Double.MIN_VALUE;
            for (int j = 0; j < 5; j++) {
                Joint jt = jointList.get(j);
                double crood = jt.getX();
                xMin = Math.min(xMin, crood);
                xMax = Math.max(xMax, crood);
                crood = jt.getY();
                yMin = Math.min(yMin, crood);
                yMax = Math.max(yMax, crood);
            }
            Joints faceJoints = facePredictor.predict(new Pair<>(poseTask.frame,
                    new Rectangle(xMin, yMin, (xMax - xMin), (yMax - yMin))));
            jointList.addAll(faceJoints.getJoints());
        }
    }
}
