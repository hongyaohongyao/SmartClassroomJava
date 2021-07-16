package xyz.hyhy.scai.exemodules;

import ai.djl.modality.Classifications;
import ai.djl.modality.cv.output.Joints;
import org.opencv.core.Mat;
import xyz.hyhy.scai.constant.CoreConst;
import xyz.hyhy.scai.core.commons.DataDict;
import xyz.hyhy.scai.core.modules.ExeModule;
import xyz.hyhy.scai.ml.ClassroomActionClassifier;
import xyz.hyhy.scai.pojo.ClassActionPOJO;
import xyz.hyhy.scai.utils.PoseEstimator;

import java.util.Arrays;
import java.util.List;

public class ClassActionModule extends ExeModule {
    private ClassroomActionClassifier classifier;
    private PoseEstimator pe;

    @Override
    protected int processData(DataDict data, DataDict globalData) throws Exception {
        try {
            ClassActionPOJO pojo = (ClassActionPOJO) data.get("pojo");
            List<Joints> joints = pojo.joints;
            // 动作分类
            List<Classifications> classifications = classifier.classify(joints);
            pojo.classifications = classifications;
            // 头部姿态估计
            int startIndex, endIndex;
            if (CoreConst.CLASS_ACTION_MODEL_TYPE == ClassroomActionClassifier.TYPE_COCO) {
                startIndex = 18;
                endIndex = 86;
            } else if (CoreConst.CLASS_ACTION_MODEL_TYPE == ClassroomActionClassifier.TYPE_HALPE) {
                startIndex = 26;
                endIndex = 94;
            } else {
                startIndex = 18;
                endIndex = 86;
            }
            Mat[][] headPoses = new Mat[joints.size()][];
            for (int i = 0; i < headPoses.length; i++) {
                headPoses[i] = pe.solvePose(joints.get(i).getJoints().subList(startIndex, endIndex));
            }
            pojo.headPoses = Arrays.asList(headPoses);
            return OK;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return CLOSE;
    }

    @Override
    public void open() throws Exception {
        classifier = new ClassroomActionClassifier(CoreConst.CLASS_ACTION_MODEL_TYPE);
        pe = PoseEstimator.getInstance();
    }

    @Override
    public void close() throws Exception {
        super.close();
        if (classifier != null)
            classifier.close();
    }
}
