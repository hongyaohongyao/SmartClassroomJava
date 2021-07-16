package xyz.hyhy.scai.translator;

import ai.djl.modality.cv.output.Joints;
import ai.djl.modality.cv.output.Joints.Joint;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import xyz.hyhy.scai.ml.ClassroomActionClassifier;
import xyz.hyhy.scai.utils.NumpyUtils;

import java.util.Arrays;
import java.util.List;

public class ClassroomActionTranslator extends JointsClassificationTranslator {
    public int[] usedKeypoints;

    public static final int[] USED_KEYPOINTS_HALPE = new int[]{
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18, 19
    };
    public static final int[] USED_KEYPOINTS_COCO = new int[]{
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    };
    public static final float[] CLASS_MASK_HALPE = new float[]{
            1, 1, 1, 1, 1,
            1, 1,
            1, 1, 1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1
    };
    public static final float[] CLASS_MASK_COCO = new float[]{
            1, 1, 1, 1, 1,
            1, 1,
            1, 1, 1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1
    };

    protected NDArray classWeights;

    public ClassroomActionTranslator(int type) {
        switch (type) {
            default: {
                usedKeypoints = USED_KEYPOINTS_COCO;
                classWeights = NumpyUtils.ndManager.create(CLASS_MASK_COCO);
            }
            case ClassroomActionClassifier.TYPE_HALPE: {
                usedKeypoints = USED_KEYPOINTS_HALPE;
                classWeights = NumpyUtils.ndManager.create(CLASS_MASK_HALPE);
            }
            break;
            case ClassroomActionClassifier.TYPE_COCO: {
                usedKeypoints = USED_KEYPOINTS_COCO;
                classWeights = NumpyUtils.ndManager.create(CLASS_MASK_COCO);
            }
            break;
        }

        classNames = Arrays.asList("seat", "write", "stretch", "hand_up_R", "hand_up_L",
                "hand_up_highly_R", "hand_up_highly_L",
                "relax", "hand_up", "pass_R", "pass_L", "pass2_R", "pass2_L",
                "turn_round_R", "turn_round_L", "turn_head_R", "turn_head_L",
                "sleep", "lower_head");
    }

    private static final int[] axes0 = new int[]{0};

    @Override
    public NDArray joints2floats(NDManager ndManager, Joints joints) {
        List<Joint> jts = joints.getJoints();
        float[][] inputArr = new float[usedKeypoints.length][2];
        int pt1 = 0;
        for (int i = 0; i < usedKeypoints.length; i++) {
            while (pt1 < jts.size() && pt1 != usedKeypoints[i])
                pt1++;
            if (pt1 < jts.size()) {
                Joint jt = jts.get(pt1);
                inputArr[i][0] = (float) jt.getX();
                inputArr[i][1] = (float) jt.getY();
            } else
                break;
        }
        NDArray input = ndManager.create(inputArr);
        // 标准化
        NDArray xyMin = input.min(axes0);
        NDArray width = input.max(axes0).sub(xyMin);
        return input.sub(xyMin).mul(2).div(width).sub(1).flatten();
    }

    private static final int LOWER_HEAD_INDEX = 18;
    private static final NDIndex LOWER_HEAD_NDINDEX = new NDIndex(LOWER_HEAD_INDEX);

    @Override
    public NDArray postprocess(NDManager ndManager, NDArray output) {
        if (output.getFloat(LOWER_HEAD_INDEX) < 0.9) //令低头必须
            output.set(LOWER_HEAD_NDINDEX, 0);
        return output.mul(classWeights);
    }
}
