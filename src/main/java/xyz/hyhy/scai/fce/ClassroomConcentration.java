package xyz.hyhy.scai.fce;

import lombok.Getter;
import lombok.Setter;

@Setter
@Getter
public class ClassroomConcentration extends ClassroomConcentrationFce {
    protected float[] bodyBehaviorCount; //行为具体指标统计
    protected float[] facialBehaviorCount; //面部行为具体指标统计
    protected float[] emotionCount; //情绪具体指标统计
    protected float headPoseThreshold = 30;


    public ClassroomConcentration() {
        super();
    }

    public void startNewCount() {
        resetCounts();
    }

    /**
     * 统计一名
     *
     * @param bodyBehavior
     * @param facialBehavior
     * @param emotion
     * @param anglePitch
     * @param hideFace
     */
    public void countOne(int bodyBehavior,
                         int facialBehavior,
                         int emotion,
                         double anglePitch,
                         boolean hideFace) {
        int bodyBehaviorFactor;
        int facialBehaviorFactor;
        int emotionFactor;
        int headPoseFactor;
        //TODO 肢体行为
        bodyBehaviorFactor = 0;
        //TODO 面部行为（疲劳检测）
        facialBehaviorFactor = 0;
        // 情绪和抬头检查
        if (hideFace) {
            emotionFactor = EMOTION_HIDE;
            headPoseFactor = HEAD_POSE_HIDE;
        } else {
            //TODO 情绪==》积极性
            emotionFactor = 0;
            //抬头角度
            if (anglePitch >= headPoseThreshold) {
                headPoseFactor = HEAD_POSE_LOOK_UP;
            } else if (anglePitch > -headPoseThreshold) {
                headPoseFactor = HEAD_POSE_HIGHER;
            } else {
                headPoseFactor = HEAD_POSE_LOWER;
            }

        }
        countOneFactors(bodyBehaviorFactor,
                facialBehaviorFactor,
                emotionFactor,
                headPoseFactor);
    }

    /**
     * 统计之后，开始计算之前，需要重置各项评价因素的权重
     */
    public void afterCountAndBeforeCalculate() {
        calculateFactorWeightsOfClassroomConcentration();
    }

    /**
     * 完成一个学生的各项统计指标计算
     */
    public void completeOneCalculate() {

    }


}
