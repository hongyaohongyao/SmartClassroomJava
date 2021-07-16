package xyz.hyhy.scai.fce;

// 肢体行为=====
//正坐（0.5，0.5） 8.80536914e-01f, 1.19167708e-01f, 2.95387203e-04f, 1.34105589e-08f, 1.11512650e-14f
//抬手（1.5，0.5） 1.06478877e-01f,  7.86778390e-01f,  1.06478877e-01f,  2.63934722e-04f,  1.19826185e-08f
//放松 （3.5，1.5） 0.043f , 0.1306f, 0.2544f, 0.3177f, 0.2544f
//严重低头（4.5，1） 1.91330982e-04f,  6.33601192e-03f,  7.71884322e-02f,  3.45934540e-01f,  5.70349634e-01f
//伸手（3.5，1） 0.0047f, 0.0574f, 0.2571f, 0.4238f, 0.2571f
//趴桌（4.5，0.5） 1.11512650e-14f,  1.34105589e-08f,  2.95387203e-04f,  1.19167708e-01f,  8.80536914e-01f
//面部行为=====
//严重疲劳（4.5，0.5） 1.11512650e-14f,  1.34105589e-08f,  2.95387203e-04f,  1.19167708e-01f,  8.80536914e-01f
//轻微疲劳（3.5，2） 0.0878f, 0.1641f, 0.2388f, 0.2705f, 0.2388f
//非疲劳（1.5，1）0.2571f, 0.4238f, 0.2571f, 0.0574f, 0.0047f
//情绪=====
//遮挡（4.5，1）1.91330982e-04f,  6.33601192e-03f,  7.71884322e-02f,  3.45934540e-01f,  5.70349634e-01f
//积极（0.7，0.7）6.32644117e-01f,  3.42976928e-01f,  2.41576694e-02f,  2.21070048e-04f,  2.62838824e-07f
//中性（1.4，1.4）0.2751f, 0.3374f, 0.2485f, 0.1098f, 0.0291f
//消极（4.2，1.4）0.0121f, 0.0617f, 0.1895f, 0.3496f, 0.3871f
//抬头=====
//遮挡（4.5，1）1.91330982e-04f,  6.33601192e-03f,  7.71884322e-02f,  3.45934540e-01f,  5.70349634e-01f
//低头上课（4.2，1.4）0.0121f, 0.0617f, 0.1895f, 0.3496f, 0.3871f
//低头自习 1.0f,0.0f,0.0f,0.0f,0.0f
//抬头上课 1.0f,0.0f,0.0f,0.0f,0.0f
//抬头自习（4.2，1.4）0.0121f, 0.0617f, 0.1895f, 0.3496f, 0.3871f
//仰头（4.2，1.4）0.0121f, 0.0617f, 0.1895f, 0.3496f, 0.3871f

import lombok.Getter;

@Getter
public class ClassroomConcentrationFce extends SecFce {
    // 行为因素
    public static final int BODY_BEHAVIOR_SEAT = 0;
    public static final int BODY_BEHAVIOR_HIGHER_HAND = 1;
    public static final int BODY_BEHAVIOR_RELAX = 2;
    public static final int BODY_BEHAVIOR_LOWER_HAND = 3;
    public static final int BODY_BEHAVIOR_PASS = 4;
    public static final int BODY_BEHAVIOR_SLEEP = 5;
    public static final int BODY_BEHAVIOR_FACTOR_NUM = 6;
    // 面部行为因素
    public static final int FACIAL_BEHAVIOR_VERY_TIRED = 0;
    public static final int FACIAL_BEHAVIOR_TIRED = 1;
    public static final int FACIAL_BEHAVIOR_NO_TIRED = 2;
    public static final int FACIAL_BEHAVIOR_FACTOR_NUM = 3;
    // 情绪因素
    public static final int EMOTION_HIDE = 0;
    public static final int EMOTION_POSITIVE = 1;
    public static final int EMOTION_NATURE = 2;
    public static final int EMOTION_NEGATIVE = 3;
    public static final int EMOTION_FACTOR_NUM = 4;
    //头部姿态
    public static final int HEAD_POSE_HIDE = 0;
    public static final int HEAD_POSE_LOWER = 1;
    public static final int HEAD_POSE_HIGHER = 2;
    public static final int HEAD_POSE_LOOK_UP = 3;
    public static final int HEAD_POSE_FACTOR_NUM = 4;

    protected float[] bodyBehaviorFactorCount; //行为因素统计
    protected float[] facialBehaviorFactorCount; //面部行为因素统计
    protected float[] emotionFactorCount; //情绪因素统计
    protected float[] headPoseFactorCount; //头部指标统计
    protected int countNum;


    public ClassroomConcentrationFce() {
        super();
        // 肢体行为
        addFactor("body behavior", new float[][]{
                {8.80536914e-01f, 1.19167708e-01f, 2.95387203e-04f, 1.34105589e-08f, 1.11512650e-14f}, //正坐
                {1.06478877e-01f, 7.86778390e-01f, 1.06478877e-01f, 2.63934722e-04f, 1.19826185e-08f}, //抬手
                {0.043f, 0.1306f, 0.2544f, 0.3177f, 0.2544f}, //放松
                {1.91330982e-04f, 6.33601192e-03f, 7.71884322e-02f, 3.45934540e-01f, 5.70349634e-01f}, //严重低头
                {0.0047f, 0.0574f, 0.2571f, 0.4238f, 0.2571f}, //伸手
                {1.11512650e-14f, 1.34105589e-08f, 2.95387203e-04f, 1.19167708e-01f, 8.80536914e-01f}, //趴桌
        });
        addFactor("facial behavior", new float[][]{
                {1.11512650e-14f, 1.34105589e-08f, 2.95387203e-04f, 1.19167708e-01f, 8.80536914e-01f}, //严重疲劳
                {0.0878f, 0.1641f, 0.2388f, 0.2705f, 0.2388f}, //轻度疲劳
                {0.2571f, 0.4238f, 0.2571f, 0.0574f, 0.0047f}, //非疲劳
        });
        addFactor("emotion", new float[][]{
                {1.91330982e-04f, 6.33601192e-03f, 7.71884322e-02f, 3.45934540e-01f, 5.70349634e-01f}, //遮挡
                {6.32644117e-01f, 3.42976928e-01f, 2.41576694e-02f, 2.21070048e-04f, 2.62838824e-07f}, //积极
                {0.2751f, 0.3374f, 0.2485f, 0.1098f, 0.0291f}, //中性
                {0.0121f, 0.0617f, 0.1895f, 0.3496f, 0.3871f}, //消极
        });
        addFactor("head pose", new float[][]{
                {1.91330982e-04f, 6.33601192e-03f, 7.71884322e-02f, 3.45934540e-01f, 5.70349634e-01f}, //遮挡
                {0.0121f, 0.0617f, 0.1895f, 0.3496f, 0.3871f}, //低头上课
                {1.0f, 0.0f, 0.0f, 0.0f, 0.0f}, //低头自习
                {1.0f, 0.0f, 0.0f, 0.0f, 0.0f}, //抬头上课
                {0.0121f, 0.0617f, 0.1895f, 0.3496f, 0.3871f}, //抬头自习
                {0.0121f, 0.0617f, 0.1895f, 0.3496f, 0.3871f}, //仰头
        });
        resetCounts();
    }

    public void resetCounts() {
        countNum = 0;
        bodyBehaviorFactorCount = new float[BODY_BEHAVIOR_FACTOR_NUM];
        facialBehaviorFactorCount = new float[FACIAL_BEHAVIOR_FACTOR_NUM];
        emotionFactorCount = new float[EMOTION_FACTOR_NUM];
        headPoseFactorCount = new float[HEAD_POSE_FACTOR_NUM];
    }

    public void countOneFactors(int bodyBehaviorFactor,
                                int facialBehaviorFactor,
                                int emotionFactor,
                                int headPoseFactor) {
        countNum++;
        bodyBehaviorFactorCount[bodyBehaviorFactor]++;
        facialBehaviorFactorCount[facialBehaviorFactor]++;
        emotionFactorCount[emotionFactor]++;
        headPoseFactorCount[headPoseFactor]++;
    }

    /**
     * 统计结束后，计算一级评价因素权重
     */
    public void calculateFactorWeightsOfClassroomConcentration() {
        setFactorWeights(new float[]{
                infoEntropyOfFactor(bodyBehaviorFactorCount),
                1.0f, // facial behavior
                infoEntropyOfFactor(emotionFactorCount),
                infoEntropyOfFactor(headPoseFactorCount)
        });
    }

}
