package xyz.hyhy.scai.constant;

import xyz.hyhy.scai.ml.ClassroomActionClassifier;

public interface CoreConst {

    int DEFAULT_MODULE_QUEUE_SIZE = 10;
    double DETECTION_PROB_THRESHOLD = 0.39;

    long BALANCE_CEILING_VALUE = 1000;

    int CLASS_ACTION_MODEL_TYPE = ClassroomActionClassifier.TYPE_HALPE;

    int IMG_WIDTH = 640;
    int IMG_HEIGHT = 480;
}
