package xyz.hyhy.scai.ml;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.output.Joints;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import xyz.hyhy.scai.AlphaPose;
import xyz.hyhy.scai.translator.ClassroomActionTranslator;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class ClassroomActionClassifier implements AutoCloseable {
    private ClassroomActionTranslator translator;
    private Criteria<Joints, Classifications> criteria;
    private ZooModel<Joints, Classifications> model;
    public static final int TYPE_HALPE = 0;
    public static final int TYPE_COCO = 1;

    public static final String MODEL_NAME_HALPE = "classroom_action_lr_front_v2_sm.onnx";
    public static final String MODEL_NAME_COCO = "classroom_action_lr_front_v2_coco.onnx";

    public ClassroomActionClassifier(int type) throws MalformedModelException, ModelNotFoundException, IOException {
        translator = new ClassroomActionTranslator(type);
        switch (type) {
            default: {
                criteria = Criteria.builder()
                        .setTypes(Joints.class, Classifications.class)
                        .optDevice(Device.cpu())
                        .optModelUrls(AlphaPose.class.getResource("/action").getPath())
                        .optModelName(MODEL_NAME_COCO)
                        .optEngine("OnnxRuntime")
                        .optTranslator(translator)
                        .build();
            }
            case TYPE_HALPE: {
                criteria = Criteria.builder()
                        .setTypes(Joints.class, Classifications.class)
                        .optDevice(Device.cpu())
                        .optModelUrls(AlphaPose.class.getResource("/action").getPath())
                        .optModelName(MODEL_NAME_HALPE)
                        .optTranslator(translator)
                        .optEngine("OnnxRuntime")
                        .build();
            }
            break;
            case TYPE_COCO: {
                criteria = Criteria.builder()
                        .setTypes(Joints.class, Classifications.class)
                        .optDevice(Device.cpu())
                        .optModelUrls(AlphaPose.class.getResource("/action").getPath())
                        .optModelName(MODEL_NAME_COCO)
                        .optTranslator(translator)
                        .optEngine("OnnxRuntime")
                        .build();
            }
            break;
        }
        model = ModelZoo.loadModel(criteria);
    }

    public Classifications classify(Joints joints) throws TranslateException {
        try (Predictor<Joints, Classifications> predictor = model.newPredictor()) {
            return predictor.predict(joints);
        }
    }

    public List<Classifications> classify(List<Joints> jointsList) throws TranslateException {
        List<Classifications> results = new ArrayList<>(jointsList.size());
        try (Predictor<Joints, Classifications> predictor = model.newPredictor()) {
            for (Joints joints : jointsList) {
                results.add(predictor.predict(joints));
            }
        }
        return results;
    }

    @Override
    public void close() throws Exception {
        if (model != null)
            model.close();
    }
}
