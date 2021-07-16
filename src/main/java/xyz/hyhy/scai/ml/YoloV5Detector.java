package xyz.hyhy.scai.ml;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.translator.YoloV5Translator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import lombok.Getter;
import lombok.Setter;
import xyz.hyhy.scai.AlphaPose;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

@Getter
@Setter
public class YoloV5Detector implements AutoCloseable {
    private Translator<Image, DetectedObjects> translator;
    private Criteria<Image, DetectedObjects> criteria;
    private ZooModel<Image, DetectedObjects> model;
    Predictor<Image, DetectedObjects> predictor;
//    public static final String MODEL_NAME = "yolov5s.onnx";
//    public static final String LIBRARY = "OnnxRuntime";

    public static final String MODEL_NAME = "yolov5s.torchscript.pt";
    public static final String LIBRARY = "PyTorch";

    public YoloV5Detector() throws MalformedModelException, ModelNotFoundException, IOException {
        translator = YoloV5Translator.builder().optSynsetArtifactName("coco.names").build();
        criteria = Criteria.builder()
                .setTypes(Image.class, DetectedObjects.class)
                .optDevice(Device.gpu())
                .optModelUrls(AlphaPose.class.getResource("/yolov5").getPath())
                .optModelName(MODEL_NAME)
                .optTranslator(translator)
                .optEngine(LIBRARY)
                .build();
        model = ModelZoo.loadModel(criteria);
        predictor = model.newPredictor();
    }

    public DetectedObjects detect(Image img) throws TranslateException {
        return predictor.predict(img);
    }

    public List<DetectedObjects> detect(List<Image> imgs) throws TranslateException {
        List<DetectedObjects> results = new ArrayList<>(imgs.size());
        try (Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
            for (Image img : imgs) {
                results.add(predictor.predict(img));
            }
        }
        return results;
    }

    @Override
    public void close() throws Exception {
        if (model != null)
            model.close();
        if (predictor != null)
            predictor.close();
    }
}
