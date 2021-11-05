package xyz.hyhy.scai.translator;

import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.BaseImageTranslator;
import ai.djl.ndarray.NDList;
import ai.djl.translate.Pipeline;
import ai.djl.translate.TranslatorContext;

import java.util.HashMap;


public class PureImageTranslator extends BaseImageTranslator<Void> {
    /**
     * Constructs an ImageTranslator with the provided builder.
     *
     * @param builder the data to build with
     */
    public PureImageTranslator(Builder builder) {
        super(builder);
    }


    @Override
    public Void processOutput(TranslatorContext translatorContext, NDList ndList) throws Exception {
        ndList.get(0);
        return null;
    }

    public static class Builder extends BaseBuilder<Builder> {

        @Override
        protected Builder self() {
            return this;
        }

        public PureImageTranslator build(boolean toTensor) {
            pipeline = toTensor ? new Pipeline(new ToTensor()) : new Pipeline();
            validate();
            return new PureImageTranslator(this);
        }

        public PureImageTranslator build(int height, int width) {
            configPreProcess(new HashMap<String, Object>() {{
                this.put("width", height);
                this.put("height", width);
                this.put("normalize", "true");
            }});
            validate();
            return new PureImageTranslator(this);
        }

        public PureImageTranslator build() {
            return build(640, 640);
        }
    }
}
