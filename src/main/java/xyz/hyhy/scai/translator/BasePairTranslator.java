package xyz.hyhy.scai.translator;

import ai.djl.modality.cv.Image;
import ai.djl.ndarray.NDList;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Pipeline;
import ai.djl.translate.Transform;
import ai.djl.translate.Translator;
import ai.djl.util.Pair;

import java.util.Map;

public abstract class BasePairTranslator<I, P, O> implements Translator<Pair<I, P>, O> {

    protected Pipeline pipeline;
    protected Batchifier batchifier;
    private Class<Pair<I, P>> pairClass = (Class<Pair<I, P>>) (new Pair<I, P>(null, null).getClass());

    /**
     * Constructs an PairTranslator with the provided builder.
     *
     * @param builder the data to build with
     */
    public BasePairTranslator(BaseBuilder<?> builder) {
        pipeline = builder.pipeline;
        batchifier = builder.batchifier;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Batchifier getBatchifier() {
        return batchifier;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Pipeline getPipeline() {
        return pipeline;
    }

    /**
     * Processes the {@link Image} input and converts it to NDList.
     *
     * @return a {@link NDList}
     */

    protected static String getStringValue(Map<String, ?> arguments, String key, String def) {
        Object value = arguments.get(key);
        if (value == null) {
            return def;
        }
        return value.toString();
    }

    protected static int getIntValue(Map<String, ?> arguments, String key, int def) {
        Object value = arguments.get(key);
        if (value == null) {
            return def;
        }
        return (int) Double.parseDouble(value.toString());
    }

    protected static float getFloatValue(Map<String, ?> arguments, String key, float def) {
        Object value = arguments.get(key);
        if (value == null) {
            return def;
        }
        return (float) Double.parseDouble(value.toString());
    }

    protected static boolean getBooleanValue(Map<String, ?> arguments, String key, boolean def) {
        Object value = arguments.get(key);
        if (value == null) {
            return def;
        }
        return Boolean.parseBoolean(value.toString());
    }

    public Class<Pair<I, P>> getPairClass() {
        return this.pairClass;
    }

    /**
     * A builder to extend for all classes extending the {@link BasePairTranslator}.
     *
     * @param <T> the concrete builder type
     */
    @SuppressWarnings("rawtypes")
    public abstract static class BaseBuilder<T extends BaseBuilder> {

        protected Image.Flag flag = Image.Flag.COLOR;
        protected Pipeline pipeline;
        protected Batchifier batchifier = Batchifier.STACK;

        /**
         * Sets the optional {@link ai.djl.modality.cv.Image.Flag} (default is {@link
         * Image.Flag#COLOR}).
         *
         * @param flag the color mode for the images
         * @return this builder
         */
        public T optFlag(Image.Flag flag) {
            this.flag = flag;
            return self();
        }

        /**
         * Sets the {@link Pipeline} to use for pre-processing the image.
         *
         * @param pipeline the pre-processing pipeline
         * @return this builder
         */
        public T setPipeline(Pipeline pipeline) {
            this.pipeline = pipeline;
            return self();
        }

        /**
         * Adds the {@link Transform} to the {@link Pipeline} use for pre-processing the image.
         *
         * @param transform the {@link Transform} to be added
         * @return this builder
         */
        public T addTransform(Transform transform) {
            if (pipeline == null) {
                pipeline = new Pipeline();
            }
            pipeline.add(transform);
            return self();
        }

        /**
         * Sets the {@link Batchifier} for the {@link Translator}.
         *
         * @param batchifier the {@link Batchifier} to be set
         * @return this builder
         */
        public T optBatchifier(Batchifier batchifier) {
            this.batchifier = batchifier;
            return self();
        }

        protected abstract T self();

        protected void validate() {
            if (pipeline == null) {
                throw new IllegalArgumentException("pipeline is required.");
            }
        }

        protected void configPreProcess(Map<String, ?> arguments) {

        }

        protected void configPostProcess(Map<String, ?> arguments) {
        }
    }
}
