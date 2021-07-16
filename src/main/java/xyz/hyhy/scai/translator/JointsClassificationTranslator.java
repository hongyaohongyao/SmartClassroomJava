package xyz.hyhy.scai.translator;

import ai.djl.modality.Classifications;
import ai.djl.modality.cv.output.Joints;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import xyz.hyhy.scai.utils.NumpyUtils;

import java.util.List;

public abstract class JointsClassificationTranslator implements Translator<Joints, Classifications> {
    protected Batchifier batchifier;
    protected List<String> classNames;

    public JointsClassificationTranslator() {
        batchifier = Batchifier.STACK;
    }

    @Override
    public Batchifier getBatchifier() {
        return batchifier;
    }

    @Override
    public Classifications processOutput(TranslatorContext ctx, NDList list) throws Exception {
        return new Classifications(classNames, postprocess(NumpyUtils.ndManager, list.singletonOrThrow()));
    }

    public abstract NDArray joints2floats(NDManager ndManager, Joints input);

    public abstract NDArray postprocess(NDManager ndManager, NDArray output);


    @Override
    public NDList processInput(TranslatorContext ctx, Joints joints) throws Exception {
        return new NDList(joints2floats(NumpyUtils.ndManager, joints));
    }
}
