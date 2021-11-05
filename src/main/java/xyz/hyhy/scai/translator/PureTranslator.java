package xyz.hyhy.scai.translator;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;


public class PureTranslator implements Translator<Shape, Void> {
    private boolean half;


    public PureTranslator(boolean half) {
        this.half = half;
    }

    public PureTranslator() {
        this(false);
    }

    @Override
    public Batchifier getBatchifier() {
        return Batchifier.STACK;
    }

    @Override
    public Void processOutput(TranslatorContext ctx, NDList list) throws Exception {
        return null;
    }

    @Override
    public NDList processInput(TranslatorContext ctx, Shape input) throws Exception {
        NDArray array = ctx.getNDManager().randomNormal(input, half ? DataType.FLOAT16 : DataType.FLOAT32);
        return new NDList(array);
    }
}
