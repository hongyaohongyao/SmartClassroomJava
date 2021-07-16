package xyz.hyhy.scai.fce;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import xyz.hyhy.scai.utils.NumpyUtils;

import java.util.ArrayList;
import java.util.List;

/**
 * 二级评价因子的模糊综合分析
 */
public class SecFce {
    protected static final NDManager ndManager = NumpyUtils.ndManager;
    protected NDArray values;
    protected List<NDArray> relationMats;
    protected List<String> factorNames;
    protected NDArray factorWeights;

    public SecFce() {
        values = ndManager.create(new float[]{5, 4, 3, 2, 1});
        relationMats = new ArrayList<>();
        factorNames = new ArrayList<>();
        factorWeights = null;
    }


    public void setFactorWeights(float[] weights, boolean softmax) {
        if (softmax)
            factorWeights = softmax(ndManager.create(weights));
        else
            setFactorWeights(weights);
    }

    public void setFactorWeights(float[] weights) {
        factorWeights = ndManager.create(weights);
    }

    public float calculateResult(float[] secResults) {
        return factorWeights.matMul(ndManager.create(secResults)).getFloat();
    }

    public float[] calculateSecResults(List<float[]> secWeights) {
        float[][] resultVectors = new float[secWeights.size()][relationMats.size()];
        for (int i = 0; i < secWeights.size(); i++) {
            resultVectors[i] = ndManager
                    .create(secWeights.get(i))
                    .matMul(relationMats.get(i)).toFloatArray();
        }
        return ndManager.create(resultVectors).matMul(values).toFloatArray();
    }

    public static float[] infoEntropyOfFactors(List<float[]> counts) {
        float[] results = new float[counts.size()];
        for (int i = 0; i < results.length; i++) {
            results[i] = infoEntropyOfFactor(counts.get(i));
        }
        return results;
    }

    public static float infoEntropyOfFactor(float[] count) {
        final float delta = 1e-8f; // 添加一个微小值可以防止负无限大(np.log(0))的发生。
        NDArray count0 = ndManager.create(count);
        count0 = count0.div(count0.sum()); // 归一化
        return count0.add(delta)
                .log()
                .matMul(count0)
                .sub(Math.log(1.0f / count.length))
                .div(Math.log(count.length))
                .maximum(0)
                .minimum(1)
                .toType(DataType.FLOAT32, false)
                .getFloat();
    }

    public static NDArray softmax(NDArray x) {
        return x.softmax(0);
    }

    public void addFactor(String name, float[][] relationMat) {
        relationMats.add(ndManager.create(relationMat));
        factorNames.add(name);
    }

    public static void main(String[] args) {
//        int times = 1;
//        long startTime = System.currentTimeMillis();
//        for (int i = 0; i < times; i++) {
//            System.out.println(softmax(ndManager.create(new float[]{1f, 0.25f, 0.25f, 0.25f})));
//        }
//        System.out.println((System.currentTimeMillis() - startTime) / (1000.0 * times));

    }

}
