package xyz.hyhy.scai.translator;/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.Joints;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.translator.BaseImageTranslator;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.Pair;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import xyz.hyhy.scai.utils.CVUtils;
import xyz.hyhy.scai.utils.ImageUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * A {@link BaseImageTranslator} that post-process the {@link NDArray} into human {@link Joints}.
 */
public class SPPETranslator2 extends BasePairTranslator<Mat, Rectangle, Joints> {
    private float threshold;

    /**
     * Creates the Pose Estimation translator from the given builder.
     *
     * @param builder the builder for the translator
     */
    public SPPETranslator2(Builder builder) {
        super(builder);
        this.threshold = builder.threshold;
    }

    private static long[] tilePram = new long[]{1, 1, 2};
    private static int[] axis2 = new int[]{2};
    private static int[] axis2n3 = new int[]{2, 3};
    private static int[] axis2n4 = new int[]{2, 4};
    private static int[] axis3n4 = new int[]{3, 4};


    private static NDArray integralOp(NDArray hm, NDManager ndManager) {
        Shape hmShape = hm.getShape();
        NDArray arr = ndManager
                .arange(hmShape.get(hmShape
                        .dimension() - 1)).toType(DataType.FLOAT32, false);
        return hm.mul(arr);
    }

    private static final NDIndex VIS_N_PRE_INDEX = new NDIndex(":,3:");
    private static final NDIndex TRANSED_INDEX = new NDIndex(":2");

    /**
     * {@inheritDoc}
     */
    @Override
    public Joints processOutput(TranslatorContext ctx, NDList list) {
        System.out.println(list);
        System.out.println(list.get(1));
        NDArray pred = list.get(0);
        pred = pred.reshape(new Shape(-1, 5));
        int numJoints = (int) pred.getShape().get(0);
//        int width = (int) list.get(3).getShape().get(1);
//        int height = (int) list.get(3).getShape().get(0);
//        int width = (int) list.get(2).getShape().get(1);
//        int height = (int) list.get(2).getShape().get(0);
        int width = 256, height = 256;
        pred.set(VIS_N_PRE_INDEX, Activation.sigmoid(pred.get(VIS_N_PRE_INDEX)));
//        System.out.println(pred.get(1));
        Rectangle bbox = (Rectangle) ctx.getAttachment("cropped_bbox");
        double x = bbox.getX();
        double y = bbox.getY();
        double w = bbox.getWidth();
        double h = bbox.getHeight();
        double centerX = x + 0.5 * w, centerY = y + 0.5 * h;
        double scaleX = w;

        NDManager ndManager = ctx.getNDManager();
        Mat trans = CVUtils.getAffineTransform(centerX, centerY, width, height, scaleX, true);
        NDArray ndTrans = CVUtils.transMat2NDArray(trans, ndManager);

        List<Joints.Joint> joints = new ArrayList<>(numJoints);
        for (int i = 0; i < numJoints; ++i) {
            NDArray xy = ndManager
                    .create(new float[]{pred.getFloat(i, 0), pred.getFloat(i, 1), 1})
                    .transpose();
            xy = ndTrans.matMul(xy)
                    .get(TRANSED_INDEX);

            joints.add(
                    new Joints.Joint(
                            xy.getFloat(0),
                            xy.getFloat(1),
                            Math.min(pred.getFloat(i, 3), pred.getFloat(i, 4))));
        }
//        System.out.println(joints);
        return new Joints(joints);
    }

    /**
     * Creates a builder to build a {@code SPPETranslator}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Creates a builder to build a {@code SPPETranslator} with specified arguments.
     *
     * @param arguments arguments to specify builder options
     * @return a new builder
     */
    public static Builder builder(Map<String, ?> arguments) {
        Builder builder = new Builder();
        builder.configPreProcess(arguments);
        builder.configPostProcess(arguments);

        return builder;
    }

    private static final NDIndex CHANNEL0 = new NDIndex(":,:,0");
    private static final NDIndex CHANNEL1 = new NDIndex(":,:,1");
    private static final NDIndex CHANNEL2 = new NDIndex(":,:,2");

    @Override
    public NDList processInput(TranslatorContext ctx, Pair<Mat, Rectangle> input) throws Exception {
        Mat frame = input.getKey().clone();
        Rectangle bbox = input.getValue();
        int x = (int) Math.max(0, bbox.getX());
        int y = (int) Math.max(0, bbox.getY());
        int w = Math.min(frame.width(), (int) (x + bbox.getWidth())) - x;
        int h = Math.min(frame.height(), (int) (y + bbox.getHeight())) - y;
        Rectangle croppedBBox = CVUtils.scale(frame, x, y, w, h, 256, 256);
        ctx.setAttachment("cropped_bbox", croppedBBox);
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BGR2RGB);
        NDArray array = ImageUtils.mat2Image(frame).toNDArray(ctx.getNDManager(), Image.Flag.COLOR);
        array = array.toType(DataType.FLOAT32, false);
        array.set(CHANNEL0, array.get(CHANNEL0).div(255));
        array.set(CHANNEL1, array.get(CHANNEL1).div(255));
        array.set(CHANNEL2, array.get(CHANNEL2).div(255));

        return new NDList(array);
    }

    /**
     * The builder for Pose Estimation translator.
     */
    public static class Builder extends BaseBuilder<Builder> {
        //Pose
        float[] mean = new float[]{0.406f, 0.457f, 0.480f};
        float[] std = new float[]{1, 1, 1};

        float threshold = 0.2f;

        Builder() {
        }

        /**
         * Sets the threshold for prediction accuracy.
         *
         * <p>Predictions below the threshold will be dropped.
         *
         * @param threshold the threshold for prediction accuracy
         * @return the builder
         */
        public Builder optThreshold(float threshold) {
            this.threshold = threshold;
            return self();
        }

        /**
         * {@inheritDoc}
         */
        @Override
        protected Builder self() {
            return this;
        }

        /**
         * {@inheritDoc}
         */
        @Override
        protected void configPostProcess(Map<String, ?> arguments) {
            threshold = getFloatValue(arguments, "threshold", 0.2f);
        }

        @Override
        protected void configPreProcess(Map<String, ?> arguments) {
            super.configPreProcess(arguments);
        }

        /**
         * Builds the translator.
         *
         * @return the new translator
         */
        public SPPETranslator2 build() {
//            if (pipeline == null) {
//                addTransform(new ToTensor());
//                addTransform(new Normalize(mean, std));
//            }
//            validate();
            return new SPPETranslator2(this);
        }
    }
}
