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

import ai.djl.Device;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.Joints;
import ai.djl.modality.cv.output.Joints.Joint;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.BaseImageTranslator;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.Pair;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import xyz.hyhy.scai.utils.CVUtils;
import xyz.hyhy.scai.utils.ImageUtils;
import xyz.hyhy.scai.utils.NumpyUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * A {@link BaseImageTranslator} that post-process the {@link NDArray} into human {@link Joints}.
 */
public class FaceDetectionTranslator extends BasePairTranslator<Mat, Rectangle, Joints> {
    private float threshold;

    /**
     * Creates the Pose Estimation translator from the given builder.
     *
     * @param builder the builder for the translator
     */
    public FaceDetectionTranslator(Builder builder) {
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

    /**
     * {@inheritDoc}
     */
    @Override
    public Joints processOutput(TranslatorContext ctx, NDList list) {
        int[] croppedInfo = (int[]) ctx.getAttachment("croppedInfo");
        int topX = croppedInfo[0];
        int topY = croppedInfo[1];
        int realW = croppedInfo[2];

        NDArray pred = list.singletonOrThrow().toDevice(Device.cpu(), false).reshape(new Shape(-1, 2));
        int numJoints = (int) pred.getShape().get(0);
        float[] flattenedPred = pred.toFloatArray();
        List<Joint> joints = new ArrayList<>(numJoints);

        for (int i = 0; i < numJoints; i++) {
            joints.add(new Joint(
                    topX + realW * flattenedPred[2 * i],
                    topY + realW * flattenedPred[2 * i + 1],
                    1));
        }
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

    @Override
    public NDList processInput(TranslatorContext ctx, Pair<Mat, Rectangle> input) throws Exception {
        double scale = 1.25;
        int inputW = 56;
        Mat frame = input.getKey().clone();
        Rectangle bbox = input.getValue();

        double w = Math.max(bbox.getHeight(), bbox.getWidth());
        double centerX = bbox.getX() + w / 2;
        double centerY = bbox.getY() + w / 2;

        w = w * scale;

        int topX = (int) (centerX - w / 2);
        int topY = (int) (centerY - w / 2);
        int realW = (int) w;


        int[] croppedInfo = new int[]{topX, topY, realW, inputW};
        ctx.setAttachment("croppedInfo", croppedInfo);

        Mat croppedImg = CVUtils.cropSquareArea(frame, topX, topY, realW);
        Imgproc.resize(croppedImg, croppedImg, new Size(inputW, inputW));
        NDArray array = ImageUtils.mat2Image(croppedImg).toNDArray(ctx.getNDManager(), Image.Flag.COLOR);
        return pipeline.transform(new NDList(array));
    }

    /**
     * The builder for Pose Estimation translator.
     */
    public static class Builder extends BaseBuilder<Builder> {
        //Pose
        float[] mean = new float[]{0.485f, 0.456f, 0.406f};
        float[] std = new float[]{0.229f, 0.224f, 0.225f};

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
        public FaceDetectionTranslator build() {
            if (pipeline == null) {
                addTransform(new ToTensor());
                addTransform(new Normalize(mean, std));
            }
            validate();
            return new FaceDetectionTranslator(this);
        }
    }

    public static void main(String[] args) {
        NDManager ndManager = NumpyUtils.ndManager;
        int height = 5;
        int width = 4;
        NDArray WX = ndManager.arange(height).repeat(width).reshape(height, width);
        NDArray WY = ndManager.arange(width).tile(height).reshape(height, width);

        System.out.println(WX);
        System.out.println(WY);
    }
}
