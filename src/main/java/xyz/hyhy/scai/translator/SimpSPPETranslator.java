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

import ai.djl.Application;
import ai.djl.Device;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.Joints;
import ai.djl.modality.cv.output.Joints.Joint;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.BaseImageTranslator;
import ai.djl.modality.cv.util.NDImageUtils;
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
import xyz.hyhy.scai.utils.CVUtils;
import xyz.hyhy.scai.utils.ImageUtils;
import xyz.hyhy.scai.utils.NumpyUtils;

import java.util.*;

/**
 * A {@link BaseImageTranslator} that post-process the {@link NDArray} into human {@link Joints}.
 */
public class SimpSPPETranslator extends BasePairTranslator<Mat, Rectangle, Joints> {
    private float threshold;

    /**
     * Creates the Pose Estimation translator from the given builder.
     *
     * @param builder the builder for the translator
     */
    public SimpSPPETranslator(Builder builder) {
        super(builder);
        this.threshold = builder.threshold;
    }

    private static long[] tilePram = new long[]{1, 1, 2};
    private static int[] axis2 = new int[]{2};
    private static int[] axis2n3 = new int[]{2, 3};
    private static int[] axis2n4 = new int[]{2, 4};
    private static int[] axis3n4 = new int[]{3, 4};
    private static NDArray ONES_NDARRAY = NumpyUtils.ndManager.ones(new Shape(136, 1));


    /**
     * {@inheritDoc}
     */
    @Override
    public Joints processOutput(TranslatorContext ctx, NDList list) {
        NDArray pred = list.singletonOrThrow().toDevice(Device.cpu(), false);
        int numJoints = (int) pred.getShape().get(0);


        NDManager ndManager = NumpyUtils.ndManager;
        NDArray predJoints = pred.get(new NDIndex(":,:2"));
        NDArray maxValues = pred.get(new NDIndex(":,2:"));

        Rectangle bbox = (Rectangle) ((Queue) ctx.getAttachment("cropped_bboxes")).poll();
        double x = bbox.getX();
        double y = bbox.getY();
        double w = bbox.getWidth();
        double h = bbox.getHeight();
        double centerX = x + 0.5 * w, centerY = y + 0.5 * h;
        double scaleX = w;

        Mat trans = CVUtils.getAffineTransform(centerX, centerY, 48, 64, scaleX, true);
        NDArray ndTrans = CVUtils.transMat2NDArray(trans, ndManager).transpose(1, 0);

        predJoints = predJoints
                .concat(ONES_NDARRAY, 1);

        NDArray xys = predJoints.matMul(ndTrans);


        float[] flattened = xys.toFloatArray();
        float[] flattenedConfidence = maxValues.toFloatArray();

        List<Joint> joints = new ArrayList<>(numJoints);
        for (int i = 0; i < numJoints; ++i) {
            joints.add(new Joint(
                    flattened[2 * i],
                    flattened[2 * i + 1],
                    flattenedConfidence[i]));
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

    @Override
    public NDList processInput(TranslatorContext ctx, Pair<Mat, Rectangle> input) throws Exception {
        Mat frame = input.getKey().clone();
        Rectangle bbox = input.getValue();
        int x = (int) Math.max(0, bbox.getX());
        int y = (int) Math.max(0, bbox.getY());
        int w = Math.min(frame.width(), (int) (x + bbox.getWidth())) - x;
        int h = Math.min(frame.height(), (int) (y + bbox.getHeight())) - y;
        Rectangle croppedBBox = CVUtils.scale(frame, x, y, w, h);


        Queue cropped_bboxes = (Queue) ctx.getAttachment("cropped_bboxes");
        if (cropped_bboxes == null) {
            cropped_bboxes = new LinkedList<>();
            ctx.setAttachment("cropped_bboxes", cropped_bboxes);
        }
        cropped_bboxes.add(croppedBBox);

        NDArray array = ImageUtils.mat2Image(frame).toNDArray(ctx.getNDManager(), Image.Flag.COLOR);

        return pipeline.transform(new NDList(array));
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
        public SimpSPPETranslator build() {
            if (pipeline == null) {
                addTransform(new ToTensor());
                addTransform(new Normalize(mean, std));
            }
            validate();
            return new SimpSPPETranslator(this);
        }
    }
}
