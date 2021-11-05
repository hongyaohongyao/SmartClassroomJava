/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package xyz.hyhy.scai.utils;

import ai.djl.modality.cv.BufferedImageFactory;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Joints;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import lombok.SneakyThrows;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.RenderedImage;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.file.Path;

/**
 * {@code BufferedImageFactory} is the default implementation of {@link ImageFactory}.
 */
public class CVImageFactory extends ImageFactory {

    private static class OtherImageFactory {
        public static BufferedImageFactory bufferedImageFactory = new BufferedImageFactory();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Image fromFile(Path path) throws IOException {
        return new CVImageWrapper(Imgcodecs.imread(path.toAbsolutePath().toString()));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Image fromUrl(URL url) throws IOException {
        return new CVImageWrapper(Imgcodecs.imread(url.getPath()));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Image fromInputStream(InputStream is) throws IOException {
        BufferedImage image = ImageIO.read(is);
        if (image == null) {
            throw new IOException("Failed to read image from input stream");
        }
        byte[] pixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        Mat newMat = new Mat(image.getHeight(), image.getWidth(), CvType.CV_8UC3);
        newMat.put(0, 0, pixels);
        return new CVImageWrapper(newMat);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Image fromImage(Object image) {
        if (image instanceof BufferedImage) {
            return OtherImageFactory.bufferedImageFactory.fromImage(image);
        } else if (image instanceof Mat) {
            return new CVImageWrapper((Mat) image);
        } else {
            throw new IllegalArgumentException("only BufferedImage, opencv.core.Mat allowed");
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Image fromNDArray(NDArray array) {
        byte[] pixels = array.toByteArray();
        Shape shape = array.getShape();
        Mat mat = new Mat((int) shape.get(0), (int) shape.get(1), CvType.CV_8UC3);
        mat.put(0, 0, pixels);
        return new CVImageWrapper(mat);
    }


    protected void save(Mat image, OutputStream os, String type) throws IOException {
        ImageIO.write((RenderedImage) HighGui.toBufferedImage(image), type, os);
    }

    private class CVImageWrapper implements Image {

        private final Mat image;

        CVImageWrapper(Mat image) {
            this.image = image;
        }

        /**
         * {@inheritDoc}
         */
        @Override
        public int getWidth() {
            return image.width();
        }

        /**
         * {@inheritDoc}
         */
        @Override
        public int getHeight() {
            return image.height();
        }

        /**
         * {@inheritDoc}
         */
        @Override
        public Object getWrappedImage() {
            return image;
        }

        /**
         * {@inheritDoc}
         */
        @Override
        public Image getSubimage(int x, int y, int w, int h) {
            return new CVImageWrapper(image.submat(new Rect(x, y, w, h)));
        }

        /**
         * {@inheritDoc}
         */
        @Override
        public Image duplicate(Type type) {
            return new CVImageWrapper(image.clone());
        }

        /**
         * {@inheritDoc}
         */
        @Override
        public NDArray toNDArray(NDManager manager, Flag flag) {
            int width = image.width();
            int height = image.height();
            int channel;
            if (flag == Flag.GRAYSCALE) {
                channel = 1;
            } else {
                channel = 3;
            }

            byte[] pixels = new byte[channel * width * height];
            image.get(0, 0, pixels);
            return manager.create(ByteBuffer.wrap(pixels), new Shape(height, width, channel), DataType.UINT8);
        }

        /**
         * {@inheritDoc}
         */
        @Override
        public void save(OutputStream os, String type) throws IOException {
            CVImageFactory.this.save(image, os, type);
        }

        /**
         * {@inheritDoc}
         */
        @SneakyThrows
        @Override
        public void drawBoundingBoxes(DetectedObjects detections) {
            throw new IllegalAccessException("No Implementation");
        }

        /**
         * {@inheritDoc}
         */
        @SneakyThrows
        @Override
        public void drawJoints(Joints joints) {
            throw new IllegalAccessException("No Implementation");
        }


    }
}
