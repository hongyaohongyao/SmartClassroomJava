package xyz.hyhy.scai.pojo;

import ai.djl.modality.Classifications;
import ai.djl.modality.cv.output.DetectedObjects.DetectedObject;
import ai.djl.modality.cv.output.Joints;
import org.opencv.core.Mat;

import java.util.List;

public class ClassActionPOJO {
    public Mat frame;
    public List<DetectedObject> detectedObjects;
    public List<Joints> joints;
    public List<Classifications> classifications;
    public List<Mat[]> headPoses;
}
