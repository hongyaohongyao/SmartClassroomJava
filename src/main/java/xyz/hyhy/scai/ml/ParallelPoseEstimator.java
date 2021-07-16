package xyz.hyhy.scai.ml;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.output.Joints;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;
import lombok.SneakyThrows;
import org.opencv.core.Mat;
import xyz.hyhy.scai.AlphaPose;
import xyz.hyhy.scai.translator.BasePairTranslator;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

public abstract class ParallelPoseEstimator implements AutoCloseable {
    protected int minLenToStartWork;
    protected static int DEFAULT_PARALLELISM = 3;

    protected ForkJoinPool pool;

    protected BasePairTranslator<Mat, Rectangle, Joints> poseTranslator;
    protected Criteria<Pair<Mat, Rectangle>, Joints> poseCriteria;
    protected ZooModel<Pair<Mat, Rectangle>, Joints> poseModel;
    Predictor<Pair<Mat, Rectangle>, Joints> posePredictor;

    @Override
    public void close() throws Exception {
        if (posePredictor != null)
            posePredictor.close();
        if (poseModel != null)
            poseModel.close();
        if (pool != null)
            pool.shutdown();
    }

    public static class PoseTask extends RecursiveAction {
        protected ParallelPoseEstimator ppe;
        protected Mat frame;
        protected List<Rectangle> bboxes;
        protected Joints[] outputs;
        protected int start;
        protected int end;
        protected int minLenToStartWork;

        public PoseTask(ParallelPoseEstimator ppe,
                        Mat frame,
                        List<Rectangle> bboxes,
                        Joints[] outputs,
                        int start, int end, int minLenToStartWork) {
            this.frame = frame;
            this.bboxes = bboxes;
            this.outputs = outputs;
            this.start = start;
            this.end = end;
            this.ppe = ppe;
            this.minLenToStartWork = minLenToStartWork;
        }

        @SneakyThrows
        @Override
        protected void compute() {
            int workLen = end - start + 1;
            if (workLen <= 0)
                return;
            if (workLen <= minLenToStartWork) {
                ppe.singleWork(this);
            } else {
                int mid = (end + start) / 2;
                invokeAll(new PoseTask(ppe, frame, bboxes, outputs,
                                start, mid, minLenToStartWork),
                        new PoseTask(ppe, frame, bboxes, outputs,
                                mid + 1, end, minLenToStartWork));
            }
        }
    }

    public ParallelPoseEstimator() throws Exception {
        this(DEFAULT_PARALLELISM, 3, false);
    }

    public abstract BasePairTranslator<Mat, Rectangle, Joints> getPoseTranslator();

    public abstract String getPoseModelName();

    public abstract String getLibrary();

    public abstract void singleWork(PoseTask poseTask) throws Exception;

    public ParallelPoseEstimator(int parallelism, int minLenToStartWork, boolean gpu) throws MalformedModelException, ModelNotFoundException, IOException {
        this.pool = new ForkJoinPool(parallelism);
        this.minLenToStartWork = minLenToStartWork;

        poseTranslator = getPoseTranslator();
        poseCriteria = Criteria.builder()
                .setTypes(poseTranslator.getPairClass(), Joints.class)
                .optDevice(gpu ? Device.gpu() : Device.cpu())
                .optModelUrls(AlphaPose.class.getResource("/sppe").getPath())
                .optModelName(getPoseModelName())
                .optTranslator(poseTranslator)
                .optEngine(getLibrary())
                .build();
        poseModel = ModelZoo.loadModel(poseCriteria);
        posePredictor = poseModel.newPredictor();
    }

    public List<Joints> infer(Mat frame, List<Rectangle> bboxes) {
        Joints[] outputs = new Joints[bboxes.size()];
        this.pool.invoke(
                new PoseTask(this, frame, bboxes, outputs,
                        0, bboxes.size() - 1, minLenToStartWork)
        );
        return Arrays.asList(outputs);
    }

    public List<Joints> inferOneByOne(Mat frame, List<Rectangle> bboxes) throws Exception {
        Joints[] outputs = new Joints[bboxes.size()];
        singleWork(new PoseTask(this, frame, bboxes, outputs,
                0, bboxes.size() - 1, minLenToStartWork));
        return Arrays.asList(outputs);
    }

    public List<Joints> batchInfer(Mat frame, List<Rectangle> bboxes) throws TranslateException {
        List<Pair<Mat, Rectangle>> inputObj = new ArrayList<>(bboxes.size());
        for (int i = 0; i < bboxes.size(); i++) {
            inputObj.add(new Pair<>(frame, bboxes.get(i)));
        }
        return posePredictor.batchPredict(inputObj);
    }
}
