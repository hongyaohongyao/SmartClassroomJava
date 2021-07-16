package xyz.hyhy.scai.utils;

import ai.djl.modality.cv.output.Joints.Joint;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import xyz.hyhy.scai.constant.CoreConst;

import java.util.List;

public class PoseEstimator {

    public static final MatOfPoint3f MODEL_POINTS_68 = getModelPoints68();
    public static final MatOfPoint3f DRAW_AXIS_POINTS = getDrawAxisPoints();
    public static final Scalar COLOR_X_AXIS = new Scalar(255, 0, 0);
    public static final Scalar COLOR_Y_AXIS = new Scalar(0, 255, 0);
    public static final Scalar COLOR_Z_AXIS = new Scalar(0, 0, 255);

    private int focalLength;
    private Mat cameraMatrix;
    private final MatOfDouble distCoeffs;

    private static class SingletonHolder {
        public static PoseEstimator INSTANCE = new PoseEstimator(CoreConst.IMG_WIDTH, CoreConst.IMG_HEIGHT);
    }

    public static PoseEstimator getInstance() {
        return SingletonHolder.INSTANCE;
    }


    public PoseEstimator(int imgWidth, int imgHeight) {
        focalLength = imgHeight;

        cameraMatrix = new MatOfDouble(
                focalLength, 0, imgHeight,
                0, focalLength, imgWidth,
                0, 0, 1)
                .reshape(1, 3);
        distCoeffs = new MatOfDouble(0, 0, 0, 0);
    }

    public Mat[] solvePose(List<Joint> joints68) {
        Point[] points = new Point[joints68.size()];
        for (int i = 0; i < points.length; i++) {
            Joint jt = joints68.get(i);
            points[i] = new Point(jt.getX(), jt.getY());
        }

        MatOfPoint2f matOfPoint2f = new MatOfPoint2f(points);
        Mat rVec = new Mat();
        Mat tVec = new Mat();
        Calib3d.solvePnP(MODEL_POINTS_68, matOfPoint2f, cameraMatrix, distCoeffs, rVec, tVec);
        return new Mat[]{rVec, tVec};
    }

    public void drawAxis(Mat frame, Mat rVec, Mat tVec) {
        MatOfPoint2f imgPoints = new MatOfPoint2f();
        Calib3d.projectPoints(DRAW_AXIS_POINTS,
                rVec, tVec,
                cameraMatrix, distCoeffs,
                imgPoints);
        List<Point> axisPoints = imgPoints.toList();
        Imgproc.line(frame, axisPoints.get(3), axisPoints.get(0), COLOR_X_AXIS, 3);
        Imgproc.line(frame, axisPoints.get(3), axisPoints.get(1), COLOR_Y_AXIS, 3);
        Imgproc.line(frame, axisPoints.get(3), axisPoints.get(2), COLOR_Z_AXIS, 3);
    }

    private static MatOfPoint3f getModelPoints68() {
        double[][] points = new double[][]{
                {-73.393523, -29.801432, -47.667532},
                {-72.775014, -10.949766, -45.909403},
                {-70.533638, 7.929818, -44.84258},
                {-66.850058, 26.07428, -43.141114},
                {-59.790187, 42.56439, -38.635298},
                {-48.368973, 56.48108, -30.750622},
                {-34.121101, 67.246992, -18.456453},
                {-17.875411, 75.056892, -3.609035},
                {0.098749, 77.061286, 0.881698},
                {17.477031, 74.758448, -5.181201},
                {32.648966, 66.929021, -19.176563},
                {46.372358, 56.311389, -30.77057},
                {57.34348, 42.419126, -37.628629},
                {64.388482, 25.45588, -40.886309},
                {68.212038, 6.990805, -42.281449},
                {70.486405, -11.666193, -44.142567},
                {71.375822, -30.365191, -47.140426},
                {-61.119406, -49.361602, -14.254422},
                {-51.287588, -58.769795, -7.268147},
                {-37.8048, -61.996155, -0.442051},
                {-24.022754, -61.033399, 6.606501},
                {-11.635713, -56.686759, 11.967398},
                {12.056636, -57.391033, 12.051204},
                {25.106256, -61.902186, 7.315098},
                {38.338588, -62.777713, 1.022953},
                {51.191007, -59.302347, -5.349435},
                {60.053851, -50.190255, -11.615746},
                {0.65394, -42.19379, 13.380835},
                {0.804809, -30.993721, 21.150853},
                {0.992204, -19.944596, 29.284036},
                {1.226783, -8.414541, 36.94806},
                {-14.772472, 2.598255, 20.132003},
                {-7.180239, 4.751589, 23.536684},
                {0.55592, 6.5629, 25.944448},
                {8.272499, 4.661005, 23.695741},
                {15.214351, 2.643046, 20.858157},
                {-46.04729, -37.471411, -7.037989},
                {-37.674688, -42.73051, -3.021217},
                {-27.883856, -42.711517, -1.353629},
                {-19.648268, -36.754742, 0.111088},
                {-28.272965, -35.134493, 0.147273},
                {-38.082418, -34.919043, -1.476612},
                {19.265868, -37.032306, 0.665746},
                {27.894191, -43.342445, -0.24766},
                {37.437529, -43.110822, -1.696435},
                {45.170805, -38.086515, -4.894163},
                {38.196454, -35.532024, -0.282961},
                {28.764989, -35.484289, 1.172675},
                {-28.916267, 28.612716, 2.24031},
                {-17.533194, 22.172187, 15.934335},
                {-6.68459, 19.029051, 22.611355},
                {0.381001, 20.721118, 23.748437},
                {8.375443, 19.03546, 22.721995},
                {18.876618, 22.394109, 15.610679},
                {28.794412, 28.079924, 3.217393},
                {19.057574, 36.298248, 14.987997},
                {8.956375, 39.634575, 22.554245},
                {0.381549, 40.395647, 23.591626},
                {-7.428895, 39.836405, 22.406106},
                {-18.160634, 36.677899, 15.121907},
                {-24.37749, 28.677771, 4.785684},
                {-6.897633, 25.475976, 20.893742},
                {0.340663, 26.014269, 22.220479},
                {8.444722, 25.326198, 21.02552},
                {24.474473, 28.323008, 5.712776},
                {8.449166, 30.596216, 20.671489},
                {0.205322, 31.408738, 21.90367},
                {-7.198266, 30.844876, 20.328022}};
        Point3[] point3s = new Point3[points.length];

        for (int i = 0; i < point3s.length; i++) {
            double[] pt = points[i];
            point3s[i] = new Point3(pt[0], pt[1], pt[2]);
        }
        return new MatOfPoint3f(point3s);
    }

    private static MatOfPoint3f getDrawAxisPoints() {
        double[][] points = new double[][]{
                {30, 0, 0},
                {0, 30, 0},
                {0, 0, 30},
                {0, 0, 0}
        };
        Point3[] point3s = new Point3[points.length];

        for (int i = 0; i < point3s.length; i++) {
            double[] pt = points[i];
            point3s[i] = new Point3(pt[0], pt[1], pt[2]);
        }
        return new MatOfPoint3f(point3s);
    }


}
