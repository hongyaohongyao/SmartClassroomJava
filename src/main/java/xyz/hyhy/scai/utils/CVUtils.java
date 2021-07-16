package xyz.hyhy.scai.utils;

import ai.djl.modality.cv.output.Joints;
import ai.djl.modality.cv.output.Joints.Joint;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;


public class CVUtils {


    public static class KP {
        public final int[][] L_PAIR;
        public final Scalar[] P_COLOR;
        public final Scalar[] LINE_COLOR;
        public final Scalar DEFAULT_COLOR = new Scalar(255, 255, 255);

        public KP(String fileName) {
            Scanner scan = new Scanner(KP.class.getResourceAsStream(fileName));
            StringBuffer sb = new StringBuffer();
            while (scan.hasNext()) {
                sb.append(scan.nextLine());
            }
            JSONObject obj = JSON.parseObject(sb.toString());
            // set L_Pair
            JSONArray arr = obj.getJSONArray("l_pair");
            int size = arr.size();
            L_PAIR = new int[size][2];
            for (int i = 0; i < size; i++) {
                JSONArray xy = (JSONArray) arr.get(i);
                L_PAIR[i][0] = xy.getIntValue(0);
                L_PAIR[i][1] = xy.getIntValue(1);
            }
            // set P_COLOR
            arr = obj.getJSONArray("p_color");
            size = arr.size();
            P_COLOR = new Scalar[size];
            for (int i = 0; i < size; i++) {
                JSONArray color = (JSONArray) arr.get(i);
                P_COLOR[i] = new Scalar(color.getIntValue(0),
                        color.getIntValue(1),
                        color.getIntValue(2));
            }
            // set LINE_COLOR
            arr = obj.getJSONArray("line_color");
            size = arr.size();
            LINE_COLOR = new Scalar[size];
            for (int i = 0; i < size; i++) {
                JSONArray color = (JSONArray) arr.get(i);
                LINE_COLOR[i] = new Scalar(color.getIntValue(0),
                        color.getIntValue(1),
                        color.getIntValue(2));
            }
        }
    }

    public static final KP KP136 = new KP("/kp136.json");
    public static final KP KP17 = new KP("/kp17.json");
    public static final KP KP86 = new KP("/kp86.json");


    public static Mat getAffineTransform(double centerX, double centerY,
                                         double outputW, double outputH,
                                         double scaleX, boolean inv) {
        double oupCenterX = outputW / 2, oupCenterY = outputH / 2;
        MatOfPoint2f srcMat = new MatOfPoint2f(new Point(centerX, centerY),
                new Point(centerX, centerY + scaleX * -0.5),
                new Point(centerX - scaleX * 0.5, centerY + scaleX * -0.5));

        MatOfPoint2f dstMat = new MatOfPoint2f(new Point(oupCenterX, oupCenterY),
                new Point(oupCenterX, oupCenterY + outputW * -0.5),
                new Point(oupCenterX - outputW * 0.5, oupCenterY + outputW * -0.5));
        if (inv) {
            return Imgproc.getAffineTransform(dstMat, srcMat);
        } else {
            return Imgproc.getAffineTransform(srcMat, dstMat);
        }
    }

    public static Rectangle scale(Mat mat,
                                  double x, double y, double w, double h) {
        return scale(mat, x, y, w, h, 256, 192);
    }


    /**
     * Convert box coordinates to center and scale.
     * adapted from https://github.com/Microsoft/human-pose-estimation.pytorch
     *
     * @param mat
     * @param x
     * @param y
     * @param w
     * @param h
     * @return cropped box
     */
    public static Rectangle scale(Mat mat,
                                  double x, double y, double w, double h, double inputH, double inputW) {
        double inpCenterX = inputW / 2, inpCenterY = inputH / 2;
        double aspectRatio = inputW / inputH;

        double scaleMult = 1.25;
        // box_to_center_scale
        double centerX = x + 0.5 * w;
        double centerY = y + 0.5 * h;

        if (w > aspectRatio * h)
            h = w / aspectRatio;
        else if (w < aspectRatio * h)
            w = h * aspectRatio;
        double scaleX = w * scaleMult;
        double scaleY = h * scaleMult;

//        double rot = 0;
//        double sn = Math.sin(rot), cs = Math.cos(rot);
        // 获取仿射矩阵
        Mat trans = getAffineTransform(centerX, centerY, inputW, inputH, scaleX, false);
        // 仿射变化
        Imgproc.warpAffine(mat, mat, trans, new Size(inputW, inputH), Imgproc.INTER_LINEAR);

//        HighGui.imshow("person", mat);
        return new Rectangle(centerX - scaleX * 0.5, centerY - scaleY * 0.5, scaleX, scaleY);
    }

    public static NDArray transMat2NDArray(Mat mat, NDManager ndManager) {
        MatOfFloat matOfFloat = new MatOfFloat();
        mat.reshape(1, (int) mat.total()).convertTo(matOfFloat, CvType.CV_32F);
        float[] f = matOfFloat.toArray();
        return ndManager.create(f, new Shape(mat.height(), mat.width()));
    }


    public static void drawLandMarks(Mat frame, Joints joints) {
        List<Joint> jointList = joints.getJoints();
        double visThres = 0.025;
        int jointNum = jointList.size();
        for (int i = 0; i < jointNum; i++) {
            Joint jt = jointList.get(i);
            double conf = jt.getConfidence();
            if (conf < visThres)
                continue;
            int corX = (int) jt.getX(), corY = (int) jt.getY();
            if (i < KP136.P_COLOR.length)
                Imgproc.circle(frame, new Point(corX, corY), 2, KP136.P_COLOR[i], -1);
            else
                Imgproc.circle(frame, new Point(corX, corY), 1, KP136.DEFAULT_COLOR, 2);
        }
    }

    public static void drawKeypointsLight(Mat frame, Joints joints, double visThres, KP kp) {
        List<Joint> jointList = joints.getJoints();
        Map<Integer, Joint> partLine = new HashMap<>();
        int jointNum = jointList.size();
        for (int i = 0; i < jointNum; i++) {
            Joint jt = jointList.get(i);
            double conf = jt.getConfidence();
            if (conf < visThres)
                continue;
            int corX = (int) jt.getX(), corY = (int) jt.getY();
            partLine.put(i, jt);
            if (i < kp.P_COLOR.length)
                Imgproc.circle(frame, new Point(corX, corY), 2, kp.P_COLOR[i], -1);
            else
                Imgproc.circle(frame, new Point(corX, corY), 1, kp.DEFAULT_COLOR, 2);
        }
        // draw limbs
        for (int i = 0; i < kp.L_PAIR.length; i++) {
            int startP = kp.L_PAIR[i][0], endP = kp.L_PAIR[i][1];
            if (!partLine.containsKey(startP) || !partLine.containsKey(endP))
                continue;
            Joint startJ = partLine.get(startP);
            Joint endJ = partLine.get(endP);
            double combineConf = startJ.getConfidence() + endJ.getConfidence();

            double mX = (startJ.getX() + endJ.getX()) / 2;
            double mY = (startJ.getY() + endJ.getY()) / 2;
            double eX = startJ.getX() - endJ.getX();
            double eY = startJ.getY() - endJ.getY();
            double len = Math.sqrt(Math.pow(eX, 2) + Math.pow(eY, 2));
            double angle = Math.toDegrees(Math.atan2(eY, eX));
            double stickWidth = 1 + combineConf;
            MatOfPoint polygon = new MatOfPoint();
            Imgproc.ellipse2Poly(new Point(mX, mY),
                    new Size(len / 2, stickWidth),
                    (int) angle,
                    0, 360, 1, polygon);

            if (i < kp.LINE_COLOR.length) {
                Imgproc.fillConvexPoly(frame, polygon, kp.LINE_COLOR[i]);
            } else {
                Imgproc.line(frame,
                        new Point(startJ.getX(), startJ.getY()),
                        new Point(endJ.getX(), endJ.getY()),
                        kp.DEFAULT_COLOR, 1);
            }
        }
    }

    public static void draw86KeypointsLight(Mat frame, Joints joints) {
        drawKeypointsLight(frame, joints, 0.25, KP86);
    }


    public static void draw17KeypointsLight(Mat frame, Joints joints) {
        drawKeypointsLight(frame, joints, 0.25, KP17);
    }

    public static void draw136KeypointsLight(Mat frame, Joints joints) {
        List<Joint> jointList = joints.getJoints();
        Map<Integer, Joint> partLine = new HashMap<>();
        double visThres = 0.0025;
        int jointNum = 94;
        for (int i = 0; i < jointNum; i++) {
            Joint jt = jointList.get(i);
            double conf = jt.getConfidence();
            if (conf < visThres)
                continue;
            int corX = (int) jt.getX(), corY = (int) jt.getY();
            partLine.put(i, jt);
            if (i < KP136.P_COLOR.length)
                Imgproc.circle(frame, new Point(corX, corY), 2, KP136.P_COLOR[i], -1);
            else
                Imgproc.circle(frame, new Point(corX, corY), 1, KP136.DEFAULT_COLOR, 2);
        }
        // draw limbs
        for (int i = 0; i < KP136.L_PAIR.length; i++) {
            int startP = KP136.L_PAIR[i][0], endP = KP136.L_PAIR[i][1];
            if (!partLine.containsKey(startP) || !partLine.containsKey(endP))
                continue;
            Joint startJ = partLine.get(startP);
            Joint endJ = partLine.get(endP);
            double combineConf = startJ.getConfidence() + endJ.getConfidence();

            double mX = (startJ.getX() + endJ.getX()) / 2;
            double mY = (startJ.getY() + endJ.getY()) / 2;
            double eX = startJ.getX() - endJ.getX();
            double eY = startJ.getY() - endJ.getY();
            double len = Math.sqrt(Math.pow(eX, 2) + Math.pow(eY, 2));
            double angle = Math.toDegrees(Math.atan2(eY, eX));
            double stickWidth = 1 + combineConf;
            MatOfPoint polygon = new MatOfPoint();
            Imgproc.ellipse2Poly(new Point(mX, mY),
                    new Size(len / 2, stickWidth),
                    (int) angle,
                    0, 360, 1, polygon);

            if (i < KP136.LINE_COLOR.length) {
                Imgproc.fillConvexPoly(frame, polygon, KP136.LINE_COLOR[i]);
            } else {
                Imgproc.line(frame,
                        new Point(startJ.getX(), startJ.getY()),
                        new Point(endJ.getX(), endJ.getY()),
                        KP136.DEFAULT_COLOR, 1);
            }
        }
    }


    public static void draw136Keypoints(Mat frame, Joints joints) {
        List<Joint> jointList = joints.getJoints();
        Map<Integer, Joint> partLine = new HashMap<>();
        double visThres = 0.025;
        int jointNum = jointList.size();
        for (int i = 0; i < jointNum; i++) {
            Joint jt = jointList.get(i);
            double conf = jt.getConfidence();
            if (conf < visThres)
                continue;
            int corX = (int) jt.getX(), corY = (int) jt.getY();
            partLine.put(i, jt);
            Mat bg = frame.clone();
            if (i < KP136.P_COLOR.length)
                Imgproc.circle(bg, new Point(corX, corY), 2, KP136.P_COLOR[i], -1);
            else
                Imgproc.circle(bg, new Point(corX, corY), 1, KP136.DEFAULT_COLOR, 2);
            // 设置透明度
            int a = i < KP136.P_COLOR.length ? 1 : 2;
            double transparency = Math.max(0, Math.min(1, conf * a));
            Core.addWeighted(bg, transparency, frame, 1 - transparency, 0, frame);
        }
        // draw limbs
        for (int i = 0; i < KP136.L_PAIR.length; i++) {
            int startP = KP136.L_PAIR[i][0], endP = KP136.L_PAIR[i][1];
            if (!partLine.containsKey(startP) || !partLine.containsKey(endP))
                continue;
            Joint startJ = partLine.get(startP);
            Joint endJ = partLine.get(endP);
            double combineConf = startJ.getConfidence() + endJ.getConfidence();

            double mX = (startJ.getX() + endJ.getX()) / 2;
            double mY = (startJ.getY() + endJ.getY()) / 2;
            double eX = startJ.getX() - endJ.getX();
            double eY = startJ.getY() - endJ.getY();
            double len = Math.sqrt(Math.pow(eX, 2) + Math.pow(eY, 2));
            double angle = Math.toDegrees(Math.atan2(eY, eX));
            double stickWidth = 1 + combineConf;
            MatOfPoint polygon = new MatOfPoint();
            Imgproc.ellipse2Poly(new Point(mX, mY),
                    new Size(len / 2, stickWidth),
                    (int) angle,
                    0, 360, 1, polygon);

            Mat bg = frame.clone();
            double transparency;
            if (i < KP136.LINE_COLOR.length) {
                Imgproc.fillConvexPoly(bg, polygon, KP136.LINE_COLOR[i]);
                transparency = Math.max(0, Math.min(1, 0.5 * combineConf - 0.1));
            } else {
                Imgproc.line(bg,
                        new Point(startJ.getX(), startJ.getY()),
                        new Point(endJ.getX(), endJ.getY()),
                        KP136.DEFAULT_COLOR, 1);
                transparency = Math.max(0, Math.min(1, combineConf));
            }
            Core.addWeighted(bg, transparency, frame, 1 - transparency, 0, frame);
        }
    }

    /**
     * OpenCV图像剪切，超出图像区域指定颜色填充
     * http://www.voidcn.com/article/p-aeiqeujf-bmt.html
     *
     * @param frame
     * @param x
     * @param y
     * @param w
     * @return
     */
    public static Mat cropSquareArea(Mat frame, int x, int y, int w) {
        final Scalar fillScalar = new Scalar(0, 0, 0);

        int x1 = Math.max(0, x);
        int y1 = Math.max(0, y);
        int x2 = Math.min(frame.width(), (x + w));
        int y2 = Math.min(frame.height(), (y + w));
        Mat subMat = frame.submat(y1, y2, x1, x2);
        // 如果需要填边
        int leftX = (-x);
        int topY = (-y);
        int rightX = x + w - frame.cols();
        int downY = y + w - frame.rows();

        if (leftX > 0 || topY > 0 || rightX > 0 || downY > 0) {
            leftX = leftX > 0 ? leftX : 0;
            topY = topY > 0 ? topY : 0;
            rightX = rightX > 0 ? rightX : 0;
            downY = downY > 0 ? downY : 0;
            Mat result = new Mat();
            Core.copyMakeBorder(subMat,
                    result,
                    topY, downY, leftX, rightX,
                    Core.BORDER_CONSTANT,
                    fillScalar);
            return result;
        } else
            return subMat;
    }

    /**
     * Mat转换成byte数组
     *
     * @param matrix        要转换的Mat
     * @param fileExtension 格式为 ".jpg", ".png", etc
     * @return
     */
    public static byte[] mat2Byte(Mat matrix, String fileExtension) {
        MatOfByte mob = new MatOfByte();
        Imgcodecs.imencode(fileExtension, matrix, mob);
        byte[] byteArray = mob.toArray();
        return byteArray;
    }

}
