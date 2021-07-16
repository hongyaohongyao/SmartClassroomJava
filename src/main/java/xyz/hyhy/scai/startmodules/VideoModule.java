package xyz.hyhy.scai.startmodules;

import org.opencv.core.Mat;
import org.opencv.videoio.VideoCapture;
import xyz.hyhy.scai.core.commons.DataDict;
import xyz.hyhy.scai.core.modules.SourceModule;
import xyz.hyhy.scai.pojo.ClassActionPOJO;

public class VideoModule extends SourceModule {

    private VideoCapture cap;
    private Mat frame;
    private boolean flag;
    private long interval;
    private Object source;


    public VideoModule() {
        this(0);
    }

    public VideoModule(Object source) {
        super();
        this.source = source;
    }

    @Override
    protected int processData(DataDict data, DataDict globalData) throws InterruptedException {
        try {
            if (flag) {
                ClassActionPOJO pojo = new ClassActionPOJO();
                pojo.frame = frame;
                data.put("pojo", pojo);
                frame = new Mat();
                flag = cap.read(frame);
                if (!flag)
                    return CLOSE;
                return OK;
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return CLOSE;
    }

    @Override
    public void open() throws Exception {
        interval = (long) globalData.getOrDefault("webcam.interval", 30L);
        if (source instanceof String) {
            cap = new VideoCapture((String) source);
        } else if (source instanceof Integer) {
            cap = new VideoCapture((Integer) source);
        } else {
            throw new Exception("Source must be integer or filename(String)");
        }

        if (!cap.isOpened()) {//isOpened函数用来判断摄像头调用是否成功
            System.out.println("Camera Error");//如果摄像头调用失败，输出错误信息
        }
        frame = new Mat();
        flag = cap.read(frame);
    }

}
