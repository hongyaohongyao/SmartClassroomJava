package xyz.hyhy.scai.utils;

import org.bytedeco.ffmpeg.global.avcodec;
import org.bytedeco.javacv.FrameRecorder;

public class FFmpegUtils {

    private FFmpegUtils() {


    }

    public static FrameRecorder getFFmpegFrameRecorder() throws FrameRecorder.Exception {
        // 直播流格式
        FrameRecorder recorder = FrameRecorder.createDefault("rtmp://localhost:1935/live/stream", 1280, 720);   //输出路径，画面高，画面宽
        recorder.setVideoCodec(avcodec.AV_CODEC_ID_H264);  //设置编码格式
        recorder.setFormat("flv");
        recorder.setFrameRate(25.0);
        recorder.setGopSize(25);
        recorder.start();
        return recorder;
    }


}
