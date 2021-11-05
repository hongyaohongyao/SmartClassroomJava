package xyz.hyhy.scai;

import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.ScalarVector;
import org.opencv.core.Core;
import xyz.hyhy.scai.core.commons.ModulesCenter;
import xyz.hyhy.scai.core.commons.TaskSolutionBuilder;
import xyz.hyhy.scai.core.tasks.TaskSolution;
import xyz.hyhy.scai.exemodules.AlphaPoseModule;
import xyz.hyhy.scai.exemodules.ClassActionModule;
import xyz.hyhy.scai.exemodules.DrawModule;
import xyz.hyhy.scai.exemodules.YoloModule;
import xyz.hyhy.scai.startmodules.VideoModule;

import java.util.concurrent.Future;


public class MainTest extends MainClass {


    public static void main(String[] args) {
        String source = AlphaPose.class.getResource("/front_cheat.mp4").getPath();
//        int source = 0;
        try {
            //注册执行模块
            ModulesCenter modulesCenter = ModulesCenter.getInstance();
            modulesCenter.newBalancer();
            modulesCenter.register("detection", new YoloModule());
            modulesCenter.register("pose", new AlphaPoseModule());
            modulesCenter.register("classAction", new ClassActionModule());
            Future future = modulesCenter.register("draw", new DrawModule());
            //设置任务方案
            TaskSolution taskSolution = TaskSolutionBuilder.builder()
                    .setSourceModule(new VideoModule(source))
                    .setStage("detection")
                    .setStage("pose")
                    .setStage("classAction")
                    .setStage("draw")
                    .build();
            // 启动任务方案
            taskSolution.on();
            taskSolution.waitForTerminate();
            future.get();
        } catch (Exception e) {
            e.printStackTrace();
        }
        System.exit(0);
    }
}
