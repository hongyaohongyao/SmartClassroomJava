package xyz.hyhy.scai.core.modules;

import xyz.hyhy.scai.constant.CoreConst;
import xyz.hyhy.scai.core.tasks.TaskData;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

public abstract class ExeModule extends BaseModule {
    protected BlockingQueue<TaskData> queue;

    public ExeModule() {
        this(new LinkedBlockingQueue<>(CoreConst.DEFAULT_MODULE_QUEUE_SIZE));
    }

    public ExeModule(BlockingQueue<TaskData> queue) {
        super();
        this.queue = queue;
    }

    public void put(TaskData taskData) throws InterruptedException {
        queue.put(taskData);
        processIntervalScale = Math.max(queue.size(), 1);
    }

    @Override
    protected TaskData productTaskData() throws Exception {
        TaskData taskData = queue.poll(3000, TimeUnit.MILLISECONDS);
        processIntervalScale = Math.max(queue.size(), 1);
//        if (taskData == null) {
//            System.out.println("dadas");
//        } else {
//            System.out.println(taskData.taskFlag == SCModule.CLOSE ? "close==================" : "");
//        }
        return taskData != null ? taskData : IGNORE_TASK_DATA;
    }

}
