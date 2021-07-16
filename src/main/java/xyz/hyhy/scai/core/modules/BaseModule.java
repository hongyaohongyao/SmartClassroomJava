package xyz.hyhy.scai.core.modules;

import lombok.Setter;
import xyz.hyhy.scai.constant.CoreConst;
import xyz.hyhy.scai.core.commons.DataDict;
import xyz.hyhy.scai.core.tasks.TaskData;
import xyz.hyhy.scai.core.tasks.TaskStage;

@Setter
public abstract class BaseModule implements SCModule {
    protected volatile boolean running;
    protected Balancer balancer;
    protected volatile int processIntervalScale;
    protected static final TaskData IGNORE_TASK_DATA = new TaskData(null, null);

    static {
        IGNORE_TASK_DATA.taskFlag = SCModule.IGNORE;
    }

    public static class Balancer {
        protected volatile long maxInterval;
        protected volatile BaseModule shortSlabModule;

        public synchronized long getSuitableInterval(long processInterval, BaseModule module) {
            if (shortSlabModule == module) {
                maxInterval = (processInterval + maxInterval) / 2;
                return 0;
            } else if (processInterval > maxInterval) {
                shortSlabModule = module;
                maxInterval = processInterval;
                return 0;
            } else {
                return maxInterval - processInterval;
            }
        }
    }


    public BaseModule() {
        this.running = true;
    }

    protected abstract int processData(DataDict data, DataDict globalData) throws Exception;

    protected abstract TaskData productTaskData() throws Exception;

    @Override
    public void run() {
        try {
            open();
            while (running) {
                TaskData taskData = productTaskData();
                long startTimestamp = System.currentTimeMillis();
                int result = taskData.taskFlag == SCModule.OK ?
                        processData(taskData.getData(), taskData.getGlobalData()) :
                        taskData.taskFlag;
                long processInterval = Math.min(
                        (System.currentTimeMillis() - startTimestamp) * processIntervalScale,
                        CoreConst.BALANCE_CEILING_VALUE);
                // 下一步
                TaskStage taskStage = taskData.getTaskStage();
                if (result == SCModule.IGNORE) {
                    continue;
                } else {
                    if (result == SCModule.CLOSE) {
                        taskData.taskFlag = SCModule.CLOSE;
                        close();
                    }
                    if (taskStage.getNextStage() != null) {
                        taskStage.toNextStage(taskData);
                    }
                }
                if (balancer != null) {
                    long suitableInterval = balancer.getSuitableInterval(processInterval, this);
                    System.out.println(String.format("process: %d, wait: %d, name: %s", processInterval, suitableInterval, this));
                    if (this instanceof ExeModule)
                        System.out.println(String.format("name: %s, queue: %d", this, ((ExeModule) this).queue.size()));
                    if (suitableInterval > 0) {
                        Thread.sleep(suitableInterval);
                    }
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        System.out.println(String.format("============================Close: %s", this));
    }

    @Override
    public void close() throws Exception {
        running = false;
    }
}
