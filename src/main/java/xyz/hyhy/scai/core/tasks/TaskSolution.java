package xyz.hyhy.scai.core.tasks;

import lombok.Getter;
import xyz.hyhy.scai.core.commons.DataDict;
import xyz.hyhy.scai.core.commons.ModulesCenter;
import xyz.hyhy.scai.core.modules.SourceModule;
import xyz.hyhy.scai.utils.CommonUtils;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

@Getter
public class TaskSolution {
    private SourceModule startModule;
    private DataDict globalData;
    private String randomID;
    private Future future;

    public TaskSolution(SourceModule startModule, DataDict globalData) {
        this.startModule = startModule;
        this.globalData = globalData;
        this.randomID = CommonUtils.getRandomIDWithDate();
    }

    public void on() {
        future = ModulesCenter.getInstance().register(this);
    }

    public Object waitForTerminate() throws ExecutionException, InterruptedException {
        return future.get();
    }

    public Object stop() throws Exception {
        ModulesCenter.getInstance().unregister(randomID);
        return future.get();
    }

}
