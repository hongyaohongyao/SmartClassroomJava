package xyz.hyhy.scai.core.commons;

import xyz.hyhy.scai.core.modules.ExeModule;
import xyz.hyhy.scai.core.modules.SourceModule;
import xyz.hyhy.scai.core.tasks.TaskData;
import xyz.hyhy.scai.core.tasks.TaskSolution;
import xyz.hyhy.scai.core.tasks.TaskStage;

import java.util.Properties;
import java.util.concurrent.Future;
import java.util.function.Function;

public class TaskSolutionBuilder {
    private TaskStage startStage;
    private TaskStage curTaskStage;
    private SourceModule sourceModule;
    private DataDict globalData;
    private Future future;

    private TaskSolutionBuilder() {
        this.curTaskStage = this.startStage = new TaskStage();
    }

    public static TaskSolutionBuilder builder() {
        return new TaskSolutionBuilder();
    }

    public TaskSolutionBuilder setSourceModule(SourceModule sourceModule) {
        this.sourceModule = sourceModule;
        return this;
    }

    /**
     * 如果需要通过
     *
     * @param nextModule 返回下一个执行模块的引用的函数式接口
     * @return
     */
    public TaskSolutionBuilder setStage(Function<TaskData, ExeModule> nextModule) {
        TaskStage nextStage = new TaskStage();
        curTaskStage.setNextModule(nextModule);
        curTaskStage.setNextStage(nextStage);
        curTaskStage = nextStage;
        return this;
    }

    public TaskSolutionBuilder setStage(String nextStageName) throws Exception {
        ModulesCenter modulesCenter = ModulesCenter.getInstance();
        ExeModule exeModule = modulesCenter.get(nextStageName);
        if (exeModule != null) {
            return setStage((taskData) -> exeModule);
        } else {
            throw new Exception("未注册的模块");
        }
    }

    public TaskSolutionBuilder setGlobalData(DataDict globalData) {
        this.globalData = globalData;
        return this;
    }

    public TaskSolutionBuilder setGlobalData(Object key, Object value) {
        if (this.globalData == null)
            this.globalData = new DataDict();
        this.globalData.put(key, value);
        return this;
    }

    public TaskSolutionBuilder setGlobalData(Properties properties) {
        if (this.globalData == null)
            this.globalData = new DataDict();
        this.globalData.putAll(properties);
        return this;
    }

    public TaskSolutionBuilder resetGlobalData() {
        this.globalData = new DataDict();
        return this;
    }

    public TaskSolution build() {
        sourceModule.setTaskStage(startStage);
        sourceModule.setGlobalData(globalData == null ? new DataDict() : globalData);
        TaskSolution taskSolution = new TaskSolution(sourceModule, globalData);
        return taskSolution;
    }
}
