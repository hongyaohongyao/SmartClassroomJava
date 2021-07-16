package xyz.hyhy.scai.core.tasks;

import lombok.Getter;
import lombok.Setter;
import xyz.hyhy.scai.core.commons.DataDict;
import xyz.hyhy.scai.core.modules.SCModule;

@Getter
@Setter
public class TaskData {
    protected DataDict data;
    protected DataDict globalData;
    protected TaskStage taskStage;
    public volatile int taskFlag;

    public TaskData(DataDict globalData, TaskStage taskStage) {
        this(globalData, taskStage, SCModule.OK);
    }

    public TaskData(DataDict globalData, TaskStage taskStage, int taskFlag) {
        this.data = new DataDict();
        this.globalData = globalData;
        this.taskStage = taskStage;
        this.taskFlag = taskFlag;
    }
}
