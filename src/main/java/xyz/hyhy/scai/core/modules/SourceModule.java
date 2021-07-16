package xyz.hyhy.scai.core.modules;

import lombok.Setter;
import xyz.hyhy.scai.core.commons.DataDict;
import xyz.hyhy.scai.core.tasks.TaskData;
import xyz.hyhy.scai.core.tasks.TaskStage;

@Setter
public abstract class SourceModule extends BaseModule {

    protected DataDict globalData;
    protected TaskStage taskStage;

    public SourceModule() {
        super();
    }

    @Override
    protected TaskData productTaskData() {
        return new TaskData(globalData, taskStage);
    }


}
