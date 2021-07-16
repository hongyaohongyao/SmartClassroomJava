package xyz.hyhy.scai.core.tasks;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import xyz.hyhy.scai.core.modules.ExeModule;

import java.util.function.Function;

@NoArgsConstructor
@AllArgsConstructor
@Getter
@Setter
public class TaskStage {
    protected TaskStage nextStage;
    protected Function<TaskData, ExeModule> nextModule;

    public void toNextStage(TaskData taskData) throws Exception {
        nextModule.apply(taskData).put(taskData);
        taskData.setTaskStage(nextStage);
    }
}
