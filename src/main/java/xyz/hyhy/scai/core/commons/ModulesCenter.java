package xyz.hyhy.scai.core.commons;

import lombok.Setter;
import xyz.hyhy.scai.core.modules.BaseModule;
import xyz.hyhy.scai.core.modules.ExeModule;
import xyz.hyhy.scai.core.modules.SCModule;
import xyz.hyhy.scai.core.modules.SourceModule;
import xyz.hyhy.scai.core.tasks.TaskSolution;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

@Setter
public class ModulesCenter implements AutoCloseable {
    private Map<String, SCModule> modules;
    private ExecutorService service;
    private BaseModule.Balancer balancer;

    public void newBalancer() {
        balancer = new BaseModule.Balancer();
    }

    private ModulesCenter() {
        modules = new HashMap<>();
        service = Executors.newCachedThreadPool();
    }

    private static class InstanceHolder {
        public static ModulesCenter INSTANCE = new ModulesCenter();
    }

    public static ModulesCenter getInstance() {
        return InstanceHolder.INSTANCE;
    }

    public Future register(String name, ExeModule module) {
        if (!modules.containsKey(name)) {
            module.setBalancer(balancer);
            modules.put(name, module);
            return service.submit(module);
        }
        return null;
    }

    public Future register(TaskSolution taskSolution) {
        if (!modules.containsKey(taskSolution.getRandomID())) {
            SourceModule module = taskSolution.getStartModule();
            module.setBalancer(balancer);
            modules.put(taskSolution.getRandomID(), module);
            return service.submit(module);
        }
        return null;
    }

    public <T extends SCModule> T unregister(String name) throws Exception {
        if (!modules.containsKey(name)) {
            SCModule module = modules.remove(name);
            module.close();
            return (T) module;
        }
        return null;
    }

    public SourceModule unregister(TaskSolution taskSolution) throws Exception {
        return unregister(taskSolution.getRandomID());
    }

    public <T extends ExeModule> T get(String name) {
        if (!modules.containsKey(name))
            return null;
        return (T) modules.get(name);
    }

    @Override
    public void close() throws Exception {
        for (SCModule module : modules.values()) {
            module.close();
        }
    }

}
