package xyz.hyhy.scai.core.modules;

public interface SCModule extends Runnable, AutoCloseable {
    int OK = 0;
    int CLOSE = 1;
    int IGNORE = -1;

    default void open() throws Exception {

    }

}
