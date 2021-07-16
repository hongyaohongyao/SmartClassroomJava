package xyz.hyhy.scai.core.commons;

import java.util.HashMap;
import java.util.Map;

public class DataDict extends HashMap<Object, Object> {

    public volatile int taskFlag;

    public DataDict(int initialCapacity, float loadFactor) {
        super(initialCapacity, loadFactor);
    }

    public DataDict(int initialCapacity) {
        super(initialCapacity);
    }

    public DataDict() {
    }

    public DataDict(Map<?, ?> m) {
        super(m);
    }
}
