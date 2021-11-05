package xyz.hyhy.scai;

import ai.djl.pytorch.jni.JniUtils;
import ai.djl.engine.Engine;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.opencv.opencv_java;
import org.opencv.core.Core;

public class MainClass {
    static {
//        System.loadLibrary("D:\\cache\\djl\\pytorch\\1.8.1-cu111-win-x86_64\\cudnn_cnn_infer64_8.dll");
        Loader.load(opencv_java.class);
        String cacheDir = System.getenv("DEFAULT_CACHE_DIR");
        System.setProperty("DJL_CACHE_DIR", cacheDir + "/djl");
        System.setProperty("ENGINE_CACHE_DIR", cacheDir + "/djl");
        initOrtLibraryProperties(cacheDir);
        Engine engine = Engine.getEngine("PyTorch");
        JniUtils.setGraphExecutorOptimize(false);
//        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void initOrtLibraryProperties(String cacheDir) {
        String[] libraryNames = new String[]{"onnxruntime", "onnxruntime4j_jni"};
        for (String libraryName : libraryNames)
            System.setProperty(String.format("onnxruntime.native.%s.path", libraryName),
                    String.format("/%s/djl/onnxruntime/%s.dll", cacheDir, libraryName));
    }
}
