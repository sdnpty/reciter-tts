# ONNX Runtime
-keep class ai.onnxruntime.** { *; }
-dontwarn ai.onnxruntime.**

# Keep TTS Service
-keep class com.qwen3.tts.engine.** { *; }
-keep class com.qwen3.tts.ui.** { *; }

# Keep native methods
-keepclasseswithmembernames class * {
    native <methods>;
}
