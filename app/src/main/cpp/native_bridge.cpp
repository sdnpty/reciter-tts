#include <jni.h>
#include <android/log.h>

#define LOG_TAG "Qwen3Native"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

extern "C" JNIEXPORT jstring JNICALL
Java_com_qwen3_tts_engine_inference_QwenInferenceEngine_nativeGetVersion(JNIEnv* env, jobject thiz) {
    LOGI("Native bridge version 1.0");
    return env->NewStringUTF("Qwen3-TTS-Android-v1.0");
}
