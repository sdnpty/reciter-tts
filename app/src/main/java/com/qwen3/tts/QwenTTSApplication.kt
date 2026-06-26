package com.qwen3.tts

import android.app.Application
import android.util.Log
import com.qwen3.tts.util.TTSLogger

class QwenTTSApplication : Application() {
    companion object {
        private const val TAG = "QwenTTSApp"
    }

    override fun onCreate() {
        super.onCreate()
        Log.d(TAG, "Reciter TTS Application started")
        setupUncaughtExceptionHandler()
    }

    private fun setupUncaughtExceptionHandler() {
        val defaultHandler = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { thread, throwable ->
            try {
                Log.e(TAG, "Uncaught exception on thread ${thread.name}", throwable)
                TTSLogger.getInstance(this).logCrash(throwable)
            } catch (_: Exception) {
                // Last resort — avoid crash in crash handler
            }
            defaultHandler?.uncaughtException(thread, throwable)
        }
    }
}
