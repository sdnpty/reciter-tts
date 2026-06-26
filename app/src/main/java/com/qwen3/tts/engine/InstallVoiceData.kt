package com.qwen3.tts.engine

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import com.qwen3.tts.util.ModelConfig

class InstallVoiceData : Activity() {
    companion object {
        private const val TAG = "InstallVoiceData"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Log.d(TAG, "Install voice data requested")

        val modelsDir = ModelConfig.getModelsDirOrCreate(this)

        Toast.makeText(
            this,
            "Please copy ONNX models to: ${modelsDir.absolutePath}",
            Toast.LENGTH_LONG
        ).show()

        // Open file manager
        try {
            val intent = Intent(Intent.ACTION_VIEW).apply {
                setDataAndType(android.net.Uri.parse(modelsDir.absolutePath), "resource/folder")
            }
            startActivity(intent)
        } catch (e: Exception) {
            Log.w(TAG, "No file manager available to open models directory", e)
        }

        finish()
    }
}
