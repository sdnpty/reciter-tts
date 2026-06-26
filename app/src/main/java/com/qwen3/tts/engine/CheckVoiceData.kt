package com.qwen3.tts.engine

import android.app.Activity
import android.content.Intent
import android.speech.tts.TextToSpeech
import android.speech.tts.TextToSpeech.Engine
import android.util.Log
import com.qwen3.tts.util.ModelConfig

class CheckVoiceData : Activity() {
    companion object {
        private const val TAG = "CheckVoiceData"
    }

    override fun onStart() {
        super.onStart()
        Log.d(TAG, "Checking voice data")

        val missing = ModelConfig.missingSynthesisModels(this)
        val profile = ModelConfig.activeProfile(this)

        val result = if (missing.isEmpty()) {
            TextToSpeech.Engine.CHECK_VOICE_DATA_PASS
        } else {
            TextToSpeech.Engine.CHECK_VOICE_DATA_FAIL
        }

        val available = ArrayList(profile.voices.map {
            val l = it.toLocale()
            "${l.language}-${l.country}"
        })

        val returnIntent = Intent().apply {
            putStringArrayListExtra(Engine.EXTRA_AVAILABLE_VOICES, available)
            if (missing.isNotEmpty()) {
                putStringArrayListExtra(Engine.EXTRA_UNAVAILABLE_VOICES, ArrayList(missing))
            }
        }

        setResult(result, returnIntent)
        finish()
    }
}
