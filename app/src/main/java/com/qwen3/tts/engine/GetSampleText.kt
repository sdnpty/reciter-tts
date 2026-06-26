package com.qwen3.tts.engine

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.speech.tts.TextToSpeech
import com.qwen3.tts.R

class GetSampleText : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val language = intent.getStringExtra("language") ?: "ru"
        val sampleText = when (language) {
            "ru" -> getString(R.string.sample_text_ru)
            "en" -> getString(R.string.sample_text_en)
            "zh" -> getString(R.string.sample_text_zh)
            else -> getString(R.string.sample_text_en)
        }

        val returnIntent = Intent().apply {
            putExtra(TextToSpeech.Engine.EXTRA_SAMPLE_TEXT, sampleText)
        }
        setResult(TextToSpeech.Engine.CHECK_VOICE_DATA_PASS, returnIntent)
        finish()
    }
}
