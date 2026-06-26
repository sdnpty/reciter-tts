package com.qwen3.tts.ui

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import com.qwen3.tts.util.ModelConfig

class TTSViewModel(application: Application) : AndroidViewModel(application) {

    private val _modelsReady = MutableLiveData(ModelConfig.synthesisReady(application))
    val modelsReady: LiveData<Boolean> = _modelsReady

    fun refreshModelStatus() {
        _modelsReady.value = ModelConfig.synthesisReady(getApplication())
    }
}
