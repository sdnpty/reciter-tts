package com.qwen3.tts.engine

import android.content.Context
import android.media.AudioFormat
import android.speech.tts.SynthesisCallback
import android.speech.tts.SynthesisRequest
import android.speech.tts.TextToSpeech
import android.speech.tts.TextToSpeechService
import android.speech.tts.Voice
import android.util.Log
import com.qwen3.tts.engine.inference.CosyVoiceInferenceEngine
import com.qwen3.tts.engine.inference.QwenInferenceEngine
import com.qwen3.tts.engine.inference.SpeechSynthesizer
import com.qwen3.tts.engine.inference.VitsInferenceEngine
import com.qwen3.tts.util.ModelConfig
import com.qwen3.tts.util.TTSLogger
import java.util.Locale
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

class QwenTTSEngine : TextToSpeechService() {
    companion object {
        private const val TAG = "QwenTTSEngine"
        private const val CHANNELS = 1
        private const val PREFS = "qwen3_tts_prefs"
        private const val INIT_TIMEOUT_SEC = 120L
    }

    @Volatile
    private var synthesizer: SpeechSynthesizer? = null
    private lateinit var logger: TTSLogger
    private val isInitialized = AtomicBoolean(false)
    private val stopRequested = AtomicBoolean(false)
    // Lets a synthesis request that arrives during startup wait for model load
    // instead of failing — model loading is heavy and must not block onCreate.
    private val initLatch = CountDownLatch(1)

    override fun onCreate() {
        super.onCreate()
        logger = TTSLogger.getInstance(this)
        Log.i(TAG, "onCreate: Initializing Reciter TTS Engine")
        logger.i(TAG, "Initializing Reciter TTS engine")

        // A leftover load checkpoint means the previous attempt crashed the
        // process natively (no JVM exception) — surface it instead of silence.
        logger.consumeStaleLoadCrash()?.let { stage ->
            logger.e(TAG, "Previous startup crashed during: \"$stage\". " +
                "If this repeats, disable NNAPI in Settings or free up device memory.")
        }

        // Heavy native model loading MUST stay off the main thread, otherwise a
        // slow load triggers an ANR and a native OOM takes the whole app down.
        Thread({ initEngine() }, "tts-engine-init").apply { isDaemon = true; start() }
    }

    private fun initEngine() {
        try {
            ModelConfig.getModelsDirOrCreate(this)
            val profile = ModelConfig.activeProfile(this)

            val missing = profile.modelFiles
                .filter { it.requiredForSynthesis }
                .map { ModelConfig.getModelFile(this, it.filename) }
                .filter { !it.exists() }
                .map { it.name }
            if (missing.isNotEmpty()) {
                Log.e(TAG, "Required ONNX models missing: ${missing.joinToString()}")
                logger.e(TAG, "Required ONNX models missing: ${missing.joinToString()}")
                return
            }

            val useNnapi = getSharedPreferences(PREFS, Context.MODE_PRIVATE)
                .getString("device", "cpu") == "nnapi"
            logger.i(TAG, "Loading '${profile.id}' (arch=${profile.architecture}, EP=${if (useNnapi) "NNAPI" else "CPU"})")

            val built: SpeechSynthesizer? = when (profile.architecture.lowercase()) {
                "qwen3-codec-ar", "qwen3-tts-ar" -> buildArEngine()
                "qwen3-codec", "" -> buildQwenEngine(profile, useNnapi)
                "vits" -> buildVitsEngine(profile)
                "cosyvoice" -> CosyVoiceInferenceEngine(this, profile, profile.sampleRateHz)
                else -> {
                    logger.e(TAG, "Architecture '${profile.architecture}' is not implemented yet — see docs/MODELS.md")
                    null
                }
            }
            synthesizer = built
            if (built != null && built.isReady()) {
                isInitialized.set(true)
                logger.i(TAG, "Engine initialized successfully")
            } else {
                logger.e(TAG, "Engine created but not ready for synthesis")
            }
        } catch (e: Throwable) {
            Log.e(TAG, "Failed to initialize engine", e)
            logger.e(TAG, "Failed to initialize engine: ${e.message}", Exception(e))
            isInitialized.set(false)
        } finally {
            initLatch.countDown()
        }
    }

    /** New autoregressive Qwen3-TTS engine (talker_step/subtalker_step + baked voices). */
    private fun buildArEngine(): SpeechSynthesizer? =
        com.qwen3.tts.engine.inference.QwenArEngine.create(this, ModelConfig.activeModelDir(this))

    private fun buildQwenEngine(profile: ModelConfig.ModelProfile, useNnapi: Boolean): SpeechSynthesizer {
        val byRole = profile.modelFiles.associateBy { it.role }
        fun pathFor(role: ModelConfig.Role, fallback: String): String =
            ModelConfig.getModelFile(this, byRole[role]?.filename ?: fallback).absolutePath
        return QwenInferenceEngine(
            context = this,
            talkerPath = pathFor(ModelConfig.Role.TALKER, "talker_base_android.onnx"),
            codePredictorPath = pathFor(ModelConfig.Role.CODE_PREDICTOR, "code_predictor_base_android.onnx"),
            code2wavPath = pathFor(ModelConfig.Role.VOCODER, "code2wav_android.onnx"),
            speakerEncoderPath = pathFor(ModelConfig.Role.SPEAKER_ENCODER, "speaker_encoder_android.onnx"),
            useNnapi = useNnapi,
            audioTokenStart = profile.audioTokenStart,
            eosTokenId = profile.eosTokenId,
            sampleRateHz = profile.sampleRateHz
        )
    }

    private fun buildVitsEngine(profile: ModelConfig.ModelProfile): SpeechSynthesizer {
        val modelFile = profile.modelFiles.firstOrNull { it.requiredForSynthesis }
            ?: profile.modelFiles.first()
        return VitsInferenceEngine(
            context = this,
            modelPath = ModelConfig.getModelFile(this, modelFile.filename).absolutePath,
            vocabPath = ModelConfig.getModelFile(this, "vocab.json").absolutePath,
            sampleRateHz = profile.sampleRateHz
        )
    }

    override fun onDestroy() {
        super.onDestroy()
        synthesizer?.release()
        synthesizer = null
        Log.i(TAG, "onDestroy: Engine released")
    }

    override fun onStop() {
        stopRequested.set(true)
        Log.d(TAG, "onStop: Stop requested")
    }

    // Voices and languages are derived from the active model profile/manifest,
    // so adding a model with new languages requires no engine code changes.
    private fun profile() = ModelConfig.activeProfile(this)

    /** Model voices plus the user's recorded (cloned) voices, so both are
     *  selectable through the standard Android TextToSpeech voice API. */
    private fun allVoiceSpecs(): List<ModelConfig.VoiceSpec> =
        profile().voices + com.qwen3.tts.util.CustomVoiceStore.list(this).map { it.toVoiceSpec() }

    private fun voiceSpecById(id: String?): ModelConfig.VoiceSpec? =
        allVoiceSpecs().firstOrNull { it.id == id }

    private fun qualityOf(spec: ModelConfig.VoiceSpec): Int = when (spec.quality.lowercase()) {
        "very_high" -> Voice.QUALITY_VERY_HIGH
        "high" -> Voice.QUALITY_HIGH
        "normal" -> Voice.QUALITY_NORMAL
        "low" -> Voice.QUALITY_LOW
        else -> Voice.QUALITY_VERY_HIGH
    }

    override fun onGetLanguage(): Array<String> {
        val v = profile().defaultVoice()?.toLocale() ?: Locale("ru", "RU")
        return arrayOf(v.language, v.country, v.variant)
    }

    override fun onIsLanguageAvailable(lang: String, country: String, variant: String): Int {
        val voices = profile().voices
        val langMatch = voices.filter { it.toLocale().language.equals(lang, true) }
        return when {
            langMatch.isEmpty() -> TextToSpeech.LANG_NOT_SUPPORTED
            country.isNotEmpty() && langMatch.any { it.toLocale().country.equals(country, true) } ->
                TextToSpeech.LANG_COUNTRY_AVAILABLE
            else -> TextToSpeech.LANG_AVAILABLE
        }
    }

    override fun onLoadLanguage(lang: String, country: String, variant: String): Int {
        val availability = onIsLanguageAvailable(lang, country, variant)
        Log.d(TAG, "onLoadLanguage: $lang-$country-$variant -> $availability")
        return availability
    }

    override fun onGetVoices(): List<Voice> = allVoiceSpecs().map { spec ->
        Voice(spec.id, spec.toLocale(), qualityOf(spec), Voice.LATENCY_NORMAL, false, emptySet())
    }

    override fun onGetDefaultVoiceNameFor(lang: String, country: String, variant: String): String? {
        val voices = profile().voices
        return (voices.firstOrNull {
            it.toLocale().language.equals(lang, true) && it.toLocale().country.equals(country, true)
        } ?: voices.firstOrNull { it.toLocale().language.equals(lang, true) }
            ?: profile().defaultVoice())?.id
    }

    override fun onIsValidVoiceName(voiceName: String): Int =
        if (voiceSpecById(voiceName) != null) TextToSpeech.LANG_COUNTRY_AVAILABLE
        else TextToSpeech.LANG_NOT_SUPPORTED

    override fun onLoadVoice(voiceName: String): Int = onIsValidVoiceName(voiceName)

    override fun onSynthesizeText(request: SynthesisRequest, callback: SynthesisCallback) {
        stopRequested.set(false)

        // Model load runs in the background; the first request may arrive before
        // it finishes. Wait (bounded) rather than fail spuriously.
        if (!isInitialized.get()) {
            try {
                initLatch.await(INIT_TIMEOUT_SEC, TimeUnit.SECONDS)
            } catch (_: InterruptedException) {}
        }

        val engine = synthesizer
        if (!isInitialized.get() || engine == null || !engine.isReady()) {
            Log.e(TAG, "Engine not initialized")
            logger.e(TAG, "Synthesis rejected: engine not initialized")
            callback.error()
            return
        }

        val text = request.text ?: run {
            callback.error()
            return
        }

        if (text.isBlank()) {
            callback.done()
            return
        }

        val pitch = request.pitch
        val speed = request.speechRate
        val voiceName = request.voiceName
            ?: profile().voiceForLanguage(request.language ?: "")?.id
            ?: profile().defaultVoice()?.id
            ?: "qwen3_ru"

        Log.d(TAG, "Synthesize: text=${text.take(50)}, pitch=$pitch, speed=$speed, voice=$voiceName")
        logger.i(TAG, "Synthesis requested: ${text.take(50)}")

        val startResult = callback.start(engine.sampleRateHz, AudioFormat.ENCODING_PCM_16BIT, CHANNELS)
        if (startResult != TextToSpeech.SUCCESS) {
            Log.e(TAG, "Failed to start audio output: $startResult")
            callback.error()
            return
        }

        try {
            engine.synthesize(text, voiceName, speed) { audioChunk ->
                if (stopRequested.get()) {
                    return@synthesize false
                }
                if (audioChunk != null && audioChunk.isNotEmpty()) {
                    logger.i(TAG, "Streaming ${audioChunk.size} PCM bytes")
                    val maxBuffer = callback.maxBufferSize
                    var offset = 0
                    while (offset < audioChunk.size) {
                        val chunkSize = minOf(maxBuffer, audioChunk.size - offset)
                        val result = callback.audioAvailable(audioChunk, offset, chunkSize)
                        if (result != TextToSpeech.SUCCESS) {
                            Log.w(TAG, "audioAvailable failed with result: $result")
                            logger.w(TAG, "audioAvailable failed with result: $result")
                        }
                        offset += chunkSize
                    }
                }
                true
            }
            callback.done()
            logger.i(TAG, "Synthesis callback completed")
        } catch (e: Exception) {
            Log.e(TAG, "Synthesis error", e)
            logger.e(TAG, "Synthesis error", e)
            callback.error()
        }
    }
}
