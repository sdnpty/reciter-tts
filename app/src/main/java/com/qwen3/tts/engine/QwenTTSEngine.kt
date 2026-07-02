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
        // Process-wide lock: if two service instances start at once (e.g. the
        // system TTS and the in-app preview), serialize the heavy model load so
        // their peak allocations don't overlap and OOM the process.
        private val LOAD_LOCK = Any()
        // The engine holds ~480 MB of off-heap tables. Android may run several
        // TextToSpeechService instances at once (system TTS + in-app preview);
        // if each built its own engine the allocations doubled and OOMed. Share
        // ONE engine across all instances, keyed by what it was built from, and
        // keep it for the process lifetime.
        @Volatile
        private var sharedSynth: SpeechSynthesizer? = null
        private var sharedKey: String? = null
        // Serializes actual synthesis: the shared engine reuses internal buffers
        // and is not safe for concurrent decode. A second request (e.g. after a
        // not-yet-effective Stop) waits instead of corrupting state / OOMing.
        private val SYNTH_LOCK = Any()
    }

    private val synthesizer: SpeechSynthesizer? get() = sharedSynth
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
            val key = "${profile.id}|${ModelConfig.activeModelDir(this).absolutePath}|$useNnapi"

            var reused = false
            val built: SpeechSynthesizer? = synchronized(LOAD_LOCK) {
                // Reuse the already-loaded engine if it matches; only rebuild when
                // the model/EP actually changed (then free the old one first).
                val existing = sharedSynth
                if (existing != null && existing.isReady() && sharedKey == key) {
                    logger.i(TAG, "Reusing loaded '${profile.id}' (EP=${if (useNnapi) "NNAPI" else "CPU"})")
                    reused = true
                    existing
                } else {
                    if (existing != null && sharedKey != key) {
                        runCatching { existing.release() }
                        sharedSynth = null; sharedKey = null
                    }
                    logger.i(TAG, "Loading '${profile.id}' (arch=${profile.architecture}, EP=${if (useNnapi) "NNAPI" else "CPU"})")
                    val archLc = profile.architecture.lowercase()
                    val b = when {
                        archLc.startsWith("sherpa") -> buildSherpaEngine(profile, archLc)
                        archLc == "qwen3-codec-ar" || archLc == "qwen3-tts-ar" -> buildArEngine(useNnapi)
                        archLc == "qwen3-codec" || archLc == "" -> buildQwenEngine(profile, useNnapi)
                        archLc == "f5" -> buildF5Engine()
                        archLc == "vits" -> buildVitsEngine(profile)
                        archLc == "cosyvoice" -> CosyVoiceInferenceEngine(this, profile, profile.sampleRateHz)
                        else -> {
                            logger.e(TAG, "Architecture '${profile.architecture}' is not implemented yet — see docs/MODELS.md")
                            null
                        }
                    }
                    if (b != null && b.isReady()) { sharedSynth = b; sharedKey = key }
                    b
                }
            }
            if (built != null && built.isReady()) {
                // Prime native arenas BEFORE going live so the first real request
                // isn't penalized. Skip when reusing an already-warmed engine.
                if (!reused) runCatching { built.warmup() }
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
    private fun buildArEngine(useNnapi: Boolean): SpeechSynthesizer? =
        com.qwen3.tts.engine.inference.QwenArEngine.create(
            this, ModelConfig.activeModelDir(this), useNnapi
        )?.also { it.onLog = { m -> logger.i("QwenAr", m) } }

    /** F5-TTS (non-autoregressive flow-matching: DiT ODE loop + Vocos). */
    private fun buildF5Engine(): SpeechSynthesizer? =
        com.qwen3.tts.engine.inference.F5InferenceEngine.create(this, ModelConfig.activeModelDir(this))

    /** sherpa-onnx (Kokoro / Piper-VITS / Matcha) — fast non-AR, built-in g2p. */
    private fun buildSherpaEngine(profile: ModelConfig.ModelProfile, arch: String): SpeechSynthesizer? =
        com.qwen3.tts.engine.inference.SherpaTtsEngine
            .create(this, ModelConfig.activeModelDir(this), arch)
            ?.also { e ->
                e.voiceSidMap = profile.voices.associate { v -> v.id to v.speakerId }
                e.voiceLangMap = profile.voices.associate { v ->
                    v.id to com.qwen3.tts.engine.inference.SherpaTtsEngine.espeakLangOf(v.locale)
                }
            }

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
        // Do NOT release the shared engine here: other service instances may be
        // using it, and rebuilding it costs ~30 s + ~480 MB. It lives for the
        // process lifetime; the OS reclaims it when the process dies.
        stopRequested.set(true)
        sharedSynth?.requestStop()
        Log.i(TAG, "onDestroy: stop requested (shared engine kept)")
    }

    override fun onStop() {
        stopRequested.set(true)
        synthesizer?.requestStop()
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
            ?: "ru_male_1"

        Log.d(TAG, "Synthesize: text=${text.take(50)}, pitch=$pitch, speed=$speed, voice=$voiceName")
        logger.i(TAG, "Synthesis requested: ${text.take(50)}")

        val startResult = callback.start(engine.sampleRateHz, AudioFormat.ENCODING_PCM_16BIT, CHANNELS)
        if (startResult != TextToSpeech.SUCCESS) {
            Log.e(TAG, "Failed to start audio output: $startResult")
            callback.error()
            return
        }

        val tStart = System.currentTimeMillis()
        var totalPcmBytes = 0L
        // Serialize: the shared engine isn't safe for concurrent decode. If a
        // previous synthesis is still winding down after a Stop, wait for it.
        try {
          synchronized(SYNTH_LOCK) {
            if (stopRequested.get()) { callback.done(); return@synchronized }
            engine.synthesize(text, voiceName, speed) { audioChunk ->
                if (stopRequested.get()) {
                    return@synthesize false
                }
                if (audioChunk != null && audioChunk.isNotEmpty()) {
                    totalPcmBytes += audioChunk.size
                    val audioSec = totalPcmBytes / 2f / engine.sampleRateHz
                    val el = System.currentTimeMillis() - tStart
                    logger.i(TAG, "chunk ${audioChunk.size}B  total=${"%.2f".format(audioSec)}s audio in " +
                        "${el}ms  RTF=${"%.2f".format(el / 1000f / audioSec.coerceAtLeast(0.01f))}")
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
            val total = System.currentTimeMillis() - tStart
            val audioSec = totalPcmBytes / 2f / engine.sampleRateHz
            logger.i(TAG, "Synthesis done: ${"%.2f".format(audioSec)}s audio in ${total}ms " +
                "(RTF=${"%.2f".format(total / 1000f / audioSec.coerceAtLeast(0.01f))})")
          }
        } catch (e: Exception) {
            Log.e(TAG, "Synthesis error", e)
            logger.e(TAG, "Synthesis error", e)
            callback.error()
        }
    }
}
