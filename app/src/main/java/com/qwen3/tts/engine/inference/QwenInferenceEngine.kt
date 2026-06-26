package com.qwen3.tts.engine.inference

import android.content.Context
import android.util.Log
import ai.onnxruntime.*
import com.qwen3.tts.util.AudioHelper
import com.qwen3.tts.util.TTSLogger
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.LongBuffer
import kotlin.math.min

/**
 * ONNX inference engine for Qwen3-TTS.
 *
 * Pipeline (matches the corrected export script):
 *   1. Tokenize text → input_ids
 *   2. Talker (with lm_head): input_ids → logits → greedy‑decode audio token sequence
 *   3. Code Predictor: coarse codes → fine codes for all quantizers
 *   4. Code2Wav: codec_tokens [batch, 16, seq] → audio [batch, 1, samples]
 *   5. Float audio → PCM‑16 bytes
 *
 * The speaker encoder is loaded but currently used only when the caller supplies
 * a reference mel spectrogram; otherwise, a default speaker embedding is used.
 */
class QwenInferenceEngine(
    private val context: Context,
    private val talkerPath: String,
    private val codePredictorPath: String,
    private val code2wavPath: String,
    private val speakerEncoderPath: String,
    /**
     * Whether to attempt the NNAPI execution provider. Defaults to OFF: NNAPI
     * with INT8-quantized models aborts natively (SIGABRT) on many device
     * drivers, which is the most common cause of the engine "instantly
     * closing" with nothing in the logs. CPU is slower but stable everywhere.
     */
    private val useNnapi: Boolean = false,
    // Model-specific decode parameters (resolved from the active ModelProfile /
    // manifest, so different models of the family work without code changes).
    private val audioTokenStart: Int = 151_936,
    private val eosTokenId: Int = 151_645,
    override val sampleRateHz: Int = 24_000
) : SpeechSynthesizer {
    companion object {
        private const val TAG = "QwenInference"
        private const val MAX_GEN_LEN = 2048
        private const val NUM_QUANTIZERS = 16
        private const val CODEBOOK_SIZE = 2048
    }

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val logger = TTSLogger.getInstance(context)
    private var talkerSession: OrtSession? = null
    private var codePredictorSession: OrtSession? = null
    private var code2wavSession: OrtSession? = null
    private var speakerEncoderSession: OrtSession? = null

    private lateinit var tokenizer: com.qwen3.tts.engine.tokenizer.Qwen3Tokenizer

    // Model I/O metadata detected at load time
    private var talkerOutputName: String = "logits"
    private var talkerOutputDim: Int = -1
    private var cpOutputName: String = "logits"
    private var cpOutputDim: Int = -1

    init {
        initSessions()
        try {
            tokenizer = com.qwen3.tts.engine.tokenizer.Qwen3Tokenizer.getInstance(context)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize Qwen3Tokenizer", e)
        }
    }

    // ── Session setup ────────────────────────────────────────

    private fun cpuSessionOptions(): OrtSession.SessionOptions {
        // Keep memory/thread pressure modest: too many threads on a big INT8
        // model multiplies arena allocations and can trip the low-memory killer.
        val cores = Runtime.getRuntime().availableProcessors()
        val intraThreads = cores.coerceIn(1, 4)
        return OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(intraThreads)
            setInterOpNumThreads(1)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            // Memory-pattern pre-allocation balloons RAM for variable-length
            // autoregressive inputs; disabling it trades a little speed for a
            // much smaller, more predictable footprint.
            setMemoryPatternOptimization(false)
        }
    }

    private fun nnapiSessionOptions(): OrtSession.SessionOptions? {
        if (!useNnapi) return null
        return try {
            cpuSessionOptions().apply { addNnapi() }
        } catch (e: Throwable) {
            Log.w(TAG, "NNAPI requested but unavailable, falling back to CPU", e)
            logger.w(TAG, "NNAPI unavailable, using CPU: ${e.message}")
            null
        }
    }

    private fun initSessions() {
        logMemory("before model load")

        // Critical path first (Talker is the largest model, ~428 MB).
        talkerSession = safeCreateSession(talkerPath, cpuSessionOptions(), "Talker")
        // Vocoder is required for any audio at all.
        val code2wavOptions = nnapiSessionOptions() ?: cpuSessionOptions()
        code2wavSession = safeCreateSession(code2wavPath, code2wavOptions, "Code2Wav")
        // Optional models last — failure here degrades quality but never crashes.
        codePredictorSession = safeCreateSession(codePredictorPath, cpuSessionOptions(), "CodePredictor")
        speakerEncoderSession = safeCreateSession(speakerEncoderPath, cpuSessionOptions(), "SpeakerEncoder")

        logMemory("after model load")

        // Detect output metadata
        talkerSession?.let { session ->
            val info = session.outputInfo
            if (info.isNotEmpty()) {
                val entry = info.entries.first()
                talkerOutputName = entry.key
                val nodeInfo = entry.value.info
                if (nodeInfo is TensorInfo) {
                    val shape = nodeInfo.shape
                    if (shape.size >= 3) talkerOutputDim = shape[2].toInt()
                }
            }
            Log.i(TAG, "Talker inputs=${session.inputNames}, outputs=${session.outputNames}, outputDim=$talkerOutputDim")
        }

        codePredictorSession?.let { session ->
            val info = session.outputInfo
            if (info.isNotEmpty()) {
                val entry = info.entries.first()
                cpOutputName = entry.key
                val nodeInfo = entry.value.info
                if (nodeInfo is TensorInfo) {
                    val shape = nodeInfo.shape
                    if (shape.size >= 3) cpOutputDim = shape[2].toInt()
                }
            }
            Log.i(TAG, "CodePredictor inputs=${session.inputNames}, outputs=${session.outputNames}, outputDim=$cpOutputDim")
        }

        code2wavSession?.let { session ->
            Log.i(TAG, "Code2Wav inputs=${session.inputNames}, outputs=${session.outputNames}")
        }
    }

    private fun safeCreateSession(path: String, options: OrtSession.SessionOptions, label: String): OrtSession? {
        val file = File(path)
        if (!file.exists()) {
            Log.e(TAG, "$label model not found: $path")
            logger.w(TAG, "$label model not found, skipping: ${file.name}")
            return null
        }
        val sizeMb = file.length() / 1024 / 1024
        // Drop a synchronous breadcrumb so that if the native loader crashes
        // hard (no JVM exception), the next launch can tell us where it died.
        logger.beginLoadStage("Loading $label session (${sizeMb} MB) [${if (useNnapi) "NNAPI?" else "CPU"}]")
        return try {
            val session = env.createSession(path, options)
            logger.endLoadStage()
            Log.i(TAG, "$label session created (${sizeMb} MB)")
            logger.i(TAG, "$label session ready (${sizeMb} MB)")
            session
        } catch (e: Throwable) {
            // Catches Exception *and* Error (e.g. UnsatisfiedLinkError, OOM that
            // surfaced as a JVM Error). True native aborts can't be caught here,
            // but the breadcrumb above will still flag them.
            logger.endLoadStage()
            Log.e(TAG, "Failed to create $label session", e)
            logger.e(TAG, "Failed to create $label session: ${e.message}", e)
            null
        }
    }

    private fun logMemory(phase: String) {
        try {
            val am = context.getSystemService(Context.ACTIVITY_SERVICE) as android.app.ActivityManager
            val mi = android.app.ActivityManager.MemoryInfo()
            am.getMemoryInfo(mi)
            val availMb = mi.availMem / 1024 / 1024
            val totalMb = mi.totalMem / 1024 / 1024
            logger.i(TAG, "Memory $phase: ${availMb}MB free / ${totalMb}MB total${if (mi.lowMemory) " (LOW MEMORY)" else ""}")
        } catch (_: Exception) {}
    }

    // ── Public API ───────────────────────────────────────────

    override fun isReady(): Boolean =
        talkerSession != null && code2wavSession != null

    /**
     * Synchronous text‑to‑speech synthesis.
     *
     * @param onAudioChunk  Receives PCM‑16 byte arrays (or null on error). Return false to stop.
     */
    override fun synthesize(
        text: String,
        voiceName: String,
        speed: Int,
        onAudioChunk: (ByteArray?) -> Boolean
    ) {
        try {
            if (!isReady()) {
                Log.e(TAG, "Engine not ready — missing ONNX sessions")
                logger.e(TAG, "Engine not ready — missing ONNX sessions")
                onAudioChunk(null)
                return
            }

            if (!this::tokenizer.isInitialized) {
                Log.e(TAG, "Tokenizer not initialized")
                logger.e(TAG, "Tokenizer not initialized")
                onAudioChunk(null)
                return
            }

            // 1. Tokenize
            val inputIds = tokenizer.encodeForTTS(text)
            if (inputIds.isEmpty()) {
                Log.w(TAG, "Empty tokenization result")
                onAudioChunk(null)
                return
            }
            Log.d(TAG, "Tokenized ${text.take(30)}... → ${inputIds.size} tokens")
            logger.d(TAG, "Tokenized text into ${inputIds.size} tokens")

            // 2. Run Talker → generate coarse audio codes
            val coarseCodes = runTalkerAutoregressive(inputIds)
            if (coarseCodes.isEmpty()) {
                Log.w(TAG, "Talker produced no audio codes")
                logger.w(TAG, "Talker produced no audio codes")
                onAudioChunk(null)
                return
            }
            Log.d(TAG, "Talker generated ${coarseCodes.size} coarse codes")
            logger.d(TAG, "Talker generated ${coarseCodes.size} coarse codes")

            // 3. Run Code Predictor → expand to all quantizers
            val allCodes = runCodePredictorExpand(coarseCodes)
            Log.d(TAG, "Expanded to ${allCodes.size / NUM_QUANTIZERS} frames × $NUM_QUANTIZERS quantizers")

            // 4. Run Code2Wav → audio waveform
            val audio = runCode2Wav(allCodes)
            if (audio.isEmpty()) {
                Log.w(TAG, "Code2Wav produced no audio")
                logger.w(TAG, "Code2Wav produced no audio")
                onAudioChunk(null)
                return
            }
            Log.d(TAG, "Audio: ${audio.size} samples (${audio.size / sampleRateHz.toFloat()}s)")
            logger.i(TAG, "Generated ${audio.size} audio samples")

            // 5. Apply speech rate (Android sends 100 = normal) and convert to PCM-16
            val paced = if (speed in 1..1000 && speed != 100) {
                AudioHelper.timeStretch(audio, speed / 100f, sampleRateHz)
            } else audio
            val pcm = floatToPcm16(paced)
            onAudioChunk(pcm)

        } catch (e: Exception) {
            Log.e(TAG, "Synthesis failed", e)
            logger.e(TAG, "Synthesis failed", e)
            onAudioChunk(null)
        }
    }

    // ── Talker: autoregressive generation ────────────────────

    /**
     * Runs the Talker model autoregressively to generate coarse audio codec codes.
     *
     * If the model outputs "logits" (has lm_head), we sample/argmax the next token.
     * If it outputs "hidden_states" (no lm_head), we fall back to a simple projection.
     */
    private fun runTalkerAutoregressive(textTokenIds: IntArray): IntArray {
        val session = talkerSession ?: return intArrayOf()
        val hasLogits = talkerOutputName == "logits"

        if (!hasLogits) {
            Log.w(TAG, "Talker outputs hidden_states, not logits — model was exported without lm_head")
            return runTalkerSinglePass(textTokenIds)
        }

        val generated = mutableListOf<Int>()
        val contextTokens = textTokenIds.toMutableList()

        for (step in 0 until MAX_GEN_LEN) {
            val seqLen = contextTokens.size
            val idsLong = LongArray(seqLen) { contextTokens[it].toLong() }
            val mask = LongArray(seqLen) { 1L }

            val idsTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(idsLong), longArrayOf(1, seqLen.toLong()))
            val maskTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(mask), longArrayOf(1, seqLen.toLong()))

            // Use try-finally to guarantee tensor/result closure even if session.run() throws.
            var bestIdx = -1
            try {
                val results = session.run(mapOf("input_ids" to idsTensor, "attention_mask" to maskTensor))
                try {
                    val logitsRaw = results[0].value

                    // Extract logits for the last position
                    val lastLogits: FloatArray = when (logitsRaw) {
                        is Array<*> -> {
                            @Suppress("UNCHECKED_CAST")
                            val arr = logitsRaw as Array<Array<FloatArray>>
                            arr[0][seqLen - 1]
                        }
                        else -> {
                            Log.e(TAG, "Unexpected logits type: ${logitsRaw?.javaClass}")
                            break
                        }
                    }

                    // Greedy: argmax
                    var bestVal = Float.NEGATIVE_INFINITY
                    for (i in lastLogits.indices) {
                        if (lastLogits[i] > bestVal) {
                            bestVal = lastLogits[i]
                            bestIdx = i
                        }
                    }
                } finally {
                    results.close()
                }
            } finally {
                idsTensor.close()
                maskTensor.close()
            }

            if (bestIdx < 0) break

            // EOS check
            if (bestIdx == eosTokenId) break

            contextTokens.add(bestIdx)

            // Collect audio tokens
            if (bestIdx >= audioTokenStart) {
                generated.add(bestIdx - audioTokenStart)
            }
        }

        return generated.toIntArray()
    }

    /**
     * Called when the Talker only outputs hidden_states (no lm_head).
     * This model variant cannot generate audio codes directly — return empty
     * so the caller reports an error instead of producing garbage audio.
     */
    private fun runTalkerSinglePass(@Suppress("UNUSED_PARAMETER") inputIds: IntArray): IntArray {
        Log.e(TAG, "Talker model was exported without lm_head — cannot generate audio codes. " +
            "Re-export with include_lm_head=True or use a compatible model checkpoint.")
        logger.e(TAG, "Incompatible talker model: missing lm_head output layer")
        return intArrayOf()
    }

    // ── Code Predictor: expand coarse → full quantizers ──────

    /**
     * Takes coarse codes (quantizer 0) and predicts fine codes for quantizers 1..15.
     *
     * If the code predictor session is unavailable, duplicates coarse codes across all quantizers.
     */
    private fun runCodePredictorExpand(coarseCodes: IntArray): LongArray {
        val numFrames = coarseCodes.size
        val allCodes = LongArray(numFrames * NUM_QUANTIZERS)

        // Fill quantizer 0 with coarse codes
        for (i in 0 until numFrames) {
            allCodes[i * NUM_QUANTIZERS] = coarseCodes[i].toLong()
        }

        val session = codePredictorSession
        if (session == null) {
            Log.w(TAG, "CodePredictor not available — using coarse codes for all quantizers")
            for (i in 0 until numFrames) {
                for (q in 1 until NUM_QUANTIZERS) {
                    allCodes[i * NUM_QUANTIZERS + q] = coarseCodes[i].toLong()
                }
            }
            return allCodes
        }

        // Run code predictor for each additional quantizer
        val hasLogits = cpOutputName == "logits"
        val seqLen = numFrames
        val idsLong = LongArray(seqLen) { coarseCodes[it].toLong() }
        val mask = LongArray(seqLen) { 1L }

        val idsTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(idsLong), longArrayOf(1, seqLen.toLong()))
        val maskTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(mask), longArrayOf(1, seqLen.toLong()))
        try {
            val results = session.run(mapOf("input_ids" to idsTensor, "attention_mask" to maskTensor))
            try {
                val output = results[0].value

                if (hasLogits && output is Array<*>) {
                    @Suppress("UNCHECKED_CAST")
                    val logits = output as Array<Array<FloatArray>>
                    val outDim = logits[0][0].size

                    // If output dimension covers all quantizers (outDim ≈ CODEBOOK_SIZE * (NUM_QUANTIZERS-1))
                    val codesPerQuantizer = outDim / (NUM_QUANTIZERS - 1).coerceAtLeast(1)

                    for (t in 0 until seqLen) {
                        for (q in 1 until NUM_QUANTIZERS) {
                            val offset = (q - 1) * codesPerQuantizer
                            val end = min(offset + codesPerQuantizer, outDim)
                            if (offset >= outDim) {
                                allCodes[t * NUM_QUANTIZERS + q] = 0L
                                continue
                            }
                            var bestIdx = 0
                            var bestVal = Float.NEGATIVE_INFINITY
                            for (i in offset until end) {
                                if (logits[0][t][i] > bestVal) {
                                    bestVal = logits[0][t][i]
                                    bestIdx = i - offset
                                }
                            }
                            allCodes[t * NUM_QUANTIZERS + q] = bestIdx.toLong().coerceIn(0, CODEBOOK_SIZE.toLong() - 1)
                        }
                    }
                } else {
                    // hidden_states output — fill with zeros for fine quantizers
                    Log.w(TAG, "CodePredictor outputs hidden_states — fine codes will be zero")
                    for (i in 0 until numFrames) {
                        for (q in 1 until NUM_QUANTIZERS) {
                            allCodes[i * NUM_QUANTIZERS + q] = 0L
                        }
                    }
                }
            } finally {
                results.close()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Code predictor failed, using coarse codes", e)
            for (i in 0 until numFrames) {
                for (q in 1 until NUM_QUANTIZERS) {
                    allCodes[i * NUM_QUANTIZERS + q] = coarseCodes[i].toLong()
                }
            }
        } finally {
            idsTensor.close()
            maskTensor.close()
        }

        return allCodes
    }

    // ── Code2Wav ─────────────────────────────────────────────

    /**
     * Decodes codec tokens to audio waveform.
     *
     * Input shape: [batch=1, num_quantizers=16, seq_len]   (matches export)
     * Output shape: [batch=1, channels=1, samples]
     */
    private fun runCode2Wav(allCodes: LongArray): FloatArray {
        val session = code2wavSession ?: return floatArrayOf()
        val numFrames = allCodes.size / NUM_QUANTIZERS

        // Reshape from [frames * quantizers] interleaved to [quantizers, frames]
        // Export expects shape: (1, NUM_QUANTIZERS, numFrames)
        val reshapedCodes = LongArray(NUM_QUANTIZERS * numFrames)
        for (t in 0 until numFrames) {
            for (q in 0 until NUM_QUANTIZERS) {
                reshapedCodes[q * numFrames + t] = allCodes[t * NUM_QUANTIZERS + q]
            }
        }

        val codesTensor = OnnxTensor.createTensor(
            env, LongBuffer.wrap(reshapedCodes),
            longArrayOf(1, NUM_QUANTIZERS.toLong(), numFrames.toLong())
        )

        // Determine the input name from the session
        val inputName = session.inputNames.firstOrNull() ?: "codec_tokens"
        try {
            val results = session.run(mapOf(inputName to codesTensor))
            try {
                val audioRaw = results[0].value
                return when (audioRaw) {
                    is Array<*> -> {
                        @Suppress("UNCHECKED_CAST")
                        val arr = audioRaw as Array<Array<FloatArray>>
                        arr[0][0]
                    }
                    is FloatArray -> audioRaw
                    else -> {
                        Log.e(TAG, "Unexpected Code2Wav output type: ${audioRaw?.javaClass}")
                        floatArrayOf()
                    }
                }
            } finally {
                results.close()
            }
        } finally {
            codesTensor.close()
        }
    }

    // ── Utilities ────────────────────────────────────────────

    fun extractSpeakerId(voiceName: String): Int {
        return voiceName.filter { it.isDigit() }.toIntOrNull() ?: 0
    }

    private fun floatToPcm16(audio: FloatArray): ByteArray = AudioHelper.floatToPcm16(audio)

    override fun release() {
        talkerSession?.close()
        codePredictorSession?.close()
        code2wavSession?.close()
        speakerEncoderSession?.close()
        env.close()
        Log.i(TAG, "Engine released")
    }
}
