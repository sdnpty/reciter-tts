package com.qwen3.tts.engine.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import com.qwen3.tts.util.AudioHelper
import com.qwen3.tts.util.TTSLogger
import org.json.JSONObject
import java.io.File
import java.nio.LongBuffer

/**
 * Single-shot VITS engine (MMS-TTS / Piper family): `input_ids` → `waveform`.
 *
 * EXPERIMENTAL. The graph contract is simple and well-defined, but the text
 * normalization/tokenization differs per model (char vocab, blank interleaving,
 * optional romanization/phonemization). This implements the common char-vocab +
 * add-blank scheme used by `transformers` `VitsTokenizer` (e.g. facebook/mms-tts-*);
 * verify on-device output for a given model. Selected when the manifest's
 * `architecture` is "vits".
 */
class VitsInferenceEngine(
    private val context: Context,
    private val modelPath: String,
    private val vocabPath: String,
    override val sampleRateHz: Int = 16_000,
    private val addBlank: Boolean = true,
    private val blankId: Long = 0L
) : SpeechSynthesizer {

    companion object { private const val TAG = "VitsInference" }

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val logger = TTSLogger.getInstance(context)
    private var session: OrtSession? = null
    private var vocab: Map<String, Int> = emptyMap()

    init {
        loadVocab()
        loadSession()
    }

    private fun loadVocab() {
        val f = File(vocabPath)
        if (!f.exists()) {
            logger.w(TAG, "VITS vocab.json not found: ${f.name}")
            return
        }
        try {
            val json = JSONObject(f.readText())
            val map = HashMap<String, Int>(json.length())
            val keys = json.keys()
            while (keys.hasNext()) {
                val k = keys.next()
                map[k] = json.getInt(k)
            }
            vocab = map
            logger.i(TAG, "VITS vocab loaded: ${vocab.size} symbols")
        } catch (e: Exception) {
            logger.e(TAG, "Failed to parse VITS vocab: ${e.message}", e)
        }
    }

    private fun loadSession() {
        val file = File(modelPath)
        if (!file.exists()) {
            logger.w(TAG, "VITS model not found: ${file.name}")
            return
        }
        val cores = Runtime.getRuntime().availableProcessors().coerceIn(1, 4)
        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(cores)
            setInterOpNumThreads(1)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        logger.beginLoadStage("Loading VITS session (${file.length() / 1024 / 1024} MB) [CPU]")
        session = try {
            env.createSession(modelPath, options).also {
                logger.endLoadStage()
                logger.i(TAG, "VITS session ready; inputs=${it.inputNames}, outputs=${it.outputNames}")
            }
        } catch (e: Throwable) {
            logger.endLoadStage()
            logger.e(TAG, "Failed to create VITS session: ${e.message}", Exception(e))
            null
        }
    }

    override fun isReady(): Boolean = session != null && vocab.isNotEmpty()

    /** Char-vocab tokenizer with optional blank interleaving (VITS add_blank). */
    private fun tokenize(text: String): LongArray {
        val ids = ArrayList<Long>(text.length * 2 + 1)
        fun push(id: Long) {
            if (addBlank && ids.isEmpty()) ids.add(blankId)
            ids.add(id)
            if (addBlank) ids.add(blankId)
        }
        for (ch in text.lowercase()) {
            val id = vocab[ch.toString()] ?: continue
            push(id.toLong())
        }
        return ids.toLongArray()
    }

    override fun synthesize(
        text: String,
        voiceName: String,
        speed: Int,
        onAudioChunk: (ByteArray?) -> Boolean
    ) {
        val s = session
        if (s == null || !isReady()) {
            logger.e(TAG, "VITS engine not ready")
            onAudioChunk(null)
            return
        }
        try {
            val ids = tokenize(text)
            if (ids.isEmpty()) {
                logger.w(TAG, "VITS produced no tokens for input")
                onAudioChunk(null)
                return
            }
            val shape = longArrayOf(1, ids.size.toLong())
            val idsTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(ids), shape)
            // Only create the mask tensor when the session actually has a mask input —
            // allocating it unconditionally means it leaks when session.run() throws
            // and it was never added to the inputs map.
            val names = s.inputNames
            val maskInputName = names.firstOrNull { it.contains("mask") }
            val maskTensor = if (maskInputName != null)
                OnnxTensor.createTensor(env, LongBuffer.wrap(LongArray(ids.size) { 1L }), shape)
            else null

            try {
                val inputs = HashMap<String, OnnxTensor>()
                inputs[names.firstOrNull { it.contains("input") } ?: names.first()] = idsTensor
                if (maskInputName != null && maskTensor != null) inputs[maskInputName] = maskTensor

                val results = s.run(inputs)
                val audio: FloatArray
                try {
                    audio = extractAudio(results[0].value)
                } finally {
                    results.close()
                }

                if (audio.isEmpty()) {
                    onAudioChunk(null)
                    return
                }
                val paced = if (speed in 1..1000 && speed != 100) {
                    AudioHelper.timeStretch(audio, speed / 100f, sampleRateHz)
                } else audio
                onAudioChunk(AudioHelper.floatToPcm16(paced))
            } finally {
                idsTensor.close()
                maskTensor?.close()
            }
        } catch (e: Exception) {
            logger.e(TAG, "VITS synthesis failed: ${e.message}", e)
            onAudioChunk(null)
        }
    }

    private fun extractAudio(raw: Any?): FloatArray = when (raw) {
        is FloatArray -> raw
        is Array<*> -> {
            // [B, samples] or [B, 1, samples]
            when (val first = raw.firstOrNull()) {
                is FloatArray -> first
                is Array<*> -> (first.firstOrNull() as? FloatArray) ?: floatArrayOf()
                else -> floatArrayOf()
            }
        }
        else -> {
            Log.e(TAG, "Unexpected VITS output type: ${raw?.javaClass}")
            floatArrayOf()
        }
    }

    override fun release() {
        session?.close()
        session = null
        Log.i(TAG, "VITS engine released")
    }
}
