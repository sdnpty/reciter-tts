package com.qwen3.tts.engine.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import com.qwen3.tts.util.AudioHelper
import com.qwen3.tts.util.TTSLogger
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.util.Random
import kotlin.math.cos

/**
 * F5-TTS engine: a NON-autoregressive flow-matching model. A DiT integrates an
 * ODE over [nfe] steps to turn noise into a target mel conditioned on a baked
 * reference voice (mel + transcript) and the target text; a Vocos vocoder turns
 * that mel into a 24 kHz waveform.
 *
 * Cost is `nfe` DiT forwards over the whole utterance (here CFG is folded into
 * the graph, so one onnx call per step), NOT 17 forwards per 12 ms frame like
 * the AR talker — which is why it is the path for near-real-time on a phone.
 *
 * Selected when the active model manifest's `architecture` is "f5".
 * Several contracts (tokenizer, mel, CFG, duration) are version-sensitive and
 * must be confirmed against `f5_ref_privet.wav` from the export session.
 */
class F5InferenceEngine private constructor(
    private val context: Context,
    private val modelDir: File
) : SpeechSynthesizer {

    companion object {
        private const val TAG = "F5Inference"

        fun create(context: Context, modelDir: File): F5InferenceEngine? {
            val needed = listOf("f5_dit.onnx", "f5_vocos.onnx", "vocab.json")
            if (needed.any { !File(modelDir, it).exists() }) return null
            return F5InferenceEngine(context, modelDir)
        }
    }

    private val env = OrtEnvironment.getEnvironment()
    private val logger = TTSLogger.getInstance(context)

    // config (mirrors f5_config.json from the export)
    private var sr = 24000
    private var nMels = 100
    private var nfe = 16
    private var cfg = 2.0f          // documented; actual CFG is baked into the graph
    private var sway = -1.0f
    private var speed = 1.0f

    override var sampleRateHz: Int = 24000; private set

    private var dit: OrtSession? = null
    private var vocos: OrtSession? = null
    private var vocab: Map<String, Int> = emptyMap()

    // baked voices: id -> (ref mel [T,M] row-major, ref_text)
    private class Ref(val mel: FloatArray, val frames: Int, val refText: String)
    private val refs = LinkedHashMap<String, Ref>()

    @Volatile private var stopRequested = false
    override fun requestStop() { stopRequested = true }

    init {
        loadConfig()
        loadVocab()
        loadVoices()
        dit = createSession("f5_dit.onnx")
        vocos = createSession("f5_vocos.onnx")
    }

    private fun createSession(name: String): OrtSession? {
        val f = File(modelDir, name)
        val cores = Runtime.getRuntime().availableProcessors().coerceIn(1, 4)
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(cores); setInterOpNumThreads(1)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        logger.beginLoadStage("Loading ${f.name} (${f.length() / 1024 / 1024} MB)")
        return try {
            env.createSession(f.absolutePath, opts).also {
                logger.endLoadStage()
                logger.i(TAG, "${f.name} ready; in=${it.inputNames} out=${it.outputNames}")
            }
        } catch (e: Throwable) {
            logger.endLoadStage()
            logger.e(TAG, "Failed to load ${f.name}: ${e.message}", Exception(e)); null
        }
    }

    private fun loadConfig() {
        val f = File(modelDir, "f5_config.json")
        if (!f.exists()) return
        runCatching {
            val o = JSONObject(f.readText())
            sr = o.optInt("sample_rate", sr); sampleRateHz = sr
            nMels = o.optInt("n_mels", nMels)
            nfe = o.optInt("nfe", nfe)
            cfg = o.optDouble("cfg_strength", cfg.toDouble()).toFloat()
            sway = o.optDouble("sway_coef", sway.toDouble()).toFloat()
            speed = o.optDouble("speed", speed.toDouble()).toFloat()
        }
    }

    private fun loadVocab() {
        val f = File(modelDir, "vocab.json")
        runCatching {
            val o = JSONObject(f.readText())
            val m = HashMap<String, Int>(o.length())
            val it = o.keys(); while (it.hasNext()) { val k = it.next(); m[k] = o.getInt(k) }
            vocab = m
            logger.i(TAG, "F5 vocab: ${vocab.size} tokens")
        }
    }

    private fun loadVoices() {
        val js = File(modelDir, "f5_voices.json")
        val bin = File(modelDir, "f5_voices.bin")
        if (!js.exists() || !bin.exists()) { logger.w(TAG, "no baked F5 voices"); return }
        runCatching {
            val raw = bin.readBytes()
            val fb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer() // fp16
            val arr = JSONObject(js.readText()).getJSONArray("voices")
            for (i in 0 until arr.length()) {
                val v = arr.getJSONObject(i)
                val frames = v.getInt("n_frames")
                val offset = v.getInt("offset")
                val n = frames * nMels
                val mel = FloatArray(n)
                for (k in 0 until n) mel[k] = h2f(fb.get(offset + k).toInt() and 0xffff)
                refs[v.getString("id")] = Ref(mel, frames, v.optString("ref_text", ""))
            }
            logger.i(TAG, "F5 baked voices: ${refs.keys}")
        }.onFailure { logger.e(TAG, "voice load failed: ${it.message}", Exception(it)) }
    }

    private fun h2f(h: Int): Float {
        val s = (h ushr 15) and 0x1; val e = (h ushr 10) and 0x1f; val m = h and 0x3ff
        val bits = when {
            e == 0 -> if (m == 0) s shl 31 else {
                var mm = m; var ee = -1; do { mm = mm shl 1; ee++ } while (mm and 0x400 == 0)
                (s shl 31) or ((127 - 15 - ee) shl 23) or ((mm and 0x3ff) shl 13)
            }
            e == 31 -> (s shl 31) or (0xff shl 23) or (m shl 13)
            else -> (s shl 31) or ((e - 15 + 127) shl 23) or (m shl 13)
        }
        return Float.fromBits(bits)
    }

    override fun isReady(): Boolean = dit != null && vocos != null && vocab.isNotEmpty()

    /** char -> id via the F5 vocab; unknown chars map to 0 (filler). */
    private fun encode(text: String): LongArray =
        LongArray(text.length) { (vocab[text[it].toString()] ?: 0).toLong() }

    override fun synthesize(
        text: String, voiceName: String, speed: Int, onAudioChunk: (ByteArray?) -> Boolean
    ) {
        stopRequested = false
        if (!isReady()) { onAudioChunk(null); return }
        try {
            for (chunk in splitSentences(text)) {
                if (stopRequested) break
                val pcm = synthOne(chunk, voiceName) ?: continue
                val paced = if (speed in 1..1000 && speed != 100)
                    AudioHelper.timeStretch(pcm, speed / 100f, sampleRateHz) else pcm
                if (!onAudioChunk(AudioHelper.floatToPcm16(paced))) break
            }
        } catch (e: Throwable) {
            logger.e(TAG, "F5 synthesis failed: ${e.message}", Exception(e)); onAudioChunk(null)
        }
    }

    private fun synthOne(genText: String, voiceName: String): FloatArray? {
        val ref = refs[voiceName] ?: refs.values.firstOrNull() ?: run {
            logger.e(TAG, "no reference voice"); return null
        }
        val M = nMels
        val tRef = ref.frames
        val refChars = ref.refText.length.coerceAtLeast(1)
        val genChars = genText.length.coerceAtLeast(1)
        // F5 duration heuristic: target frames proportional to text length ratio.
        val genFrames = (tRef.toFloat() / refChars * genChars / speed).toInt().coerceIn(8, 4096)
        val total = tRef + genFrames
        val fullText = ref.refText + genText
        val textIds = encode(fullText)

        // cond: ref mel at the front, zeros after. x: gaussian noise.
        val cond = FloatArray(total * M)
        System.arraycopy(ref.mel, 0, cond, 0, tRef * M)
        val rng = Random(1234)
        val x = FloatArray(total * M) { rng.nextGaussian().toFloat() }

        val tSched = swaySchedule(nfe, sway)
        val t0 = System.currentTimeMillis()

        val condT = tensor3(cond, total, M)
        val textT = OnnxTensor.createTensor(env, LongBuffer.wrap(textIds), longArrayOf(1, textIds.size.toLong()))
        try {
            for (i in 0 until nfe) {
                if (stopRequested) break
                val dt = tSched[i + 1] - tSched[i]
                val xT = tensor3(x, total, M)
                val timeT = OnnxTensor.createTensor(env, FloatBuffer.wrap(floatArrayOf(tSched[i])), longArrayOf(1))
                try {
                    val res = dit!!.run(mapOf("x" to xT, "cond" to condT, "text" to textT, "time" to timeT))
                    try {
                        val vel = (res[0] as OnnxTensor).floatBuffer
                        for (k in x.indices) x[k] += dt * vel.get(k)
                    } finally { res.close() }
                } finally { xT.close(); timeT.close() }
                if (i % 4 == 0) logger.i(TAG, "ode step $i/$nfe  ${System.currentTimeMillis() - t0}ms")
            }
        } finally { condT.close(); textT.close() }

        // generated mel = x[tRef:total] as [1, M, genFrames] (vocos layout)
        val genMel = FloatArray(M * genFrames)
        for (f in 0 until genFrames) for (m in 0 until M)
            genMel[m * genFrames + f] = x[(tRef + f) * M + m]
        val melT = tensor3to(genMel, M, genFrames)
        return try {
            val res = vocos!!.run(mapOf((vocos!!.inputNames.first()) to melT))
            try { extractAudio((res[0] as OnnxTensor)) } finally { res.close() }
        } finally { melT.close() }
    }

    /** sway-sampled time schedule t[0..nfe], t0=0..1. */
    private fun swaySchedule(steps: Int, coef: Float): FloatArray {
        val t = FloatArray(steps + 1)
        for (i in 0..steps) {
            val u = i.toFloat() / steps
            t[i] = u + coef * (cos(Math.PI.toFloat() / 2f * u) - 1f + u)
        }
        return t
    }

    private fun tensor3(data: FloatArray, t: Int, m: Int): OnnxTensor =
        OnnxTensor.createTensor(env, FloatBuffer.wrap(data), longArrayOf(1, t.toLong(), m.toLong()))

    private fun tensor3to(data: FloatArray, m: Int, t: Int): OnnxTensor =
        OnnxTensor.createTensor(env, FloatBuffer.wrap(data), longArrayOf(1, m.toLong(), t.toLong()))

    private fun extractAudio(t: OnnxTensor): FloatArray {
        val fb = t.floatBuffer ?: return floatArrayOf()
        val out = FloatArray(fb.remaining()); fb.get(out); return out
    }

    private fun splitSentences(text: String): List<String> {
        val parts = text.split(Regex("(?<=[.!?…\\n])\\s+")).map { it.trim() }.filter { it.isNotEmpty() }
        return if (parts.isEmpty()) listOf(text.trim()) else parts
    }

    override fun release() {
        dit?.close(); vocos?.close(); dit = null; vocos = null
    }
}
