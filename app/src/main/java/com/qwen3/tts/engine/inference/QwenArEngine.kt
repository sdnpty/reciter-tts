package com.qwen3.tts.engine.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import com.qwen3.tts.engine.tokenizer.Qwen3Tokenizer
import com.qwen3.tts.util.AudioHelper
import org.json.JSONObject
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * Autoregressive Qwen3-TTS decoder, ported from the validated Python reference
 * (tools/infer_onnx_reference.py + tools/build_prefill.py — both bit-exact vs
 * model.generate). See docs/PIPELINE.md.
 *
 * Model dir must contain: talker_step.onnx, subtalker_step.onnx, codec_embed.onnx,
 * code2wav.onnx, text_cond_table.f16 [151936,1024], subtalker_codec_embed.f16
 * [15,2048,1024], subtalker_heads.f16 [15,2048,1024], and baked voice x-vectors.
 *
 * @param inputIdsFor must return the chat-templated token ids = role(3) + text + suffix(5).
 */
class QwenArEngine(
    modelDir: File,
    private val inputIdsFor: (String) -> IntArray,
    private val voices: Map<String, FloatArray>,   // name -> float32[1024]
) : SpeechSynthesizer {
    companion object {
        private const val TAG = "QwenAr"

        /**
         * Builds an AR engine for an installed model directory. Loads the baked
         * voice x-vectors (`baked_voices.bin` + `voices.json`) and wires the
         * tokenizer (`role + encodeForTTS(text) + suffix` from `ar_config.json`).
         * Returns null if the required AR files are missing.
         */
        fun create(context: Context, modelDir: File): QwenArEngine? {
            val required = listOf("talker_step.onnx", "subtalker_step.onnx",
                "codec_embed.onnx", "code2wav.onnx", "text_cond_table.f16",
                "subtalker_codec_embed.f16", "subtalker_heads.f16")
            if (required.any { !File(modelDir, it).exists() }) {
                Log.e(TAG, "AR model files missing in ${modelDir.name}")
                return null
            }

            val voices = loadVoices(modelDir)
            val cfg = runCatching { JSONObject(File(modelDir, "ar_config.json").readText()) }
                .getOrNull()
            val role = cfg?.optJSONArray("role_tokens").toIntList(intArrayOf(151644, 77091, 198))
            val suffix = cfg?.optJSONArray("suffix_tokens")
                .toIntList(intArrayOf(151645, 198, 151644, 77091, 198))
            val tokenizer = Qwen3Tokenizer.getInstance(context)

            val inputIdsFor: (String) -> IntArray = { text ->
                (role.toList() + tokenizer.encodeForTTS(text).toList() + suffix.toList())
                    .toIntArray()
            }
            return QwenArEngine(modelDir, inputIdsFor, voices)
        }

        /** Reads concatenated float32[1024] x-vectors + their names. */
        private fun loadVoices(modelDir: File): Map<String, FloatArray> {
            val bin = File(modelDir, "baked_voices.bin")
            val namesJson = File(modelDir, "voices.json")
            if (!bin.exists() || !namesJson.exists()) return emptyMap()
            val names = runCatching {
                val arr = JSONObject(namesJson.readText()).optJSONArray("voices")
                (0 until (arr?.length() ?: 0)).map { arr!!.getString(it) }
            }.getOrDefault(emptyList())
            if (names.isEmpty()) return emptyMap()
            val raw = bin.readBytes()
            val fb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            val out = LinkedHashMap<String, FloatArray>()
            for ((i, name) in names.withIndex()) {
                val v = FloatArray(H)
                val start = i * H
                if (start + H > fb.limit()) break
                for (d in 0 until H) v[d] = fb.get(start + d)
                out[name] = v
            }
            return out
        }

        private fun org.json.JSONArray?.toIntList(fallback: IntArray): IntArray {
            if (this == null) return fallback
            return IntArray(length()) { getInt(it) }
        }
        const val H = 1024
        const val TALKER_LAYERS = 28
        const val SUB_LAYERS = 5
        const val KVH = 8
        const val HD = 128
        const val GROUPS = 15
        const val EOS = 2150
        const val CODEBOOK0 = 2048
        const val VOCAB = 3072
        const val SR = 24000
        const val REP_PENALTY = 1.05f
        // token ids
        const val TTS_BOS = 151672; const val TTS_EOS = 151673; const val TTS_PAD = 151671
        const val C_PAD = 2148; const val C_BOS = 2149
        const val C_THINK = 2154; const val C_THINK_BOS = 2156; const val C_THINK_EOS = 2157
        const val LANG_RU = 2069
        const val MAX_FRAMES = 1500
    }

    private val env = OrtEnvironment.getEnvironment()
    private fun opts() = OrtSession.SessionOptions().apply {
        setIntraOpNumThreads(Runtime.getRuntime().availableProcessors().coerceIn(1, 4))
        setInterOpNumThreads(1)
    }
    private val talker = env.createSession(File(modelDir, "talker_step.onnx").absolutePath, opts())
    private val sub = env.createSession(File(modelDir, "subtalker_step.onnx").absolutePath, opts())
    private val cemb = env.createSession(File(modelDir, "codec_embed.onnx").absolutePath, opts())
    private val c2w = env.createSession(File(modelDir, "code2wav.onnx").absolutePath, opts())

    private val textCond = RandomAccessFile(File(modelDir, "text_cond_table.f16"), "r")
    private val subEmb = loadF16(File(modelDir, "subtalker_codec_embed.f16"), GROUPS * 2048 * H) // [15*2048*1024]
    private val subHead = loadF16(File(modelDir, "subtalker_heads.f16"), GROUPS * 2048 * H)

    // ── helpers ──────────────────────────────────────────────
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
    private fun loadF16(f: File, n: Int): FloatArray {
        val raw = f.readBytes(); val out = FloatArray(n)
        val bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN)
        for (i in 0 until n) out[i] = h2f(bb.short.toInt() and 0xffff)
        return out
    }
    /** one row (1024) of text_cond_table by token id. */
    private fun textCondRow(id: Int): FloatArray {
        textCond.seek(id.toLong() * H * 2)
        val b = ByteArray(H * 2); textCond.readFully(b)
        val bb = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray(H) { h2f(bb.short.toInt() and 0xffff) }
    }
    private fun t1(data: FloatArray, vararg shape: Long) =
        OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape)
    private fun tl(data: LongArray, vararg shape: Long) =
        OnnxTensor.createTensor(env, LongBuffer.wrap(data), shape)

    private fun codecEmbed(code: Int): FloatArray {
        val t = tl(longArrayOf(code.toLong()), 1, 1)
        val r = cemb.run(mapOf("codes" to t))
        @Suppress("UNCHECKED_CAST")
        val out = (r[0].value as Array<Array<FloatArray>>)[0][0]
        r.close(); t.close(); return out
    }

    private fun emptyKv(n: Int) = Array(2 * n) { t1(FloatArray(0), 1, KVH.toLong(), 0, HD.toLong()) }

    /** talker step → (logits, hidden, newKv). Closes the old kv tensors. */
    private fun talkerStep(emb: FloatArray, pos: Int, kv: Array<OnnxTensor>): Triple<FloatArray, FloatArray, Array<OnnxTensor>> {
        val feed = HashMap<String, OnnxTensor>()
        feed["inputs_embeds"] = t1(emb, 1, 1, H.toLong())
        feed["position_ids"] = tl(longArrayOf(pos.toLong(), pos.toLong(), pos.toLong()), 3, 1, 1)
        feed["cache_position"] = tl(longArrayOf(pos.toLong()), 1)
        for (i in 0 until TALKER_LAYERS) { feed["past_k_$i"] = kv[2 * i]; feed["past_v_$i"] = kv[2 * i + 1] }
        val r = talker.run(feed)
        @Suppress("UNCHECKED_CAST")
        val logits = (r[0].value as Array<Array<FloatArray>>)[0][0]
        @Suppress("UNCHECKED_CAST")
        val hidden = (r[1].value as Array<Array<FloatArray>>)[0][0]
        val newKv = Array(2 * TALKER_LAYERS) { r[2 + it].value.let { v -> @Suppress("UNCHECKED_CAST") (v as Array<Array<Array<FloatArray>>>); copyKv(v) } }
        r.close(); feed["inputs_embeds"]!!.close(); feed["position_ids"]!!.close(); feed["cache_position"]!!.close()
        kv.forEach { it.close() }
        return Triple(logits, hidden, newKv)
    }

    // present kv come back as ORT-managed; re-wrap into our own tensors for the next call
    private fun copyKv(v: Any): OnnxTensor {
        @Suppress("UNCHECKED_CAST")
        val a = v as Array<Array<Array<FloatArray>>>   // [1,KVH,L,HD]
        val L = a[0][0].size
        val buf = FloatArray(KVH * L * HD); var p = 0
        for (h in 0 until KVH) for (l in 0 until L) for (d in 0 until HD) buf[p++] = a[0][h][l][d]
        return t1(buf, 1, KVH.toLong(), L.toLong(), HD.toLong())
    }

    private fun subStep(emb: FloatArray, pos: Int, kv: Array<OnnxTensor>): Pair<FloatArray, Array<OnnxTensor>> {
        val feed = HashMap<String, OnnxTensor>()
        feed["inputs_embeds"] = t1(emb, 1, 1, H.toLong())
        feed["position_ids"] = tl(longArrayOf(pos.toLong()), 1, 1)
        feed["cache_position"] = tl(longArrayOf(pos.toLong()), 1)
        for (i in 0 until SUB_LAYERS) { feed["past_k_$i"] = kv[2 * i]; feed["past_v_$i"] = kv[2 * i + 1] }
        val r = sub.run(feed)
        @Suppress("UNCHECKED_CAST")
        val hidden = (r[0].value as Array<Array<FloatArray>>)[0][0]
        val newKv = Array(2 * SUB_LAYERS) { copyKv(r[1 + it].value) }
        r.close(); feed["inputs_embeds"]!!.close(); feed["position_ids"]!!.close(); feed["cache_position"]!!.close()
        kv.forEach { it.close() }
        return Pair(hidden, newKv)
    }

    private fun add(a: FloatArray, b: FloatArray): FloatArray = FloatArray(H) { a[it] + b[it] }

    private fun argmaxHead(hidden: FloatArray, g: Int): Int {
        // logits[v] = hidden · subHead[g][v]; subHead layout [g][v][H]
        val base = g * 2048 * H
        var best = 0; var bv = Float.NEGATIVE_INFINITY
        for (v in 0 until 2048) {
            var s = 0f; val off = base + v * H
            for (d in 0 until H) s += hidden[d] * subHead[off + d]
            if (s > bv) { bv = s; best = v }
        }
        return best
    }
    private fun subEmbRow(g: Int, code: Int): FloatArray {
        val off = (g * 2048 + code) * H
        return FloatArray(H) { subEmb[off + it] }
    }

    private fun selectCode0(logits: FloatArray, history: Set<Int>): Int {
        for (c in history) logits[c] = if (logits[c] > 0) logits[c] / REP_PENALTY else logits[c] * REP_PENALTY
        for (i in CODEBOOK0 until VOCAB) if (i != EOS) logits[i] = Float.NEGATIVE_INFINITY
        var best = 0; var bv = Float.NEGATIVE_INFINITY
        for (i in logits.indices) if (logits[i] > bv) { bv = logits[i]; best = i }
        return best
    }

    // ── prefill ──────────────────────────────────────────────
    private fun buildPrefill(ids: IntArray, xvec: FloatArray): Pair<List<FloatArray>, List<FloatArray>> {
        val ttsBos = textCondRow(TTS_BOS); val ttsEos = textCondRow(TTS_EOS); val ttsPad = textCondRow(TTS_PAD)
        fun codec(id: Int) = codecEmbed(id)
        val codecInput = ArrayList<FloatArray>()           // 7
        codecInput += codec(C_THINK); codecInput += codec(C_THINK_BOS); codecInput += codec(LANG_RU); codecInput += codec(C_THINK_EOS)
        codecInput += xvec
        codecInput += codec(C_PAD); codecInput += codec(C_BOS)
        val left = ArrayList<FloatArray>()                 // 6
        repeat(5) { left += ttsPad }; left += ttsBos
        val prefill = ArrayList<FloatArray>()
        for (i in 0..2) prefill += textCondRow(ids[i])     // role
        for (i in 0..5) prefill += add(left[i], codecInput[i])
        prefill += add(textCondRow(ids[3]), codecInput[6])
        val trailing = ArrayList<FloatArray>()
        for (i in 4 until ids.size - 5) trailing += textCondRow(ids[i])
        trailing += ttsEos
        return Pair(prefill, trailing)
    }

    // ── public ───────────────────────────────────────────────
    override val sampleRateHz: Int = SR

    override fun isReady(): Boolean = voices.isNotEmpty()

    override fun synthesize(
        text: String,
        voiceName: String,
        speed: Int,
        onAudioChunk: (ByteArray?) -> Boolean
    ) {
        try {
            var pcm = synthesizePcm(text, voiceName)
            if (speed != 100) {
                pcm = AudioHelper.timeStretch(pcm, speed / 100f, SR)
            }
            onAudioChunk(AudioHelper.floatToPcm16(pcm))
        } catch (e: Throwable) {
            Log.e(TAG, "AR synthesis failed: ${e.message}", e)
            onAudioChunk(null)
        }
    }

    private fun synthesizePcm(text: String, voiceName: String): FloatArray {
        val xvec = voices[voiceName] ?: voices.values.firstOrNull() ?: error("no voices")
        val ids = inputIdsFor(text)
        val (prefill, trailing) = buildPrefill(ids, xvec)

        var kv = emptyKv(TALKER_LAYERS)
        var logits = FloatArray(VOCAB); var hidden = FloatArray(H)
        for ((i, e) in prefill.withIndex()) {
            val (lg, hd, nk) = talkerStep(e, i, kv); logits = lg; hidden = hd; kv = nk
        }
        val history = HashSet<Int>()
        var code0 = selectCode0(logits, history)
        var pastHidden = hidden
        var pos = prefill.size; var step = 0
        val frames = ArrayList<IntArray>()
        while (code0 != EOS && frames.size < MAX_FRAMES) {
            val lastId = codecEmbed(code0)
            // subtalker
            var skv = emptyKv(SUB_LAYERS)
            run { val (_, n) = subStep(pastHidden, 0, skv); skv = n }
            var sh: FloatArray
            run { val (hh, n) = subStep(lastId, 1, skv); sh = hh; skv = n }
            val res = IntArray(GROUPS)
            res[0] = argmaxHead(sh, 0)
            for (g in 1 until GROUPS) {
                val (hh, n) = subStep(subEmbRow(g - 1, res[g - 1]), 1 + g, skv); sh = hh; skv = n
                res[g] = argmaxHead(sh, g)
            }
            skv.forEach { it.close() }
            frames += IntArray(16).also { it[0] = code0; for (g in 0 until GROUPS) it[g + 1] = res[g] }
            history += code0
            // next talker input
            var s = lastId
            for (g in 0 until GROUPS) s = add(s, subEmbRow(g, res[g]))
            s = add(s, if (step < trailing.size) trailing[step] else textCondRow(TTS_PAD))
            val (lg, hd, nk) = talkerStep(s, pos, kv); logits = lg; hidden = hd; kv = nk
            code0 = selectCode0(logits, history); pastHidden = hidden; pos++; step++
        }
        kv.forEach { it.close() }
        Log.i(TAG, "generated ${frames.size} frames")
        return code2wav(frames)
    }

    private fun code2wav(frames: List<IntArray>): FloatArray {
        val t = frames.size
        val codes = LongArray(16 * t)
        for (f in 0 until t) for (q in 0 until 16) codes[q * t + f] = frames[f][q].toLong()
        val ct = tl(codes, 1, 16, t.toLong())
        val r = c2w.run(mapOf("codec_tokens" to ct))
        @Suppress("UNCHECKED_CAST")
        val audio = (r[0].value as Array<Array<FloatArray>>)[0][0]
        r.close(); ct.close()
        return audio
    }

    override fun release() {
        listOf(talker, sub, cemb, c2w).forEach { runCatching { it.close() } }
        runCatching { textCond.close() }
    }
}
