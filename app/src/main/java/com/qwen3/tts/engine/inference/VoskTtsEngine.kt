package com.qwen3.tts.engine.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import com.qwen3.tts.util.AudioHelper
import com.qwen3.tts.util.TTSLogger
import org.json.JSONObject
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * Vosk-TTS (alphacep) — русский многоголосый VITS со словарными ударениями.
 *
 * Порт текстового фронтенда vosk_tts (путь `g2p_noembed`, модели БЕЗ BERT,
 * например vosk-model-tts-ru-0.7-multi): словарь `dictionary`
 * («слово вероятность фонемы…») + правила G2P для незнакомых слов, затем
 * phoneme_id_map из config.json с чередованием blank(0).
 *
 * Граф: input[1,T] int64, input_lengths[1] int64, scales[3] float32
 * (noise, 1/rate, duration_noise), sid[1] int64 → waveform float32 @22050.
 * Выбирается при `architecture` = "vosk-vits" в манифесте.
 */
class VoskTtsEngine private constructor(
    private val context: Context,
    private val modelDir: File,
) : SpeechSynthesizer {

    companion object {
        private const val TAG = "VoskTts"

        fun create(context: Context, modelDir: File): VoskTtsEngine? {
            val e = VoskTtsEngine(context, modelDir)
            return if (e.isReady()) e else null
        }

        // ── Порт g2p.py (alphacep/vosk-tts, Apache 2.0) ──────────
        private val SOFT_LETTERS = "яёюиье".toSet()
        private val START_SYL = "#ъьаяоёуюэеиы-".toSet()
        private val OTHERS = setOf('#', '+', '-', 'ь', 'ъ')
        private val SOFTHARD_CONS = mapOf(
            'б' to "b", 'в' to "v", 'г' to "g", 'д' to "d", 'з' to "z",
            'к' to "k", 'л' to "l", 'м' to "m", 'н' to "n", 'п' to "p",
            'р' to "r", 'с' to "s", 'т' to "t", 'ф' to "f", 'х' to "h",
        )
        private val OTHER_CONS = mapOf(
            'ж' to "zh", 'ц' to "c", 'ч' to "ch", 'ш' to "sh", 'щ' to "sch", 'й' to "j",
        )
        private val VOWELS = mapOf(
            'а' to "a", 'я' to "a", 'у' to "u", 'ю' to "u", 'о' to "o",
            'ё' to "o", 'э' to "e", 'е' to "e", 'и' to "i", 'ы' to "y",
        )

        /** г2п для слова вне словаря; вход может содержать '+' перед ударной гласной. */
        internal fun convert(stressWord: String): List<String> {
            val raw = "#$stressWord#"
            // (фонема-строка, ударение)
            val stressPhones = ArrayList<Pair<String, Int>>(raw.length)
            var stress = 0
            for (ch in raw) {
                if (ch == '+') { stress = 1; continue }
                stressPhones.add(ch.toString() to stress)
                stress = 0
            }
            // pallatize (как в python: последний символ не трогаем)
            for (i in 0 until stressPhones.size - 1) {
                val c = stressPhones[i].first[0]
                SOFTHARD_CONS[c]?.let { base ->
                    val soft = stressPhones[i + 1].first[0] in SOFT_LETTERS
                    stressPhones[i] = (if (soft) base + "j" else base) to 0
                }
                OTHER_CONS[c]?.let { stressPhones[i] = it to 0 }
            }
            // convert_vowels
            val out = ArrayList<String>(stressPhones.size + 4)
            var prev = ""
            for ((p, st) in stressPhones) {
                val c = p[0]
                if (prev.length == 1 && prev[0] in START_SYL && c in "яюеё") out.add("j")
                if (c in VOWELS) out.add(VOWELS[c] + st) else out.add(p)
                prev = p
            }
            return out.filter { !(it.length == 1 && it[0] in OTHERS) }
        }
    }

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val logger = TTSLogger.getInstance(context)
    private var session: OrtSession? = null
    @Volatile private var dic: Map<String, String> = emptyMap()
    private var phonemeIdMap: Map<String, Long> = emptyMap()
    private var noiseLevel = 0.8f
    private var speechRate = 1.0f
    private var durationNoiseLevel = 0.8f
    private var outScale = 1.0f
    override var sampleRateHz: Int = 22_050; private set

    @Volatile private var stopRequested = false
    override fun requestStop() { stopRequested = true }

    /** voice id -> speaker index (sid), заполняется билдером из манифеста. */
    var voiceSidMap: Map<String, Int> = emptyMap()

    init {
        try {
            loadConfig()
            loadSession()
            // Словарь (2M слов, десятки секунд на телефоне) грузим В ФОНЕ:
            // движок готов сразу, первые фразы идут через G2P-фолбэк (чуть
            // хуже ударения пару секунд), затем словарь подхватывается.
            Thread({
                try { loadDictionary() } catch (e: Throwable) {
                    logger.e(TAG, "dictionary load failed: ${e.message}", Exception(e))
                }
            }, "vosk-dict").start()
        } catch (e: Throwable) {
            logger.e(TAG, "vosk-tts init failed: ${e.message}", Exception(e))
        }
    }

    private fun loadConfig() {
        val f = File(modelDir, "config.json")
        if (!f.exists()) { logger.e(TAG, "config.json missing"); return }
        val json = JSONObject(f.readText())
        // phoneme_id_map: строка -> int (в моделях без BERT значения скалярные)
        json.optJSONObject("phoneme_id_map")?.let { m ->
            val map = HashMap<String, Long>(m.length())
            for (k in m.keys()) {
                val v = m.get(k)
                map[k] = when (v) {
                    is Int -> v.toLong()
                    is Long -> v
                    else -> (v as? org.json.JSONArray)?.optLong(0) ?: continue
                }
            }
            phonemeIdMap = map
        }
        json.optJSONObject("inference")?.let {
            noiseLevel = it.optDouble("noise_level", 0.8).toFloat()
            speechRate = it.optDouble("speech_rate", 1.0).toFloat()
            durationNoiseLevel = it.optDouble("duration_noise_level", 0.8).toFloat()
            outScale = it.optDouble("scale", 1.0).toFloat()
        }
        json.optJSONObject("audio")?.optInt("sample_rate", 0)?.takeIf { it > 0 }
            ?.let { sampleRateHz = it }
        if (json.has("model_type") || File(modelDir, "bert").isDirectory) {
            logger.w(TAG, "model declares BERT/multistream — this port supports the " +
                "no-BERT models (e.g. vosk-model-tts-ru-0.7-multi); output may be wrong")
        }
    }

    /** dictionary: `слово вероятность фонемы...`; при дублях берём максимум вероятности. */
    private fun loadDictionary() {
        val f = File(modelDir, "dictionary")
        if (!f.exists()) { logger.e(TAG, "dictionary missing"); return }
        val t0 = System.currentTimeMillis()

        // Бинарный кэш: текстовый разбор 2M строк на телефоне занимает десятки
        // секунд, готовые пары word/phones читаются в разы быстрее.
        val cache = File(modelDir, "dictionary.bin")
        if (cache.exists() && cache.lastModified() >= f.lastModified()) {
            try {
                java.io.DataInputStream(cache.inputStream().buffered(1 shl 16)).use { ins ->
                    val n = ins.readInt()
                    val map = HashMap<String, String>(n * 4 / 3 + 16)
                    repeat(n) { map[ins.readUTF()] = ins.readUTF() }
                    dic = map
                }
                logger.i(TAG, "vosk dictionary loaded from cache: ${dic.size} words " +
                    "in ${System.currentTimeMillis() - t0}ms")
                return
            } catch (e: Throwable) {
                logger.w(TAG, "dictionary.bin unreadable (${e.message}), reparsing")
                cache.delete()
            }
        }

        val map = HashMap<String, String>(1 shl 21)
        val probs = HashMap<String, Float>(1 shl 21)
        f.bufferedReader().useLines { lines ->
            for (line in lines) {
                // split(maxsplit=2): слово, вероятность, остальное — фонемы
                val i1 = line.indexOf(' '); if (i1 <= 0) continue
                val i2 = line.indexOf(' ', i1 + 1); if (i2 <= i1) continue
                val word = line.substring(0, i1)
                val prob = line.substring(i1 + 1, i2).toFloatOrNull() ?: continue
                if ((probs[word] ?: 0f) < prob) {
                    map[word] = line.substring(i2 + 1).trim()
                    probs[word] = prob
                }
            }
        }
        dic = map
        logger.i(TAG, "vosk dictionary parsed: ${dic.size} words in ${System.currentTimeMillis() - t0}ms")

        try {
            val tmp = File(modelDir, "dictionary.bin.tmp")
            java.io.DataOutputStream(tmp.outputStream().buffered(1 shl 16)).use { out ->
                out.writeInt(map.size)
                for ((w, p) in map) { out.writeUTF(w); out.writeUTF(p) }
            }
            if (!tmp.renameTo(cache)) tmp.delete()
        } catch (e: Throwable) {
            logger.w(TAG, "failed to write dictionary.bin: ${e.message}")
        }
    }

    private fun loadSession() {
        val f = File(modelDir, "model.onnx")
        if (!f.exists()) { logger.e(TAG, "model.onnx missing"); return }
        val cores = Runtime.getRuntime().availableProcessors().coerceIn(1, 4)
        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(cores)
            setInterOpNumThreads(1)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        logger.beginLoadStage("Loading vosk-tts (${f.length() / 1024 / 1024} MB) [CPU]")
        session = try {
            env.createSession(f.absolutePath, options).also {
                logger.endLoadStage()
                logger.i(TAG, "vosk-tts ready; inputs=${it.inputNames}")
            }
        } catch (e: Throwable) {
            logger.endLoadStage()
            logger.e(TAG, "vosk-tts session failed: ${e.message}", Exception(e)); null
        }
    }

    // Словарь НЕ обязателен для готовности: пока он грузится в фоне, слова
    // идут через G2P-фолбэк — старт мгновенный, ударения доуточнятся.
    override fun isReady(): Boolean =
        session != null && phonemeIdMap.isNotEmpty()

    override fun warmup() {
        // Первый прогон ORT платит за аллокацию арен; прогреваем на межд-
        // ометии, чтобы первый реальный запрос стартовал без этой пени.
        try {
            val ids = textToIds("а.")
            if (ids.isEmpty()) return
            val scales = floatArrayOf(noiseLevel, 1f, durationNoiseLevel)
            val s = session ?: return
            OnnxTensor.createTensor(env, LongBuffer.wrap(ids), longArrayOf(1, ids.size.toLong())).use { input ->
            OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(ids.size.toLong())), longArrayOf(1)).use { lens ->
            OnnxTensor.createTensor(env, FloatBuffer.wrap(scales), longArrayOf(3)).use { sc ->
            OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(0L)), longArrayOf(1)).use { sidT ->
                s.run(mapOf("input" to input, "input_lengths" to lens, "scales" to sc, "sid" to sidT)).close()
            }}}}
            logger.i(TAG, "warmup done")
        } catch (e: Throwable) {
            logger.w(TAG, "warmup failed: ${e.message}")
        }
    }

    // Пунктуация, которую модель знает как отдельные токены (как в synth.py).
    private val puncRegex = Regex("([,.?!;:\"() ])")

    /** Порт g2p_noembed: текст -> id фонем с blank(0) между ними. */
    internal fun textToIds(text: String): LongArray {
        val phonemes = ArrayList<String>(text.length)
        phonemes.add("^")
        // re.split с группой сохраняет разделители
        val lower = text.lowercase().replace("—", "-")
        val parts = ArrayList<String>()
        var last = 0
        for (m in puncRegex.findAll(lower)) {
            if (m.range.first > last) parts.add(lower.substring(last, m.range.first))
            parts.add(m.value)
            last = m.range.last + 1
        }
        if (last < lower.length) parts.add(lower.substring(last))

        for (word in parts) {
            if (word.isEmpty()) continue
            when {
                puncRegex.matches(word) || word == "-" -> phonemes.add(word)
                dic.containsKey(word) -> dic[word]!!.split(" ").filter { it.isNotEmpty() }
                    .forEach { phonemes.add(it) }
                else -> convert(word).forEach { phonemes.add(it) }
            }
        }
        phonemes.add("$")

        val ids = ArrayList<Long>(phonemes.size * 2)
        var unknown = 0
        fun idOf(p: String): Long? = phonemeIdMap[p] ?: run { unknown++; null }
        idOf(phonemes[0])?.let { ids.add(it) }
        for (i in 1 until phonemes.size) {
            val id = idOf(phonemes[i]) ?: continue
            ids.add(0L)   // blank
            ids.add(id)
        }
        if (unknown > 0) logger.w(TAG, "$unknown phonemes not in phoneme_id_map (skipped)")
        return ids.toLongArray()
    }

    private fun splitSentences(text: String): List<String> {
        val parts = text.split(Regex("(?<=[.!?…\\n])\\s+")).map { it.trim() }.filter { it.isNotEmpty() }
        return if (parts.isEmpty()) listOf(text.trim()) else parts
    }

    /**
     * Быстрый старт: у ПЕРВОГО предложения запроса отрезаем первую клаузу
     * (до запятой/точки с запятой/тире после ~15 символов) и синтезируем её
     * отдельно — звук начинается через доли секунды, как у облачных движков,
     * остальное догенерируется, пока клауза играет (RTF < 1).
     * Возвращает (текст, пауза-после?).
     */
    private fun planChunks(text: String): List<Pair<String, Boolean>> {
        val sentences = splitSentences(text)
        val out = ArrayList<Pair<String, Boolean>>(sentences.size + 1)
        sentences.forEachIndexed { idx, s ->
            if (idx == 0 && s.length > 40) {
                val cut = Regex("[,;:—–]\\s").find(s, 15)?.range?.last
                if (cut != null && cut < s.length - 10) {
                    out.add(s.substring(0, cut + 1).trim() to false)   // клауза, без паузы
                    out.add(s.substring(cut + 1).trim() to true)
                    return@forEachIndexed
                }
            }
            out.add(s to true)
        }
        return out
    }

    override fun synthesize(
        text: String, voiceName: String, speed: Int, onAudioChunk: (ByteArray?) -> Boolean
    ) {
        stopRequested = false
        val s = session ?: run { onAudioChunk(null); return }
        val sid = (voiceName.toIntOrNull() ?: voiceSidMap[voiceName] ?: 0).toLong()
        val rate = if (speed in 1..1000) speed / 100f else 1f
        val scales = floatArrayOf(noiseLevel, 1f / (speechRate * rate), durationNoiseLevel)
        logger.i(TAG, "synth voice='$voiceName' -> sid=$sid, rate=$rate")
        // ~120 мс тишины между предложениями: даёт книжный ритм и не даёт
        // AudioTrack «съедать» хвост фразы на стыке чанков.
        val pauseBytes = ByteArray(sampleRateHz * 2 * 120 / 1000)
        try {
            for ((sentence, pauseAfter) in planChunks(text)) {
                if (stopRequested) break
                // Словарь/латиница/числа: без нормализации латиница и цифры
                // молча выпадают из синтеза (их нет в phoneme_id_map).
                var s2 = com.qwen3.tts.util.RuNormalizer.normalize(sentence).trim()
                // VITS «проглатывает» конец фразы без завершающей пунктуации.
                if (s2.isNotEmpty() && s2.last() !in ".!?;:…,") s2 += "."
                val ids = textToIds(s2)
                if (ids.isEmpty()) continue
                OnnxTensor.createTensor(env, LongBuffer.wrap(ids), longArrayOf(1, ids.size.toLong())).use { input ->
                OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(ids.size.toLong())), longArrayOf(1)).use { lens ->
                OnnxTensor.createTensor(env, FloatBuffer.wrap(scales), longArrayOf(3)).use { sc ->
                OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(sid)), longArrayOf(1)).use { sidT ->
                    s.run(mapOf("input" to input, "input_lengths" to lens, "scales" to sc, "sid" to sidT)).use { result ->
                        @Suppress("UNCHECKED_CAST")
                        val value = result[0].value
                        val samples = flattenAudio(value)
                        if (samples.isNotEmpty()) {
                            if (outScale != 1.0f) for (i in samples.indices) samples[i] *= outScale
                            if (!onAudioChunk(AudioHelper.floatToPcm16(samples))) return
                            if (pauseAfter && !stopRequested && !onAudioChunk(pauseBytes)) return
                        }
                    }
                }}}}
            }
        } catch (e: Throwable) {
            logger.e(TAG, "vosk synthesis failed: ${e.message}", Exception(e)); onAudioChunk(null)
        }
    }

    /** Выход VITS бывает [T], [1,T] или [1,1,T] — приводим к плоскому массиву. */
    private fun flattenAudio(value: Any?): FloatArray = when (value) {
        is FloatArray -> value
        is Array<*> -> flattenAudio(value.firstOrNull())
        else -> FloatArray(0)
    }

    override fun release() { session?.close(); session = null }
}
