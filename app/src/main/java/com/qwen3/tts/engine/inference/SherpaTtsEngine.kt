package com.qwen3.tts.engine.inference

import android.content.Context
import com.k2fsa.sherpa.onnx.OfflineTts
import com.k2fsa.sherpa.onnx.OfflineTtsConfig
import com.k2fsa.sherpa.onnx.OfflineTtsKokoroModelConfig
import com.k2fsa.sherpa.onnx.OfflineTtsModelConfig
import com.k2fsa.sherpa.onnx.OfflineTtsVitsModelConfig
import com.qwen3.tts.util.AudioHelper
import com.qwen3.tts.util.TTSLogger
import java.io.File

/**
 * On-device TTS via sherpa-onnx (k2-fsa). Runs the same model families AudiFlo's
 * "Orion" uses — Kokoro-82M multilingual, Piper/VITS (native Russian), Matcha —
 * with built-in espeak-ng phonemization, so no text frontend to port. These are
 * small non-autoregressive models that run faster than real time on a phone CPU.
 *
 * Selected when the active model manifest's `architecture` starts with "sherpa-"
 * (sherpa-kokoro / sherpa-vits / sherpa-matcha). The model archive (from
 * sherpa-onnx releases, repackaged with a model.json) is extracted with its
 * directory tree intact (espeak-ng-data/ must survive).
 */
class SherpaTtsEngine private constructor(
    private val context: Context,
    private val modelDir: File,
    private val arch: String,
) : SpeechSynthesizer {

    companion object {
        private const val TAG = "SherpaTts"

        fun create(context: Context, modelDir: File, arch: String): SherpaTtsEngine? {
            val e = SherpaTtsEngine(context, modelDir, arch.lowercase())
            return if (e.tts != null) e else null
        }
    }

    private val logger = TTSLogger.getInstance(context)
    private var tts: OfflineTts? = null
    override var sampleRateHz: Int = 24000; private set

    @Volatile private var stopRequested = false
    override fun requestStop() { stopRequested = true }

    init {
        tts = build()
        tts?.let { sampleRateHz = it.sampleRate() }
    }

    /** Path of the first file in [modelDir] matching [pred], or "" if none. */
    private fun find(pred: (String) -> Boolean): String =
        modelDir.listFiles()?.firstOrNull { it.isFile && pred(it.name) }?.absolutePath ?: ""

    private fun path(name: String) = File(modelDir, name).let { if (it.exists()) it.absolutePath else "" }

    private fun build(): OfflineTts? {
        val tokens = path("tokens.txt")
        val dataDir = File(modelDir, "espeak-ng-data").let { if (it.isDirectory) it.absolutePath else "" }
        val cores = Runtime.getRuntime().availableProcessors().coerceIn(1, 4)

        val model = when {
            arch.contains("kokoro") -> OfflineTtsModelConfig(
                kokoro = OfflineTtsKokoroModelConfig(
                    model = path("model.onnx").ifEmpty { find { it.endsWith(".onnx") } },
                    voices = path("voices.bin"),
                    tokens = tokens,
                    dataDir = dataDir,
                    dictDir = path("dict"),
                    lexicon = modelDir.listFiles()
                        ?.filter { it.isFile && it.name.startsWith("lexicon") && it.name.endsWith(".txt") }
                        ?.joinToString(",") { it.absolutePath } ?: "",
                ),
                numThreads = cores, provider = "cpu",
            )
            else -> OfflineTtsModelConfig(   // vits / piper
                vits = OfflineTtsVitsModelConfig(
                    model = find { it.endsWith(".onnx") },
                    tokens = tokens,
                    dataDir = dataDir,
                    lexicon = path("lexicon.txt"),
                ),
                numThreads = cores, provider = "cpu",
            )
        }
        val config = OfflineTtsConfig(model = model, maxNumSentences = 1)
        logger.beginLoadStage("Loading sherpa-onnx ($arch)")
        return try {
            OfflineTts(assetManager = null, config = config).also {
                logger.endLoadStage()
                logger.i(TAG, "sherpa-onnx ready: $arch, sr=${it.sampleRate()}, speakers=${it.numSpeakers()}")
            }
        } catch (e: Throwable) {
            logger.endLoadStage()
            logger.e(TAG, "sherpa-onnx init failed: ${e.message}", Exception(e)); null
        }
    }

    override fun isReady(): Boolean = tts != null

    override fun synthesize(
        text: String, voiceName: String, speed: Int, onAudioChunk: (ByteArray?) -> Boolean
    ) {
        stopRequested = false
        val engine = tts ?: run { onAudioChunk(null); return }
        val sid = voiceName.toIntOrNull() ?: voiceSidMap[voiceName] ?: 0
        val rate = if (speed in 1..1000) speed / 100f else 1f
        try {
            val audio = engine.generate(text, sid, rate)
            val samples = audio.samples
            if (samples != null && samples.isNotEmpty()) {
                val chunkSize = 4096
                var offset = 0
                while (offset < samples.size && !stopRequested) {
                    val length = minOf(chunkSize, samples.size - offset)
                    val chunk = samples.copyOfRange(offset, offset + length)
                    val continueStreaming = onAudioChunk(AudioHelper.floatToPcm16(chunk))
                    if (!continueStreaming) {
                        break
                    }
                    offset += length
                }
            }
        } catch (e: Throwable) {
            logger.e(TAG, "sherpa synthesis failed: ${e.message}", Exception(e)); onAudioChunk(null)
        }
    }

    /** voice id -> speaker index, filled by the engine builder from the manifest. */
    var voiceSidMap: Map<String, Int> = emptyMap()

    override fun release() { tts?.release(); tts = null }
}
