package com.qwen3.tts.engine.inference

/**
 * Common contract for all TTS inference engines, so the [com.qwen3.tts.engine.QwenTTSEngine]
 * service can run different model architectures (Qwen3 codec, VITS, …) behind
 * one interface. Implementations are selected from the active model's
 * `architecture` field (see ModelConfig / docs/MODELS.md).
 */
interface SpeechSynthesizer {
    /** True when all critical sessions loaded and the engine can synthesize. */
    fun isReady(): Boolean

    /** Output PCM sample rate (Hz). */
    val sampleRateHz: Int

    /**
     * Synthesize [text] for [voiceName] at [speed] (100 = normal). Emits PCM-16
     * byte chunks via [onAudioChunk] (null on error). Return false from the
     * callback to stop early.
     */
    fun synthesize(
        text: String,
        voiceName: String,
        speed: Int,
        onAudioChunk: (ByteArray?) -> Boolean
    )

    /** Release native sessions and resources. */
    fun release()

    /**
     * Request that an in-flight [synthesize] abort as soon as possible. Default
     * is a no-op for engines that only stop between callback chunks.
     */
    fun requestStop() {}

    /**
     * Optionally pre-run the model once to prime native arenas, so the first
     * real request isn't penalized by allocation. Default is a no-op.
     */
    fun warmup() {}
}
