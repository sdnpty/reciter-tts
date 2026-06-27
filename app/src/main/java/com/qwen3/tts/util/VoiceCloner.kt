package com.qwen3.tts.util

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.util.Log
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Turns a recorded reference clip into a speaker x-vector on-device.
 *
 * The exported `speaker_encoder.onnx` bakes the Kaldi fbank frontend into the
 * graph, so it maps a raw 16 kHz mono waveform straight to the 1024-d x-vector
 * — no feature engineering is needed here. The vector is plug-compatible with
 * the baked voices consumed by [com.qwen3.tts.engine.inference.QwenArEngine].
 */
class VoiceCloner private constructor(private val session: OrtSession) {

    companion object {
        private const val TAG = "VoiceCloner"

        /** Opens the speaker encoder for [modelDir], or null if it isn't present. */
        fun forModel(modelDir: File): VoiceCloner? {
            val f = File(modelDir, "speaker_encoder.onnx")
            if (!f.exists()) return null
            return try {
                val env = OrtEnvironment.getEnvironment()
                val opts = OrtSession.SessionOptions().apply {
                    setIntraOpNumThreads(Runtime.getRuntime().availableProcessors().coerceIn(1, 4))
                }
                VoiceCloner(env.createSession(f.absolutePath, opts))
            } catch (e: Throwable) {
                Log.e(TAG, "Failed to open speaker_encoder.onnx: ${e.message}", e)
                null
            }
        }

        /** Reads a 16-bit PCM mono WAV into a normalized float array (skips header). */
        fun readWavMono(file: File): FloatArray {
            val bytes = file.readBytes()
            if (bytes.size <= 44) return FloatArray(0)
            // Locate the "data" chunk rather than assuming a fixed 44-byte header.
            var pos = 12
            var dataOff = 44
            var dataLen = bytes.size - 44
            while (pos + 8 <= bytes.size) {
                val id = String(bytes, pos, 4, Charsets.US_ASCII)
                val sz = ByteBuffer.wrap(bytes, pos + 4, 4).order(ByteOrder.LITTLE_ENDIAN).int
                if (id == "data") { dataOff = pos + 8; dataLen = sz; break }
                pos += 8 + sz + (sz and 1)
            }
            dataLen = dataLen.coerceAtMost(bytes.size - dataOff)
            val n = dataLen / 2
            val bb = ByteBuffer.wrap(bytes, dataOff, dataLen).order(ByteOrder.LITTLE_ENDIAN)
            return FloatArray(n) { bb.short / 32768f }
        }
    }

    private val env = OrtEnvironment.getEnvironment()
    private val inputName: String = session.inputNames.firstOrNull() ?: "waveform"

    /** Computes the x-vector for a 16 kHz mono clip; returns null on failure. */
    fun clone(wavFile: File): FloatArray? {
        val wav = readWavMono(wavFile)
        if (wav.size < 16000) {   // < 1 s is too little to characterize a voice
            Log.w(TAG, "clip too short for cloning: ${wav.size} samples")
            return null
        }
        return try {
            val t = OnnxTensor.createTensor(
                env, FloatBuffer.wrap(wav), longArrayOf(1, wav.size.toLong())
            )
            val r = session.run(mapOf(inputName to t))
            val out = flatten(r[0].value)
            r.close(); t.close()
            out
        } catch (e: Throwable) {
            Log.e(TAG, "clone failed: ${e.message}", e)
            null
        }
    }

    /** Flattens whatever rank the encoder returns into a 1-D float vector. */
    private fun flatten(value: Any?): FloatArray = when (value) {
        is FloatArray -> value
        is Array<*> -> value.flatMap { flatten(it).toList() }.toFloatArray()
        else -> FloatArray(0)
    }

    fun release() {
        runCatching { session.close() }
    }
}
