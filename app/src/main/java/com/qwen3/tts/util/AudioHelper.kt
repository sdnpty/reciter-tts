package com.qwen3.tts.util

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.roundToInt

object AudioHelper {

    /**
     * Pitch-preserving time-stretch (WSOLA). [speed] > 1 = faster/shorter,
     * < 1 = slower/longer, 1.0 = unchanged. Unlike plain resampling this keeps
     * the speaker's pitch. Falls back to resampling for very short clips where
     * windowed overlap-add has too few frames to work with.
     */
    fun timeStretch(input: FloatArray, speed: Float, @Suppress("UNUSED_PARAMETER") sampleRate: Int): FloatArray {
        if (input.isEmpty() || speed <= 0.1f || speed > 4f || abs(speed - 1f) < 0.01f) return input

        val frame = 1024
        if (input.size < frame * 2) return changeSpeed(input, speed)

        val synHop = frame / 2
        val anaHop = (synHop * speed).roundToInt().coerceAtLeast(1)
        val search = 256

        val win = FloatArray(frame) { 0.5f - 0.5f * cos(2.0 * PI * it / (frame - 1)).toFloat() }

        val outLen = (input.size / speed).toInt() + frame
        val out = FloatArray(outLen)
        val norm = FloatArray(outLen)

        var anaCenter = 0
        var syn = 0
        var naturalStart = -1   // template of the "natural continuation" to align to

        while (syn + frame <= outLen && anaCenter + frame <= input.size) {
            val a: Int = if (naturalStart < 0) {
                anaCenter
            } else {
                // Search the offset whose frame best continues the previous output.
                val lo = (anaCenter - search).coerceAtLeast(0)
                val hi = (anaCenter + search).coerceAtMost(input.size - frame)
                var best = anaCenter.coerceIn(0, input.size - frame)
                var bestCorr = -Float.MAX_VALUE
                var cand = lo
                while (cand <= hi) {
                    var corr = 0f
                    var i = 0
                    while (i < frame) {   // subsampled correlation for speed
                        corr += input[cand + i] * input[naturalStart + i]
                        i += 8
                    }
                    if (corr > bestCorr) { bestCorr = corr; best = cand }
                    cand++
                }
                best
            }

            for (i in 0 until frame) {
                out[syn + i] += input[a + i] * win[i]
                norm[syn + i] += win[i]
            }

            naturalStart = (a + synHop).coerceIn(0, input.size - frame)
            syn += synHop
            anaCenter += anaHop
        }

        for (i in out.indices) if (norm[i] > 1e-6f) out[i] /= norm[i]

        val finalLen = (input.size / speed).toInt().coerceIn(1, out.size)
        return out.copyOf(finalLen)
    }

    /**
     * Changes playback speed by [factor] (1.0 = unchanged, 1.5 = 50% faster,
     * 0.8 = 20% slower) via linear resampling. This is a lightweight v1: it
     * shifts pitch along with tempo. Returns the input unchanged for factors
     * that are ~1.0 or out of a sane range.
     */
    fun changeSpeed(audio: FloatArray, factor: Float): FloatArray {
        if (audio.isEmpty() || factor <= 0.1f || factor > 4f || kotlin.math.abs(factor - 1f) < 0.01f) {
            return audio
        }
        val outLen = (audio.size / factor).toInt().coerceAtLeast(1)
        val out = FloatArray(outLen)
        for (i in 0 until outLen) {
            val srcPos = i * factor
            val idx = srcPos.toInt()
            val frac = srcPos - idx
            val a = audio[idx.coerceIn(0, audio.size - 1)]
            val b = audio[(idx + 1).coerceIn(0, audio.size - 1)]
            out[i] = a + (b - a) * frac
        }
        return out
    }

    fun floatToPcm16(audio: FloatArray): ByteArray {
        val bytes = ByteArray(audio.size * 2)
        for (i in audio.indices) {
            val sample = (audio[i] * 32767).toInt().coerceIn(-32768, 32767)
            bytes[i * 2] = (sample and 0xFF).toByte()
            bytes[i * 2 + 1] = ((sample shr 8) and 0xFF).toByte()
        }
        return bytes
    }

    fun buildSpeechAudioTrack(
        sampleRate: Int,
        encoding: Int,
        bufferSizeInBytes: Int,
        transferMode: Int = AudioTrack.MODE_STATIC
    ): AudioTrack {
        return AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setSampleRate(sampleRate)
                    .setEncoding(encoding)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build()
            )
            .setBufferSizeInBytes(bufferSizeInBytes)
            .setTransferMode(transferMode)
            .build()
    }
}
