package com.qwen3.tts.util

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import java.io.File
import java.io.RandomAccessFile

/**
 * Minimal microphone recorder for voice cloning references. Captures 16 kHz
 * mono PCM-16 (the rate the speaker encoder expects) straight to a WAV file.
 *
 * Usage:
 *   val rec = VoiceRecorder()
 *   rec.start(outFile)        // requires RECORD_AUDIO permission
 *   ... user speaks ...
 *   val seconds = rec.stop()  // finalizes the WAV header
 */
class VoiceRecorder {

    companion object {
        const val SAMPLE_RATE = 16_000
        private const val CHANNEL = AudioFormat.CHANNEL_IN_MONO
        private const val ENCODING = AudioFormat.ENCODING_PCM_16BIT
    }

    @Volatile private var recording = false
    private var thread: Thread? = null
    private var record: AudioRecord? = null
    @Volatile private var bytesWritten = 0L

    val isRecording: Boolean get() = recording

    /** Begins recording into [outFile]. Throws if the mic can't be opened. */
    fun start(outFile: File) {
        if (recording) return
        val minBuf = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL, ENCODING)
            .coerceAtLeast(4096)
        @Suppress("MissingPermission")
        val ar = AudioRecord(
            MediaRecorder.AudioSource.MIC, SAMPLE_RATE, CHANNEL, ENCODING, minBuf * 2
        )
        if (ar.state != AudioRecord.STATE_INITIALIZED) {
            ar.release()
            throw IllegalStateException("AudioRecord init failed")
        }
        record = ar
        bytesWritten = 0L
        recording = true
        outFile.parentFile?.mkdirs()

        thread = Thread {
            val raf = RandomAccessFile(outFile, "rw")
            raf.setLength(0)
            writePlaceholderHeader(raf)               // patched on stop()
            val buf = ByteArray(minBuf)
            ar.startRecording()
            while (recording) {
                val n = ar.read(buf, 0, buf.size)
                if (n > 0) { raf.write(buf, 0, n); bytesWritten += n }
            }
            ar.stop()
            patchHeader(raf, bytesWritten)
            raf.close()
        }.also { it.start() }
    }

    /** Stops recording and returns the captured duration in seconds. */
    fun stop(): Float {
        if (!recording) return 0f
        recording = false
        thread?.join(2000)
        thread = null
        record?.release()
        record = null
        return bytesWritten / 2f / SAMPLE_RATE
    }

    fun cancel() {
        runCatching { stop() }
    }

    private fun writePlaceholderHeader(raf: RandomAccessFile) {
        raf.write(ByteArray(44))   // 44-byte WAV header, filled in patchHeader
    }

    private fun patchHeader(raf: RandomAccessFile, dataLen: Long) {
        val totalLen = dataLen + 36
        val byteRate = SAMPLE_RATE * 2
        fun le32(v: Long) = byteArrayOf(
            (v and 0xFF).toByte(), ((v shr 8) and 0xFF).toByte(),
            ((v shr 16) and 0xFF).toByte(), ((v shr 24) and 0xFF).toByte()
        )
        fun le16(v: Int) = byteArrayOf((v and 0xFF).toByte(), ((v shr 8) and 0xFF).toByte())
        raf.seek(0)
        raf.writeBytes("RIFF"); raf.write(le32(totalLen)); raf.writeBytes("WAVE")
        raf.writeBytes("fmt "); raf.write(le32(16)); raf.write(le16(1)); raf.write(le16(1))
        raf.write(le32(SAMPLE_RATE.toLong())); raf.write(le32(byteRate.toLong()))
        raf.write(le16(2)); raf.write(le16(16))
        raf.writeBytes("data"); raf.write(le32(dataLen))
    }
}
