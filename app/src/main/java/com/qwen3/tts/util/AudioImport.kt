package com.qwen3.tts.util

import android.content.Context
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import android.util.Log
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Decodes an arbitrary audio file (mp3/m4a/aac/wav/ogg/flac — whatever the
 * device's codecs support) to a 16 kHz mono PCM-16 WAV, the format the speaker
 * encoder expects. Used to add a voice from a saved clip instead of the mic.
 */
object AudioImport {

    private const val TAG = "AudioImport"
    private const val TARGET_RATE = 16_000

    /** Returns the captured duration in seconds, or 0 on failure. */
    fun decodeToWav16kMono(context: Context, uri: Uri, outFile: File): Float {
        val extractor = MediaExtractor()
        try {
            context.contentResolver.openFileDescriptor(uri, "r").use { pfd ->
                extractor.setDataSource(pfd!!.fileDescriptor)
            }
            var trackIdx = -1
            var format: MediaFormat? = null
            for (i in 0 until extractor.trackCount) {
                val f = extractor.getTrackFormat(i)
                if (f.getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true) {
                    trackIdx = i; format = f; break
                }
            }
            if (trackIdx < 0 || format == null) return 0f
            extractor.selectTrack(trackIdx)

            val mime = format.getString(MediaFormat.KEY_MIME)!!
            val srcRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            val srcCh = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT).coerceAtLeast(1)

            val codec = MediaCodec.createDecoderByType(mime)
            codec.configure(format, null, null, 0)
            codec.start()

            val mono = ArrayList<Float>(srcRate * 8)
            val info = MediaCodec.BufferInfo()
            var sawInputEnd = false
            var sawOutputEnd = false
            while (!sawOutputEnd) {
                if (!sawInputEnd) {
                    val inIdx = codec.dequeueInputBuffer(10_000)
                    if (inIdx >= 0) {
                        val buf = codec.getInputBuffer(inIdx)!!
                        val n = extractor.readSampleData(buf, 0)
                        if (n < 0) {
                            codec.queueInputBuffer(inIdx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            sawInputEnd = true
                        } else {
                            codec.queueInputBuffer(inIdx, 0, n, extractor.sampleTime, 0)
                            extractor.advance()
                        }
                    }
                }
                val outIdx = codec.dequeueOutputBuffer(info, 10_000)
                if (outIdx >= 0) {
                    val outBuf = codec.getOutputBuffer(outIdx)!!
                    outBuf.order(ByteOrder.LITTLE_ENDIAN)
                    val shorts = outBuf.asShortBuffer()
                    // Downmix interleaved channels to mono.
                    val total = info.size / 2
                    var i = 0
                    while (i + srcCh <= total) {
                        var acc = 0f
                        for (c in 0 until srcCh) acc += shorts.get(i + c) / 32768f
                        mono.add(acc / srcCh)
                        i += srcCh
                    }
                    codec.releaseOutputBuffer(outIdx, false)
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) sawOutputEnd = true
                } else if (outIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    // ignore
                }
            }
            codec.stop(); codec.release()

            var samples = mono.toFloatArray()
            if (srcRate != TARGET_RATE && samples.isNotEmpty()) {
                samples = AudioHelper.changeSpeed(samples, srcRate.toFloat() / TARGET_RATE)
            }
            if (samples.isEmpty()) return 0f
            writeWav16kMono(outFile, samples)
            return samples.size.toFloat() / TARGET_RATE
        } catch (e: Throwable) {
            Log.e(TAG, "decode failed: ${e.message}", e)
            return 0f
        } finally {
            runCatching { extractor.release() }
        }
    }

    private fun writeWav16kMono(outFile: File, samples: FloatArray) {
        outFile.parentFile?.mkdirs()
        val pcm = AudioHelper.floatToPcm16(samples)
        val raf = RandomAccessFile(outFile, "rw")
        raf.setLength(0)
        val byteRate = TARGET_RATE * 2
        fun le32(v: Long) = byteArrayOf(
            (v and 0xFF).toByte(), ((v shr 8) and 0xFF).toByte(),
            ((v shr 16) and 0xFF).toByte(), ((v shr 24) and 0xFF).toByte()
        )
        fun le16(v: Int) = byteArrayOf((v and 0xFF).toByte(), ((v shr 8) and 0xFF).toByte())
        raf.writeBytes("RIFF"); raf.write(le32(pcm.size + 36L)); raf.writeBytes("WAVE")
        raf.writeBytes("fmt "); raf.write(le32(16)); raf.write(le16(1)); raf.write(le16(1))
        raf.write(le32(TARGET_RATE.toLong())); raf.write(le32(byteRate.toLong()))
        raf.write(le16(2)); raf.write(le16(16))
        raf.writeBytes("data"); raf.write(le32(pcm.size.toLong())); raf.write(pcm)
        raf.close()
    }
}
