package com.qwen3.tts.engine.inference

import org.junit.Assert.*
import org.junit.Test
import java.lang.reflect.Method

class QwenInferenceEngineTest {

    // --- extractSpeakerId tests ---

    @Test
    fun `extractSpeakerId returns digit from voice name`() {
        // "qwen3_ru_5" -> filter digits -> "35" -> 35
        assertEquals(35, callExtractSpeakerId("qwen3_ru_5"))
    }

    @Test
    fun `extractSpeakerId returns digit embedded in prefix`() {
        // "qwen3_ru" -> filter digits -> "3" -> 3
        assertEquals(3, callExtractSpeakerId("qwen3_ru"))
    }

    @Test
    fun `extractSpeakerId returns 0 for name without digits`() {
        assertEquals(0, callExtractSpeakerId("speaker_default"))
    }

    @Test
    fun `extractSpeakerId extracts multi-digit number`() {
        assertEquals(12, callExtractSpeakerId("speaker_12"))
    }

    @Test
    fun `extractSpeakerId returns concatenated digits`() {
        // "v2_speaker_3" -> filter digits -> "23" -> 23
        assertEquals(23, callExtractSpeakerId("v2_speaker_3"))
    }

    @Test
    fun `extractSpeakerId returns 0 for empty string`() {
        assertEquals(0, callExtractSpeakerId(""))
    }

    @Test
    fun `extractSpeakerId handles only digits string`() {
        assertEquals(42, callExtractSpeakerId("42"))
    }

    // --- floatToPcm16 tests ---

    @Test
    fun `floatToPcm16 empty input returns empty output`() {
        val result = callFloatToPcm16(floatArrayOf())
        assertEquals(0, result.size)
    }

    @Test
    fun `floatToPcm16 output size is double input size`() {
        val input = floatArrayOf(0.0f, 0.5f, -0.5f)
        val result = callFloatToPcm16(input)
        assertEquals(input.size * 2, result.size)
    }

    @Test
    fun `floatToPcm16 zero sample produces zero bytes`() {
        val result = callFloatToPcm16(floatArrayOf(0.0f))
        assertEquals(0.toByte(), result[0])
        assertEquals(0.toByte(), result[1])
    }

    @Test
    fun `floatToPcm16 positive one maps to max PCM16`() {
        val result = callFloatToPcm16(floatArrayOf(1.0f))
        assertEquals(32767, readSample(result, 0))
    }

    @Test
    fun `floatToPcm16 negative one maps to min PCM16`() {
        val result = callFloatToPcm16(floatArrayOf(-1.0f))
        assertEquals(-32767, readSample(result, 0))
    }

    @Test
    fun `floatToPcm16 clamps values above 1`() {
        val result = callFloatToPcm16(floatArrayOf(2.0f))
        assertEquals(32767, readSample(result, 0))
    }

    @Test
    fun `floatToPcm16 clamps values below -1`() {
        val result = callFloatToPcm16(floatArrayOf(-2.0f))
        assertEquals(-32768, readSample(result, 0))
    }

    @Test
    fun `floatToPcm16 half sample`() {
        // 0.5f * 32767 = 16383
        val result = callFloatToPcm16(floatArrayOf(0.5f))
        assertEquals(16383, readSample(result, 0))
    }

    @Test
    fun `floatToPcm16 little-endian byte order`() {
        val result = callFloatToPcm16(floatArrayOf(0.5f))
        // 16383 = 0x3FFF -> LE: low=0xFF, high=0x3F
        val lo = result[0].toInt() and 0xFF
        val hi = result[1].toInt() and 0xFF
        assertEquals(0xFF, lo)
        assertEquals(0x3F, hi)
    }

    @Test
    fun `floatToPcm16 preserves multiple samples in order`() {
        val input = floatArrayOf(0.0f, 1.0f, -1.0f)
        val result = callFloatToPcm16(input)
        assertEquals(6, result.size)
        assertEquals(0, readSample(result, 0))
        assertEquals(32767, readSample(result, 1))
        assertEquals(-32767, readSample(result, 2))
    }

    // --- helpers ---

    private fun readSample(bytes: ByteArray, index: Int): Int {
        val lo = bytes[index * 2].toInt() and 0xFF
        val hi = bytes[index * 2 + 1].toInt() and 0xFF
        val unsigned = (hi shl 8) or lo
        return if (unsigned >= 32768) unsigned - 65536 else unsigned
    }

    private fun createUninitializedInstance(): Any {
        val unsafeClass = Class.forName("sun.misc.Unsafe")
        val theUnsafe = unsafeClass.getDeclaredField("theUnsafe")
        theUnsafe.isAccessible = true
        val unsafe = theUnsafe.get(null)
        val allocateInstance = unsafeClass.getMethod("allocateInstance", Class::class.java)
        return allocateInstance.invoke(unsafe, QwenInferenceEngine::class.java)
    }

    private fun callExtractSpeakerId(voiceName: String): Int {
        val method = QwenInferenceEngine::class.java
            .getDeclaredMethod("extractSpeakerId", String::class.java)
        method.isAccessible = true
        val instance = createUninitializedInstance()
        return method.invoke(instance, voiceName) as Int
    }

    private fun callFloatToPcm16(audio: FloatArray): ByteArray {
        val method = QwenInferenceEngine::class.java
            .getDeclaredMethod("floatToPcm16", FloatArray::class.java)
        method.isAccessible = true
        val instance = createUninitializedInstance()
        return method.invoke(instance, audio) as ByteArray
    }
}
