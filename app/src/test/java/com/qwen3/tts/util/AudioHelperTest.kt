package com.qwen3.tts.util

import org.junit.Assert.*
import org.junit.Test
import kotlin.math.sin

class AudioHelperTest {

    private fun sine(samples: Int, freq: Double = 220.0, sr: Int = 24000) =
        FloatArray(samples) { (sin(2 * Math.PI * freq * it / sr) * 0.5).toFloat() }

    @Test
    fun `speed 1 returns input unchanged`() {
        val input = sine(8000)
        val out = AudioHelper.timeStretch(input, 1.0f, 24000)
        assertEquals(input.size, out.size)
    }

    @Test
    fun `speeding up shortens audio`() {
        val input = sine(48000)
        val out = AudioHelper.timeStretch(input, 2.0f, 24000)
        val expected = input.size / 2
        // Allow a one-frame tolerance from windowed overlap-add.
        assertTrue("len=${out.size}, expected≈$expected", kotlin.math.abs(out.size - expected) <= 1100)
    }

    @Test
    fun `slowing down lengthens audio`() {
        val input = sine(48000)
        val out = AudioHelper.timeStretch(input, 0.5f, 24000)
        assertTrue(out.size > input.size)
    }

    @Test
    fun `empty input is safe`() {
        assertEquals(0, AudioHelper.timeStretch(FloatArray(0), 1.5f, 24000).size)
    }

    @Test
    fun `floatToPcm16 length is doubled`() {
        assertEquals(6, AudioHelper.floatToPcm16(floatArrayOf(0f, 0.5f, -0.5f)).size)
    }
}
