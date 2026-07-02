package com.qwen3.tts.engine.inference

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Проверяет, что Kotlin-порт G2P из vosk-tts даёт побуквенно тот же результат,
 * что оригинальный python `vosk_tts.g2p.convert` (эталоны сняты с pip-пакета
 * vosk-tts 0.3.61).
 */
class VoskG2pTest {

    private fun g2p(word: String): String = VoskTtsEngine.convert(word).joinToString(" ")

    @Test fun `stressed word`() =
        assertEquals("a0 b s t r a1 k c i0 j u0", g2p("абстр+акцию"))

    @Test fun `plain word with palatalization`() =
        assertEquals("p rj i0 vj e0 t", g2p("привет"))

    @Test fun `hard sign j-glide`() =
        assertEquals("s j e0 sh", g2p("съешь"))

    @Test fun `initial jo`() =
        assertEquals("j o0 l k a0", g2p("ёлка"))

    @Test fun `stress on ending with j-glide`() =
        assertEquals("b o0 lj sh a1 j a0", g2p("больш+ая"))

    @Test fun `soft sign consonant cluster`() =
        assertEquals("zh i0 z nj", g2p("жизнь"))
}
