package com.qwen3.tts.util

import org.junit.Assert.assertEquals
import org.junit.Test

class RuNormalizerTest {

    // ── Числа ────────────────────────────────────────────────────

    @Test fun zero() = assertEquals("ноль", RuNormalizer.numberToWords("0"))

    @Test fun teens() = assertEquals("пятнадцать", RuNormalizer.numberToWords("15"))

    @Test fun year() = assertEquals(
        "одна тысяча девятьсот сорок один", RuNormalizer.numberToWords("1941"))

    @Test fun `thousand is feminine`() =
        assertEquals("две тысячи пять", RuNormalizer.numberToWords("2005"))

    @Test fun millions() =
        assertEquals("три миллиона сто тысяч", RuNormalizer.numberToWords("3100000"))

    @Test fun `plural form 11-14`() =
        assertEquals("двенадцать тысяч", RuNormalizer.numberToWords("12000"))

    @Test fun `leading zeros`() = assertEquals("семь", RuNormalizer.numberToWords("007"))

    // ── Транслитерация ───────────────────────────────────────────

    @Test fun reciter() = assertEquals("реситер", RuNormalizer.transliterate("reciter"))

    @Test fun `g softens before e`() =
        assertEquals("джеордже", RuNormalizer.transliterate("george"))

    @Test fun digraphs() = assertEquals("шип", RuNormalizer.transliterate("ship"))

    // ── Полная нормализация ──────────────────────────────────────

    @Test fun `mixed sentence`() = assertEquals(
        "Глава пять. реситер ттс читает сто пять страниц",
        RuNormalizer.normalize("Глава 5. Reciter TTS читает 105 страниц"))

    @Test fun `symbols`() = assertEquals(
        "дом  номер три", RuNormalizer.normalize("дом №3"))
}
