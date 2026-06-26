package com.qwen3.tts.engine.tokenizer

import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import java.lang.reflect.Method

class Qwen3TokenizerTest {

    private lateinit var tokenizer: Qwen3Tokenizer
    private lateinit var byteEncoder: Map<Byte, Char>
    private lateinit var byteDecoder: Map<Char, Byte>
    private lateinit var companionInstance: Any
    private val companionClass = Class.forName(
        "com.qwen3.tts.engine.tokenizer.Qwen3Tokenizer\$Companion"
    )

    @Before
    fun setUp() {
        // Get the Companion singleton
        val companionField = Qwen3Tokenizer::class.java.getField("Companion")
        companionInstance = companionField.get(null)

        // Invoke createFallbackTokenizer()
        val createMethod = companionClass.getDeclaredMethod("createFallbackTokenizer")
        createMethod.isAccessible = true
        tokenizer = createMethod.invoke(companionInstance) as Qwen3Tokenizer

        // Invoke buildByteEncoder()
        val buildMethod = companionClass.getDeclaredMethod("buildByteEncoder")
        buildMethod.isAccessible = true
        @Suppress("UNCHECKED_CAST")
        byteEncoder = buildMethod.invoke(companionInstance) as Map<Byte, Char>

        byteDecoder = byteEncoder.map { (k, v) -> v to k }.toMap()
    }

    // --- byteEncoder tests ---

    @Test
    fun `byteEncoder maps all 256 byte values`() {
        assertEquals(256, byteEncoder.size)
    }

    @Test
    fun `byteEncoder produces unique char for each byte`() {
        val chars = byteEncoder.values.toSet()
        assertEquals(256, chars.size)
    }

    @Test
    fun `byteEncoder chars are in extended Unicode range`() {
        for ((_, c) in byteEncoder) {
            assertTrue("Char ${c.code} should be >= 256", c.code >= 256)
        }
    }

    // --- byteDecoder tests ---

    @Test
    fun `byteDecoder is inverse of byteEncoder`() {
        for ((b, c) in byteEncoder) {
            assertEquals(b, byteDecoder[c])
        }
    }

    @Test
    fun `byteDecoder maps all 256 char values back`() {
        assertEquals(256, byteDecoder.size)
    }

    // --- charToBytes / stringToBytes / bytesToChar roundtrip ---

    @Test
    fun `charToBytes roundtrips ASCII char`() {
        val charToBytes = companionClass.getDeclaredMethod("charToBytes", Char::class.java)
        charToBytes.isAccessible = true
        val bytesToChar = companionClass.getDeclaredMethod("bytesToChar", String::class.java)
        bytesToChar.isAccessible = true

        val encoded = charToBytes.invoke(companionInstance, 'A') as String
        assertTrue(encoded.isNotEmpty())
        val decoded = bytesToChar.invoke(companionInstance, encoded) as String
        assertEquals("A", decoded)
    }

    @Test
    fun `charToBytes roundtrips Cyrillic char`() {
        val charToBytes = companionClass.getDeclaredMethod("charToBytes", Char::class.java)
        charToBytes.isAccessible = true
        val bytesToChar = companionClass.getDeclaredMethod("bytesToChar", String::class.java)
        bytesToChar.isAccessible = true

        val encoded = charToBytes.invoke(companionInstance, 'Я') as String
        assertTrue(encoded.isNotEmpty())
        val decoded = bytesToChar.invoke(companionInstance, encoded) as String
        assertEquals("Я", decoded)
    }

    @Test
    fun `stringToBytes roundtrips mixed text`() {
        val stringToBytes = companionClass.getDeclaredMethod("stringToBytes", String::class.java)
        stringToBytes.isAccessible = true
        val bytesToChar = companionClass.getDeclaredMethod("bytesToChar", String::class.java)
        bytesToChar.isAccessible = true

        val original = "Hello Мир!"
        val encoded = stringToBytes.invoke(companionInstance, original) as String
        val decoded = bytesToChar.invoke(companionInstance, encoded) as String
        assertEquals(original, decoded)
    }

    @Test
    fun `stringToBytes produces consistent results`() {
        val stringToBytes = companionClass.getDeclaredMethod("stringToBytes", String::class.java)
        stringToBytes.isAccessible = true

        val first = stringToBytes.invoke(companionInstance, "test") as String
        val second = stringToBytes.invoke(companionInstance, "test") as String
        assertEquals(first, second)
    }

    @Test
    fun `charToBytes handles space character`() {
        val charToBytes = companionClass.getDeclaredMethod("charToBytes", Char::class.java)
        charToBytes.isAccessible = true
        val bytesToChar = companionClass.getDeclaredMethod("bytesToChar", String::class.java)
        bytesToChar.isAccessible = true

        val encoded = charToBytes.invoke(companionInstance, ' ') as String
        assertTrue(encoded.isNotEmpty())
        val decoded = bytesToChar.invoke(companionInstance, encoded) as String
        assertEquals(" ", decoded)
    }

    // --- encode() tests ---

    @Test
    fun `encode blank string returns empty array`() {
        assertArrayEquals(intArrayOf(), tokenizer.encode(""))
    }

    @Test
    fun `encode whitespace-only string returns empty array`() {
        assertArrayEquals(intArrayOf(), tokenizer.encode("   "))
    }

    @Test
    fun `encode with special tokens wraps with im_start and im_end`() {
        val result = tokenizer.encode("test", addSpecialTokens = true)
        assertTrue("Should contain at least 3 tokens", result.size >= 3)
        assertEquals(151644, result.first()) // <|im_start|>
        assertEquals(151645, result.last())  // <|im_end|>
    }

    @Test
    fun `encode without special tokens has no wrapper`() {
        val result = tokenizer.encode("test", addSpecialTokens = false)
        if (result.isNotEmpty()) {
            assertNotEquals(151644, result.first())
            assertNotEquals(151645, result.last())
        }
    }

    @Test
    fun `encode produces non-empty result for ASCII text`() {
        assertTrue(tokenizer.encode("hello", addSpecialTokens = false).isNotEmpty())
    }

    @Test
    fun `encode produces non-empty result for Cyrillic text`() {
        assertTrue(tokenizer.encode("привет", addSpecialTokens = false).isNotEmpty())
    }

    @Test
    fun `encode is deterministic`() {
        val text = "consistent test"
        assertArrayEquals(tokenizer.encode(text), tokenizer.encode(text))
    }

    // --- encodeForTTS() tests ---

    @Test
    fun `encodeForTTS returns empty for blank text`() {
        assertEquals(0, tokenizer.encodeForTTS("").size)
    }

    @Test
    fun `encodeForTTS does not add special tokens`() {
        val result = tokenizer.encodeForTTS("hello world")
        assertTrue(result.isNotEmpty())
        assertFalse(result.contains(151644))
        assertFalse(result.contains(151645))
    }

    @Test
    fun `encodeForTTS produces tokens for Cyrillic`() {
        assertTrue(tokenizer.encodeForTTS("тест").isNotEmpty())
    }

    @Test
    fun `encodeForTTS is deterministic`() {
        val text = "повторяемый тест"
        assertArrayEquals(tokenizer.encodeForTTS(text), tokenizer.encodeForTTS(text))
    }

    // --- decode() tests ---

    @Test
    fun `decode empty array returns empty string`() {
        assertEquals("", tokenizer.decode(intArrayOf()))
    }

    @Test
    fun `decode skips special tokens by default`() {
        assertEquals("", tokenizer.decode(intArrayOf(151644, 151645)))
    }

    // --- getters ---

    @Test
    fun `getVocabSize returns positive value`() {
        assertTrue(tokenizer.getVocabSize() > 0)
    }

    @Test
    fun `getPadId returns expected constant`() {
        assertEquals(151643, tokenizer.getPadId())
    }

    @Test
    fun `getEosId returns expected constant`() {
        assertEquals(151647, tokenizer.getEosId())
    }

    @Test
    fun `getBosId returns expected constant`() {
        assertEquals(151646, tokenizer.getBosId())
    }

    @Test
    fun `getAudioTokenStart returns expected constant`() {
        assertEquals(151936, tokenizer.getAudioTokenStart())
    }
}
