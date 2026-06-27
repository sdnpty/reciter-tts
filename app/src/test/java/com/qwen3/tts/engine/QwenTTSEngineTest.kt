package com.qwen3.tts.engine

import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import java.lang.reflect.Method

class QwenTTSEngineTest {

    private lateinit var engine: Any
    private lateinit var isLanguageAvailable: Method
    private lateinit var isValidVoiceName: Method
    private lateinit var loadLanguage: Method
    private lateinit var loadVoice: Method
    private lateinit var getLanguage: Method

    @Before
    fun setUp() {
        val unsafeClass = Class.forName("sun.misc.Unsafe")
        val theUnsafe = unsafeClass.getDeclaredField("theUnsafe")
        theUnsafe.isAccessible = true
        val unsafe = theUnsafe.get(null)
        val allocateInstance = unsafeClass.getMethod("allocateInstance", Class::class.java)
        engine = allocateInstance.invoke(unsafe, QwenTTSEngine::class.java)

        isLanguageAvailable = QwenTTSEngine::class.java
            .getDeclaredMethod("onIsLanguageAvailable", String::class.java, String::class.java, String::class.java)
        isLanguageAvailable.isAccessible = true

        isValidVoiceName = QwenTTSEngine::class.java
            .getDeclaredMethod("onIsValidVoiceName", String::class.java)
        isValidVoiceName.isAccessible = true

        loadLanguage = QwenTTSEngine::class.java
            .getDeclaredMethod("onLoadLanguage", String::class.java, String::class.java, String::class.java)
        loadLanguage.isAccessible = true

        loadVoice = QwenTTSEngine::class.java
            .getDeclaredMethod("onLoadVoice", String::class.java)
        loadVoice.isAccessible = true

        getLanguage = QwenTTSEngine::class.java
            .getDeclaredMethod("onGetLanguage")
        getLanguage.isAccessible = true
    }

    // --- onIsLanguageAvailable tests ---

    @Test
    fun `Russian language is available`() {
        val result = isLanguageAvailable.invoke(engine, "ru", "", "") as Int
        assertTrue(result >= 0)
    }

    @Test
    fun `Russian with country is country-available`() {
        val result = isLanguageAvailable.invoke(engine, "ru", "RU", "") as Int
        assertEquals(1, result) // LANG_COUNTRY_AVAILABLE
    }

    @Test
    fun `English not supported in Russian-only fallback`() {
        // The bundled AR model ships Russian voices (FLEURS); other languages
        // become available only when a model declaring them is installed.
        val result = isLanguageAvailable.invoke(engine, "en", "", "") as Int
        assertEquals(-2, result)
    }

    @Test
    fun `Chinese not supported in Russian-only fallback`() {
        val result = isLanguageAvailable.invoke(engine, "zh", "", "") as Int
        assertEquals(-2, result)
    }

    @Test
    fun `Unsupported language returns not-supported`() {
        val result = isLanguageAvailable.invoke(engine, "fr", "", "") as Int
        assertEquals(-2, result) // LANG_NOT_SUPPORTED
    }

    @Test
    fun `Japanese language is not supported`() {
        val result = isLanguageAvailable.invoke(engine, "ja", "", "") as Int
        assertEquals(-2, result)
    }

    @Test
    fun `Empty language is not supported`() {
        val result = isLanguageAvailable.invoke(engine, "", "", "") as Int
        assertEquals(-2, result)
    }

    // --- onIsValidVoiceName tests ---

    @Test
    fun `ru_male_1 is valid voice`() {
        val result = isValidVoiceName.invoke(engine, "ru_male_1") as Int
        assertEquals(1, result)
    }

    @Test
    fun `ru_female_1 is valid voice`() {
        val result = isValidVoiceName.invoke(engine, "ru_female_1") as Int
        assertEquals(1, result)
    }

    @Test
    fun `legacy qwen3_zh is no longer a valid voice`() {
        val result = isValidVoiceName.invoke(engine, "qwen3_zh") as Int
        assertEquals(-2, result)
    }

    @Test
    fun `undeclared voice is invalid`() {
        // Voice validity is now driven by the active model's declared voices,
        // not a name prefix — an unknown id is not a valid voice.
        val result = isValidVoiceName.invoke(engine, "qwen3_custom") as Int
        assertEquals(-2, result)
    }

    @Test
    fun `non-qwen3 voice is invalid`() {
        val result = isValidVoiceName.invoke(engine, "google_tts") as Int
        assertEquals(-2, result)
    }

    @Test
    fun `empty voice name is invalid`() {
        val result = isValidVoiceName.invoke(engine, "") as Int
        assertEquals(-2, result)
    }

    // --- onLoadLanguage tests ---

    @Test
    fun `onLoadLanguage returns same as onIsLanguageAvailable`() {
        val available = isLanguageAvailable.invoke(engine, "ru", "RU", "") as Int
        val loaded = loadLanguage.invoke(engine, "ru", "RU", "") as Int
        assertEquals(available, loaded)
    }

    @Test
    fun `onLoadLanguage for unsupported language`() {
        val result = loadLanguage.invoke(engine, "fr", "", "") as Int
        assertEquals(-2, result)
    }

    // --- onLoadVoice tests ---

    @Test
    fun `onLoadVoice returns same as onIsValidVoiceName`() {
        val valid = isValidVoiceName.invoke(engine, "ru_male_1") as Int
        val loaded = loadVoice.invoke(engine, "ru_male_1") as Int
        assertEquals(valid, loaded)
    }

    @Test
    fun `onLoadVoice for invalid voice`() {
        val result = loadVoice.invoke(engine, "invalid_voice") as Int
        assertEquals(-2, result)
    }

    // --- onGetLanguage tests ---

    @Test
    fun `onGetLanguage returns Russian locale`() {
        @Suppress("UNCHECKED_CAST")
        val result = getLanguage.invoke(engine) as Array<String>
        assertEquals(3, result.size)
        assertEquals("ru", result[0])
        assertEquals("RU", result[1])
        assertEquals("", result[2])
    }
}
