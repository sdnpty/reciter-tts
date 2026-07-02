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

    // With no model installed, the engine must not advertise ANY language or
    // voice — the compile-time Qwen3 fallback profile is metadata only. In this
    // JVM test environment installedModels() is always empty (no filesystem),
    // which is exactly the "fresh install" state.

    @Test
    fun `Russian not advertised until a model is installed`() {
        val result = isLanguageAvailable.invoke(engine, "ru", "", "") as Int
        assertEquals(-2, result) // LANG_NOT_SUPPORTED
    }

    @Test
    fun `Russian with country not advertised until a model is installed`() {
        val result = isLanguageAvailable.invoke(engine, "ru", "RU", "") as Int
        assertEquals(-2, result)
    }

    @Test
    fun `English not advertised until a model is installed`() {
        val result = isLanguageAvailable.invoke(engine, "en", "", "") as Int
        assertEquals(-2, result)
    }

    @Test
    fun `Chinese not advertised until a model is installed`() {
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
    fun `ru_male_1 not a valid voice until a model is installed`() {
        val result = isValidVoiceName.invoke(engine, "ru_male_1") as Int
        assertEquals(-2, result)
    }

    @Test
    fun `ru_female_1 not a valid voice until a model is installed`() {
        val result = isValidVoiceName.invoke(engine, "ru_female_1") as Int
        assertEquals(-2, result)
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
