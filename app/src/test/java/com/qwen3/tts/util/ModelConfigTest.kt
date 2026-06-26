package com.qwen3.tts.util

import org.junit.Assert.*
import org.junit.Test

class ModelConfigTest {

    @Test
    fun `parses a full manifest`() {
        val json = """
            {
              "id": "test-model",
              "displayName": "Test Model",
              "family": "qwen3-tts",
              "architecture": "vits",
              "sampleRateHz": 16000,
              "codecFrameRateHz": 25,
              "audioTokenStart": 100000,
              "eosTokenId": 2,
              "tokenizer": { "type": "qwen-bpe", "files": ["vocab.json", "merges.txt"] },
              "files": [
                { "filename": "talker.onnx", "role": "TALKER", "sizeMb": 400, "required": true },
                { "filename": "vocoder.onnx", "role": "VOCODER", "sizeMb": 200, "required": true }
              ],
              "voices": [
                { "id": "v_ru", "locale": "ru-RU", "displayName": "Рус", "speakerId": 0 },
                { "id": "v_en", "locale": "en-US", "displayName": "Eng", "speakerId": 1 }
              ]
            }
        """.trimIndent()

        val p = ModelConfig.parseManifestJson(json)
        assertNotNull(p)
        p!!
        assertEquals("test-model", p.id)
        assertEquals("vits", p.architecture)
        assertEquals(16000, p.sampleRateHz)
        assertEquals(100000, p.audioTokenStart)
        assertEquals(2, p.eosTokenId)
        assertEquals(2, p.modelFiles.size)
        assertEquals(2, p.voices.size)
        assertEquals(listOf("ru", "en"), p.languages)
        assertEquals("v_ru", p.defaultVoice()?.id)
        assertEquals("v_en", p.voiceForLanguage("en")?.id)
    }

    @Test
    fun `missing fields fall back to bundled profile`() {
        val p = ModelConfig.parseManifestJson("""{ "id": "minimal" }""")
        assertNotNull(p)
        p!!
        assertEquals("minimal", p.id)
        // Falls back to the bundled profile's values.
        assertEquals(ModelConfig.ACTIVE_PROFILE.architecture, p.architecture)
        assertEquals(ModelConfig.ACTIVE_PROFILE.sampleRateHz, p.sampleRateHz)
        assertEquals(ModelConfig.ACTIVE_PROFILE.audioTokenStart, p.audioTokenStart)
        assertEquals(ModelConfig.ACTIVE_PROFILE.voices.size, p.voices.size)
        assertEquals(ModelConfig.ACTIVE_PROFILE.modelFiles.size, p.modelFiles.size)
    }

    @Test
    fun `invalid json returns null`() {
        assertNull(ModelConfig.parseManifestJson("not json {"))
    }

    @Test
    fun `voice locale parsing`() {
        val v = ModelConfig.VoiceSpec("x", "zh-CN", "中文")
        assertEquals("zh", v.toLocale().language)
        assertEquals("CN", v.toLocale().country)
        assertEquals("zh", v.languageTag())
    }
}
