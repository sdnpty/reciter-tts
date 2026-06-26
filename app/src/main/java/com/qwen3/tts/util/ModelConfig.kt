package com.qwen3.tts.util

import android.content.Context
import org.json.JSONObject
import java.io.File
import java.util.Locale

/**
 * Model catalog + capability resolution.
 *
 * A model set can ship a `model.json` manifest next to its ONNX files (inside
 * the same ZIP). When present, the app reads the model's declared languages,
 * voices, file roles, sample rate and token offsets from it — so new models of
 * the family can be added without touching app code. When absent, the bundled
 * [ACTIVE_PROFILE] is used as a fallback.
 */
object ModelConfig {

    const val MANIFEST_FILE = "model.json"

    data class ModelFile(
        val filename: String,
        val expectedSizeMb: Long,
        val role: Role,
        val requiredForSynthesis: Boolean = true
    )

    enum class Role {
        // qwen3-codec / vits
        TALKER, CODE_PREDICTOR, VOCODER, SPEAKER_ENCODER,
        // cosyvoice / flow-matching families
        LLM, FLOW, SPEECH_TOKENIZER;
        companion object {
            fun fromString(s: String): Role? = entries.firstOrNull { it.name.equals(s, true) }
        }
    }

    data class VoiceSpec(
        val id: String,
        val locale: String,        // BCP-47-ish, e.g. "ru-RU"
        val displayName: String,
        val speakerId: Int = 0,
        val quality: String = "very_high"
    ) {
        fun toLocale(): Locale {
            val parts = locale.split('-', '_')
            return when (parts.size) {
                1 -> Locale(parts[0])
                2 -> Locale(parts[0], parts[1])
                else -> Locale(parts[0], parts[1], parts[2])
            }
        }
        fun languageTag(): String = locale.split('-', '_').firstOrNull()?.lowercase() ?: locale
    }

    data class ModelProfile(
        val id: String,
        val displayName: String,
        val family: String,
        /**
         * Inference architecture. Selects which engine runs the model:
         *  - "qwen3-codec": autoregressive talker → neural codec → code2wav (default)
         *  - "vits": single-shot VITS (MMS-TTS / Piper) input_ids → waveform
         * Other families (cosyvoice, f5, xtts) are planned — see docs/MODELS.md.
         */
        val architecture: String,
        val sampleRateHz: Int,
        val codecFrameRateHz: Int,
        val audioTokenStart: Int,
        val eosTokenId: Int,
        val modelFiles: List<ModelFile>,
        val tokenizerFiles: List<String>,
        val voices: List<VoiceSpec>
    ) {
        /** Distinct primary language tags, in declared order. */
        val languages: List<String> get() = voices.map { it.languageTag() }.distinct()

        fun defaultVoice(): VoiceSpec? = voices.firstOrNull()

        fun voiceById(id: String?): VoiceSpec? = voices.firstOrNull { it.id == id }

        fun voiceForLanguage(lang: String): VoiceSpec? =
            voices.firstOrNull { it.languageTag().equals(lang, true) }
    }

    // ── Bundled fallback profile (Qwen3-TTS 0.6B) ────────────────

    val QWEN3_TTS_12HZ_06B = ModelProfile(
        id = "qwen3-tts-12hz-0.6b",
        displayName = "Qwen3 TTS 12Hz 0.6B",
        family = "qwen3-tts",
        architecture = "qwen3-codec",
        sampleRateHz = 24_000,
        codecFrameRateHz = 12,
        audioTokenStart = 151_936,
        eosTokenId = 151_645,
        modelFiles = listOf(
            ModelFile("talker_base_android.onnx", 428L, Role.TALKER),
            ModelFile("code_predictor_base_android.onnx", 77L, Role.CODE_PREDICTOR, requiredForSynthesis = false),
            ModelFile("code2wav_android.onnx", 210L, Role.VOCODER),
            ModelFile("speaker_encoder_android.onnx", 9L, Role.SPEAKER_ENCODER, requiredForSynthesis = false)
        ),
        tokenizerFiles = listOf("vocab.json", "merges.txt"),
        voices = listOf(
            VoiceSpec("qwen3_ru", "ru-RU", "Русский", speakerId = 0),
            VoiceSpec("qwen3_en", "en-US", "English", speakerId = 1),
            VoiceSpec("qwen3_zh", "zh-CN", "中文", speakerId = 2)
        )
    )

    val SUPPORTED_PROFILES: List<ModelProfile> = listOf(QWEN3_TTS_12HZ_06B)

    /** Compile-time fallback when no manifest is installed. */
    val ACTIVE_PROFILE: ModelProfile = QWEN3_TTS_12HZ_06B

    val ARCHIVE_RENAME_MAP: Map<String, String>
        get() = mapOf(
            "talker_base.onnx" to "talker_base_android.onnx",
            "code_predictor_base.onnx" to "code_predictor_base_android.onnx",
            "code2wav.onnx" to "code2wav_android.onnx",
            "speaker_encoder.onnx" to "speaker_encoder_android.onnx"
        )

    // ── Directories & slots ──────────────────────────────────────
    //
    // Several models can be installed side by side, each in its own slot
    // directory: models/<slotId>/. The legacy flat layout (ONNX files directly
    // under models/) is still recognised as the "default" slot for backward
    // compatibility. The active slot is stored in SharedPreferences.

    private const val PREFS = "qwen3_tts_prefs"
    private const val ACTIVE_KEY = "active_model_id"
    private const val LEGACY_SLOT = "default"

    /** Root directory that holds all model slots. */
    fun getModelsDir(context: Context): File =
        context.getExternalFilesDir("models") ?: File(context.filesDir, "models")

    fun getModelsDirOrCreate(context: Context): File =
        getModelsDir(context).also { if (!it.exists()) it.mkdirs() }

    /** Resolve a model file within the active slot directory. */
    fun getModelFile(context: Context, filename: String): File =
        File(activeModelDir(context), filename)

    data class InstalledModel(val id: String, val dir: File, val profile: ModelProfile) {
        fun synthesisReady(): Boolean =
            profile.modelFiles.filter { it.requiredForSynthesis }.all { File(dir, it.filename).exists() }
    }

    /** Scan the models root for installed models (subdir slots + legacy flat). */
    fun installedModels(context: Context): List<InstalledModel> {
        val root = getModelsDir(context)
        val list = mutableListOf<InstalledModel>()

        root.listFiles()
            ?.filter { it.isDirectory && !it.name.startsWith("_") }
            ?.sortedBy { it.name }
            ?.forEach { dir ->
                val hasManifest = File(dir, MANIFEST_FILE).exists()
                val hasOnnx = dir.listFiles()?.any { it.name.endsWith(".onnx", ignoreCase = true) } == true
                if (hasManifest || hasOnnx) {
                    val profile = profileForDir(dir)
                    val id = if (hasManifest) profile.id else dir.name
                    list.add(InstalledModel(id, dir, profile))
                }
            }

        // Legacy flat layout: ONNX files directly under the root.
        val rootHasOnnx = root.listFiles()?.any { it.name.endsWith(".onnx", ignoreCase = true) } == true
        if (rootHasOnnx) {
            list.add(0, InstalledModel(LEGACY_SLOT, root, profileForDir(root)))
        }
        return list
    }

    fun activeModel(context: Context): InstalledModel? {
        val models = installedModels(context)
        if (models.isEmpty()) return null
        val stored = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getString(ACTIVE_KEY, null)
        return models.firstOrNull { it.id == stored }
            ?: models.firstOrNull { it.synthesisReady() }
            ?: models.first()
    }

    fun activeModelDir(context: Context): File = activeModel(context)?.dir ?: getModelsDir(context)

    fun setActiveModel(context: Context, id: String) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putString(ACTIVE_KEY, id).apply()
        invalidateProfileCache()
    }

    // ── Active profile resolution (manifest-aware, cached) ───────

    @Volatile private var cachedProfile: ModelProfile? = null
    @Volatile private var cachedKey: String = ""

    private fun profileForDir(dir: File): ModelProfile {
        val manifest = File(dir, MANIFEST_FILE)
        return (if (manifest.exists()) parseManifest(manifest) else null) ?: ACTIVE_PROFILE
    }

    fun activeProfile(context: Context): ModelProfile {
        val dir = activeModelDir(context)
        val manifest = File(dir, MANIFEST_FILE)
        val key = "${dir.absolutePath}:${if (manifest.exists()) manifest.lastModified() else 0L}"
        cachedProfile?.let { if (key == cachedKey) return it }
        val profile = profileForDir(dir)
        cachedProfile = profile
        cachedKey = key
        return profile
    }

    /** Drop the cached profile (call after install/delete/switch). */
    fun invalidateProfileCache() {
        cachedProfile = null
        cachedKey = ""
    }

    private fun parseManifest(file: File): ModelProfile? =
        try { parseManifestJson(file.readText()) } catch (e: Exception) { null }

    /** Parse a manifest from its JSON text. Visible for testing. */
    internal fun parseManifestJson(text: String): ModelProfile? = try {
        val json = JSONObject(text)
        val fallback = ACTIVE_PROFILE

        val files = json.optJSONArray("files")?.let { arr ->
            (0 until arr.length()).mapNotNull { i ->
                val o = arr.getJSONObject(i)
                val role = Role.fromString(o.optString("role")) ?: return@mapNotNull null
                ModelFile(
                    filename = o.getString("filename"),
                    expectedSizeMb = o.optLong("sizeMb", 0L),
                    role = role,
                    requiredForSynthesis = o.optBoolean("required", true)
                )
            }
        }?.takeIf { it.isNotEmpty() } ?: fallback.modelFiles

        val voices = json.optJSONArray("voices")?.let { arr ->
            (0 until arr.length()).map { i ->
                val o = arr.getJSONObject(i)
                VoiceSpec(
                    id = o.getString("id"),
                    locale = o.optString("locale", "en-US"),
                    displayName = o.optString("displayName", o.getString("id")),
                    speakerId = o.optInt("speakerId", 0),
                    quality = o.optString("quality", "very_high")
                )
            }
        }?.takeIf { it.isNotEmpty() } ?: fallback.voices

        val tokenizerFiles = json.optJSONObject("tokenizer")?.optJSONArray("files")?.let { arr ->
            (0 until arr.length()).map { arr.getString(it) }
        } ?: fallback.tokenizerFiles

        ModelProfile(
            id = json.optString("id", fallback.id),
            displayName = json.optString("displayName", fallback.displayName),
            family = json.optString("family", fallback.family),
            architecture = json.optString("architecture", fallback.architecture),
            sampleRateHz = json.optInt("sampleRateHz", fallback.sampleRateHz),
            codecFrameRateHz = json.optInt("codecFrameRateHz", fallback.codecFrameRateHz),
            audioTokenStart = json.optInt("audioTokenStart", fallback.audioTokenStart),
            eosTokenId = json.optInt("eosTokenId", fallback.eosTokenId),
            modelFiles = files,
            tokenizerFiles = tokenizerFiles,
            voices = voices
        )
    } catch (e: Exception) {
        null
    }

    // ── Presence / readiness (active-profile aware) ──────────────

    private fun requiredArtifactFilenames(p: ModelProfile) =
        p.modelFiles.map { it.filename } + p.tokenizerFiles

    fun requiredModels(context: Context): List<ModelFile> = activeProfile(context).modelFiles

    fun synthesisReady(context: Context): Boolean {
        val dir = activeModelDir(context)
        return activeProfile(context).modelFiles
            .filter { it.requiredForSynthesis }
            .all { File(dir, it.filename).exists() }
    }

    fun allModelsPresent(context: Context): Boolean {
        val dir = activeModelDir(context)
        return activeProfile(context).modelFiles.all { File(dir, it.filename).exists() }
    }

    fun missingOnnxModels(context: Context): List<String> {
        val dir = activeModelDir(context)
        return activeProfile(context).modelFiles
            .map { it.filename }.filter { !File(dir, it).exists() }
    }

    fun missingSynthesisModels(context: Context): List<String> {
        val dir = activeModelDir(context)
        return activeProfile(context).modelFiles
            .filter { it.requiredForSynthesis }
            .map { it.filename }.filter { !File(dir, it).exists() }
    }

    fun missingArtifacts(context: Context): List<String> {
        val dir = activeModelDir(context)
        return requiredArtifactFilenames(activeProfile(context)).filter { !File(dir, it).exists() }
    }
}
