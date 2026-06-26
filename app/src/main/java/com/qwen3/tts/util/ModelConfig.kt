package com.qwen3.tts.util

import android.content.Context
import org.json.JSONObject
import java.io.File
import java.util.Locale

/**
 * Central configuration and discovery hub for on-device TTS models.
 *
 * Models are installed in `<filesDir>/models/<slot_id>/` where each slot
 * contains an optional `model_manifest.json` plus ONNX files. When no
 * manifest is present a built-in default profile is used so the engine
 * can still attempt to load models by conventional filenames.
 */
object ModelConfig {

    // ── Constants ─────────────────────────────────────────────────────────────

    const val MANIFEST_FILE = "model_manifest.json"
    private const val PREFS = "qwen3_tts_prefs"
    private const val KEY_ACTIVE_MODEL = "active_model_slot"

    /**
     * Renames applied to ZIP entry basenames during extraction so that
     * models with varying export names land under a canonical filename.
     */
    val ARCHIVE_RENAME_MAP: Map<String, String> = emptyMap()

    // ── Data classes ──────────────────────────────────────────────────────────

    enum class Role {
        TALKER, CODE_PREDICTOR, VOCODER, SPEAKER_ENCODER, LM_HEAD, UNKNOWN
    }

    data class ModelFileSpec(
        val filename: String,
        val role: Role = Role.UNKNOWN,
        val requiredForSynthesis: Boolean = true,
        val expectedSizeMb: Int = 0
    )

    data class VoiceSpec(
        val id: String,
        val displayName: String,
        val languageCode: String,   // BCP-47 primary language, e.g. "ru"
        val countryCode: String = "",
        val quality: String = "very_high"
    ) {
        fun toLocale(): Locale = if (countryCode.isNotBlank())
            Locale(languageCode, countryCode)
        else
            Locale(languageCode)

        /** Returns the primary language tag (e.g. "ru", "en", "zh"). */
        fun languageTag(): String = languageCode
    }

    data class ModelProfile(
        val id: String,
        val displayName: String,
        val architecture: String = "qwen3-codec",
        val sampleRateHz: Int = 24_000,
        val audioTokenStart: Int = TokenizerConstants.AUDIO_TOKEN_START,
        val eosTokenId: Int = TokenizerConstants.EOS_ID,
        val modelFiles: List<ModelFileSpec> = emptyList(),
        val voices: List<VoiceSpec> = emptyList(),
        val tokenizerFiles: List<String> = listOf("vocab.json", "merges.txt"),
        val languages: List<String> = emptyList()
    ) {
        fun defaultVoice(): VoiceSpec? = voices.firstOrNull()
        fun voiceById(id: String): VoiceSpec? = voices.firstOrNull { it.id == id }
        fun voiceForLanguage(lang: String): VoiceSpec? =
            voices.firstOrNull { it.languageCode.equals(lang, ignoreCase = true) }
    }

    data class InstalledModel(
        val id: String,
        val dir: File,
        val profile: ModelProfile
    )

    // ── Default / fallback profile ────────────────────────────────────────────

    private val DEFAULT_PROFILE = ModelProfile(
        id = "qwen3_tts_base",
        displayName = "Qwen3-TTS Base",
        architecture = "qwen3-codec",
        sampleRateHz = 24_000,
        audioTokenStart = TokenizerConstants.AUDIO_TOKEN_START,
        eosTokenId = TokenizerConstants.EOS_ID,
        modelFiles = listOf(
            ModelFileSpec("talker_base_android.onnx", Role.TALKER, true, 428),
            ModelFileSpec("code_predictor_base_android.onnx", Role.CODE_PREDICTOR, false, 60),
            ModelFileSpec("code2wav_android.onnx", Role.VOCODER, true, 120),
            ModelFileSpec("speaker_encoder_android.onnx", Role.SPEAKER_ENCODER, false, 30)
        ),
        voices = listOf(
            VoiceSpec("qwen3_ru", "Russian", "ru", "RU", "very_high"),
            VoiceSpec("qwen3_en", "English", "en", "US", "very_high"),
            VoiceSpec("qwen3_zh", "Chinese", "zh", "CN", "very_high")
        ),
        tokenizerFiles = listOf("vocab.json", "merges.txt"),
        languages = listOf("ru", "en", "zh")
    )

    // ── Profile cache ─────────────────────────────────────────────────────────

    @Volatile
    private var cachedProfile: ModelProfile? = null

    fun invalidateProfileCache() {
        cachedProfile = null
    }

    // ── Directory helpers ─────────────────────────────────────────────────────

    fun getModelsDir(context: Context): File =
        File(context.filesDir, "models")

    fun getModelsDirOrCreate(context: Context): File =
        getModelsDir(context).also { it.mkdirs() }

    fun activeModelDir(context: Context): File {
        val root = getModelsDir(context)
        val slotId = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .getString(KEY_ACTIVE_MODEL, null)
        return if (slotId != null) File(root, slotId) else root
    }

    fun getModelFile(context: Context, filename: String): File =
        File(activeModelDir(context), filename)

    // ── Active model ──────────────────────────────────────────────────────────

    fun setActiveModel(context: Context, slotId: String) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .edit().putString(KEY_ACTIVE_MODEL, slotId).apply()
        invalidateProfileCache()
    }

    fun activeModel(context: Context): InstalledModel? {
        val all = installedModels(context)
        if (all.isEmpty()) return null
        val slotId = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .getString(KEY_ACTIVE_MODEL, null)
        return all.firstOrNull { it.id == slotId } ?: all.first()
    }

    fun activeProfile(context: Context): ModelProfile {
        cachedProfile?.let { return it }
        val profile = loadProfileForDir(activeModelDir(context))
        cachedProfile = profile
        return profile
    }

    // ── Installed model discovery ─────────────────────────────────────────────

    fun installedModels(context: Context): List<InstalledModel> {
        val root = getModelsDir(context)
        if (!root.exists()) return emptyList()
        // A slot is any subdirectory of root, or root itself if it directly contains ONNX files.
        val subdirs = root.listFiles { f -> f.isDirectory } ?: emptyArray()
        val result = mutableListOf<InstalledModel>()
        for (dir in subdirs) {
            val hasOnnx = dir.listFiles { f -> f.extension.equals("onnx", ignoreCase = true) }
                ?.isNotEmpty() == true
            val hasManifest = File(dir, MANIFEST_FILE).exists()
            if (hasOnnx || hasManifest) {
                val profile = loadProfileForDir(dir)
                result.add(InstalledModel(dir.name, dir, profile))
            }
        }
        // Also check root dir directly (legacy single-model layout)
        if (result.isEmpty()) {
            val rootOnnx = root.listFiles { f -> f.extension.equals("onnx", ignoreCase = true) }
                ?.isNotEmpty() == true
            if (rootOnnx) {
                result.add(InstalledModel("default", root, loadProfileForDir(root)))
            }
        }
        return result
    }

    // ── Validation ────────────────────────────────────────────────────────────

    fun synthesisReady(context: Context): Boolean =
        missingOnnxModels(context).isEmpty()

    fun missingOnnxModels(context: Context): List<String> {
        val profile = activeProfile(context)
        val dir = activeModelDir(context)
        return profile.modelFiles
            .filter { it.requiredForSynthesis }
            .map { it.filename }
            .filter { !File(dir, it).exists() }
    }

    fun missingSynthesisModels(context: Context): List<String> = missingOnnxModels(context)

    // ── Manifest parsing ──────────────────────────────────────────────────────

    private fun loadProfileForDir(dir: File): ModelProfile {
        val manifest = File(dir, MANIFEST_FILE)
        if (!manifest.exists()) return DEFAULT_PROFILE
        return try {
            parseManifest(manifest.readText(), dir)
        } catch (e: Exception) {
            DEFAULT_PROFILE
        }
    }

    private fun parseManifest(json: String, dir: File): ModelProfile {
        val obj = JSONObject(json)
        val id = obj.optString("id", "unknown")
        val displayName = obj.optString("display_name", id)
        val architecture = obj.optString("architecture", "qwen3-codec")
        val sampleRate = obj.optInt("sample_rate_hz", 24_000)
        val audioTokenStart = obj.optInt("audio_token_start", TokenizerConstants.AUDIO_TOKEN_START)
        val eosTokenId = obj.optInt("eos_token_id", TokenizerConstants.EOS_ID)

        // Parse model files
        val modelFiles = mutableListOf<ModelFileSpec>()
        val filesArray = obj.optJSONArray("model_files")
        if (filesArray != null) {
            for (i in 0 until filesArray.length()) {
                val mf = filesArray.getJSONObject(i)
                val filename = mf.getString("filename")
                val roleStr = mf.optString("role", "UNKNOWN").uppercase()
                val role = try { Role.valueOf(roleStr) } catch (_: Exception) { Role.UNKNOWN }
                val required = mf.optBoolean("required_for_synthesis", true)
                val expectedSizeMb = mf.optInt("expected_size_mb", 0)
                modelFiles.add(ModelFileSpec(filename, role, required, expectedSizeMb))
            }
        } else {
            // Fall back to scanning the directory for ONNX files
            dir.listFiles { f -> f.extension.equals("onnx", ignoreCase = true) }
                ?.forEach { f ->
                    modelFiles.add(ModelFileSpec(f.name, Role.UNKNOWN, true, (f.length() / 1024 / 1024).toInt()))
                }
        }

        // Parse voices
        val voices = mutableListOf<VoiceSpec>()
        val voicesArray = obj.optJSONArray("voices")
        if (voicesArray != null) {
            for (i in 0 until voicesArray.length()) {
                val v = voicesArray.getJSONObject(i)
                voices.add(
                    VoiceSpec(
                        id = v.getString("id"),
                        displayName = v.optString("display_name", v.getString("id")),
                        languageCode = v.optString("language", "ru"),
                        countryCode = v.optString("country", ""),
                        quality = v.optString("quality", "very_high")
                    )
                )
            }
        } else {
            voices.addAll(DEFAULT_PROFILE.voices)
        }

        val tokenizerFiles = mutableListOf<String>()
        val tokArray = obj.optJSONArray("tokenizer_files")
        if (tokArray != null) {
            for (i in 0 until tokArray.length()) tokenizerFiles.add(tokArray.getString(i))
        } else {
            tokenizerFiles.addAll(DEFAULT_PROFILE.tokenizerFiles)
        }

        val languages = voices.map { it.languageCode }.distinct()

        return ModelProfile(
            id = id,
            displayName = displayName,
            architecture = architecture,
            sampleRateHz = sampleRate,
            audioTokenStart = audioTokenStart,
            eosTokenId = eosTokenId,
            modelFiles = modelFiles,
            voices = voices,
            tokenizerFiles = tokenizerFiles,
            languages = languages
        )
    }
}
