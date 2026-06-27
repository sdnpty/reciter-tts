package com.qwen3.tts.util

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * On-device store for user-recorded voices (mic cloning references).
 *
 * Each voice is a 16 kHz mono WAV clip under `filesDir/custom_voices/<id>.wav`
 * plus an entry in `index.json`. When the AR engine is active these clips are
 * turned into speaker x-vectors by the speaker encoder; until then they are
 * persisted so no recording is lost.
 */
object CustomVoiceStore {

    data class CustomVoice(
        val id: String,
        val displayName: String,
        val locale: String,
        val clipPath: String
    ) {
        fun toVoiceSpec(): ModelConfig.VoiceSpec =
            ModelConfig.VoiceSpec(id = id, locale = locale, displayName = displayName)
    }

    private const val INDEX = "index.json"

    fun dir(context: Context): File =
        File(context.filesDir, "custom_voices").apply { mkdirs() }

    fun clipFile(context: Context, id: String): File = File(dir(context), "$id.wav")

    /** Cached speaker x-vector (float32 little-endian) for a custom voice. */
    fun xvecFile(context: Context, id: String): File = File(dir(context), "$id.xvec")

    fun saveXVector(context: Context, id: String, vec: FloatArray) {
        val bb = ByteBuffer.allocate(vec.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        vec.forEach { bb.putFloat(it) }
        xvecFile(context, id).writeBytes(bb.array())
    }

    /** All cached custom x-vectors, keyed by voice id (for the AR engine). */
    fun loadXVectors(context: Context): Map<String, FloatArray> {
        val out = LinkedHashMap<String, FloatArray>()
        list(context).forEach { v ->
            val f = xvecFile(context, v.id)
            if (f.exists()) {
                val raw = f.readBytes()
                val fb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
                out[v.id] = FloatArray(fb.limit()) { fb.get(it) }
            }
        }
        return out
    }

    fun list(context: Context): List<CustomVoice> {
        val idx = File(dir(context), INDEX)
        if (!idx.exists()) return emptyList()
        return runCatching {
            val arr = JSONArray(idx.readText())
            (0 until arr.length()).mapNotNull { i ->
                val o = arr.getJSONObject(i)
                val id = o.optString("id").ifEmpty { return@mapNotNull null }
                val clip = clipFile(context, id)
                if (!clip.exists()) return@mapNotNull null
                CustomVoice(
                    id = id,
                    displayName = o.optString("displayName", id),
                    locale = o.optString("locale", "ru-RU"),
                    clipPath = clip.absolutePath
                )
            }
        }.getOrDefault(emptyList())
    }

    /**
     * Registers a freshly recorded clip (already written to [clipFile]) under a
     * display name. Generates a stable id and returns the stored voice.
     */
    fun add(context: Context, displayName: String, locale: String = "ru-RU"): CustomVoice {
        val id = "custom_" + System.currentTimeMillis()
        val current = list(context).toMutableList()
        current.add(CustomVoice(id, displayName.trim().ifEmpty { id }, locale, clipFile(context, id).absolutePath))
        save(context, current)
        return current.last()
    }

    fun remove(context: Context, id: String) {
        clipFile(context, id).delete()
        xvecFile(context, id).delete()
        save(context, list(context).filterNot { it.id == id })
    }

    private fun save(context: Context, voices: List<CustomVoice>) {
        val arr = JSONArray()
        voices.forEach {
            arr.put(JSONObject().apply {
                put("id", it.id); put("displayName", it.displayName); put("locale", it.locale)
            })
        }
        File(dir(context), INDEX).writeText(arr.toString(2))
    }
}
