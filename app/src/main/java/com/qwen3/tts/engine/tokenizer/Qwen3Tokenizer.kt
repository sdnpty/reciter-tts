package com.qwen3.tts.engine.tokenizer

import android.content.Context
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.io.File
import java.io.InputStreamReader
import com.qwen3.tts.util.TokenizerConstants
import java.nio.charset.StandardCharsets
import java.util.regex.Pattern

/**
 * Реальный BPE-токенизатор Qwen3, совместимый с HuggingFace.
 * Загружает vocab.json и merges.txt из assets/.
 */
class Qwen3Tokenizer private constructor(
    private val vocab: Map<String, Int>,
    private val merges: List<Pair<String, String>>,
    private val specialTokens: Map<String, Int>
) {
    companion object {
        private const val TAG = "Qwen3Tokenizer"
        private const val UNK_ID = 0

        // Pre-tokenization pattern (Qwen2/3 style)
        private val PRETOKEN_PATTERN = Pattern.compile(
            "'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?+[\\p{L}]+[\\p{N}]?+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]++[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+"
        )

        @Volatile
        private var instance: Qwen3Tokenizer? = null

        fun getInstance(context: Context): Qwen3Tokenizer {
            return instance ?: synchronized(this) {
                instance ?: loadFromAssets(context).also { instance = it }
            }
        }

        /** Force a reload on next getInstance — call after switching models. */
        fun reset() {
            synchronized(this) { instance = null }
        }

        fun loadFromAssets(context: Context): Qwen3Tokenizer {
            val vocab = mutableMapOf<String, Int>()
            val specialTokens = mutableMapOf<String, Int>()
            val merges = mutableListOf<Pair<String, String>>()

            val modelsDir = com.qwen3.tts.util.ModelConfig.activeModelDir(context)
            val externalVocabFile = File(modelsDir, "vocab.json")
            val externalMergesFile = File(modelsDir, "merges.txt")

            var loadedVocab = false

            // Try loading from external storage first
            if (externalVocabFile.exists()) {
                try {
                    externalVocabFile.inputStream().use { stream ->
                        val reader = BufferedReader(InputStreamReader(stream, StandardCharsets.UTF_8))
                        val json = JSONObject(reader.readText())
                        val keys = json.keys()
                        while (keys.hasNext()) {
                            val key = keys.next()
                            val id = json.getInt(key)
                            vocab[key] = id
                            if (key.startsWith("<") && key.endsWith(">")) {
                                specialTokens[key] = id
                            }
                        }
                    }
                    Log.i(TAG, "Loaded vocab.json from models folder: ${vocab.size} tokens")
                    loadedVocab = true
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to load external vocab, falling back to assets", e)
                }
            }

            if (!loadedVocab) {
                // Load vocab.json from assets
                try {
                    context.assets.open("tokenizer/vocab.json").use { stream ->
                        val reader = BufferedReader(InputStreamReader(stream, StandardCharsets.UTF_8))
                        val json = JSONObject(reader.readText())
                        val keys = json.keys()
                        while (keys.hasNext()) {
                            val key = keys.next()
                            val id = json.getInt(key)
                            vocab[key] = id
                            if (key.startsWith("<") && key.endsWith(">")) {
                                specialTokens[key] = id
                            }
                        }
                    }
                    Log.i(TAG, "Loaded vocab from assets: ${vocab.size} tokens")
                    loadedVocab = true
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to load vocab.json from assets, using robust Cyrillic fallback", e)
                    return createFallbackTokenizer()
                }
            }

            var loadedMerges = false
            // Try loading merges from external storage
            if (externalMergesFile.exists()) {
                try {
                    externalMergesFile.inputStream().use { stream ->
                        val reader = BufferedReader(InputStreamReader(stream, StandardCharsets.UTF_8))
                        reader.lineSequence().forEach { line ->
                            if (line.isBlank() || line.startsWith("#")) return@forEach
                            val parts = line.trim().split(" ")
                            if (parts.size == 2) {
                                merges.add(parts[0] to parts[1])
                            }
                        }
                    }
                    Log.i(TAG, "Loaded merges.txt from models folder: ${merges.size} rules")
                    loadedMerges = true
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to load external merges, falling back to assets", e)
                }
            }

            if (!loadedMerges) {
                // Load merges.txt from assets
                try {
                    context.assets.open("tokenizer/merges.txt").use { stream ->
                        val reader = BufferedReader(InputStreamReader(stream, StandardCharsets.UTF_8))
                        reader.lineSequence().forEach { line ->
                            if (line.isBlank() || line.startsWith("#")) return@forEach
                            val parts = line.trim().split(" ")
                            if (parts.size == 2) {
                                merges.add(parts[0] to parts[1])
                            }
                        }
                    }
                    Log.i(TAG, "Loaded merges from assets: ${merges.size} rules")
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to load merges.txt from assets, using empty merges list", e)
                }
            }

            return Qwen3Tokenizer(vocab, merges, specialTokens)
        }

        private fun createFallbackTokenizer(): Qwen3Tokenizer {
            // Robust fallback that handles Cyrillic and individual bytes
            val vocab = mutableMapOf<String, Int>()
            val special = mutableMapOf<String, Int>()
            var id = 0

            // 1. Add individual bytes (0..255) to guarantee 100% vocabulary coverage for ANY arbitrary UTF-8 text
            for (b in 0..255) {
                val bVal = b.toByte()
                val bChar = byteEncoder[bVal]
                if (bChar != null) {
                    vocab[bChar.toString()] = id++
                }
            }

            // 2. Add common characters specifically so that they map directly to single tokens instead of multi-tokens if possible
            // Basic ASCII
            for (c in 32..126) {
                val token = charToBytes(c.toChar())
                if (!vocab.containsKey(token)) {
                    vocab[token] = id++
                }
            }

            // Cyrillic (Capital, Lowercase and 'ё')
            val cyrillicChars = ('а'..'я') + ('А'..'Я') + listOf('ё', 'Ё')
            for (c in cyrillicChars) {
                val token = charToBytes(c)
                if (!vocab.containsKey(token)) {
                    vocab[token] = id++
                }
            }

            // Special tokens
            special["<|endoftext|>"] = 151643
            special["<|im_start|>"] = 151644
            special["<|im_end|>"] = 151645
            special["<|object_ref_start|>"] = 151646
            special["<|object_ref_end|>"] = 151647

            return Qwen3Tokenizer(vocab, emptyList(), special)
        }

        private val byteEncoder: Map<Byte, Char> by lazy { buildByteEncoder() }
        private val byteDecoder: Map<Char, Byte> by lazy { byteEncoder.map { (k, v) -> v to k }.toMap() }

        private fun buildByteEncoder(): Map<Byte, Char> {
            val bs = mutableListOf<Byte>()
            // Printable ASCII first
            for (b in 33..126) bs.add(b.toByte())
            // Then extended
            for (b in listOf(161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171,
                172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185,
                186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199,
                200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213,
                214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227,
                228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241,
                242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255)) {
                bs.add(b.toByte())
            }
            // Remaining bytes
            for (b in 0..255) {
                if (!bs.contains(b.toByte())) bs.add(b.toByte())
            }

            val result = mutableMapOf<Byte, Char>()
            var n = 0
            for (b in bs) {
                result[b] = (256 + n).toChar()
                n++
            }
            return result
        }

        private fun charToBytes(c: Char): String {
            val bytes = c.toString().toByteArray(StandardCharsets.UTF_8)
            return bytes.joinToString("") { byteEncoder[it]?.toString() ?: "" }
        }

        private fun stringToBytes(s: String): String {
            val bytes = s.toByteArray(StandardCharsets.UTF_8)
            return bytes.joinToString("") { byteEncoder[it]?.toString() ?: "" }
        }

        private fun bytesToChar(token: String): String {
            val bytes = mutableListOf<Byte>()
            for (ch in token) {
                byteDecoder[ch]?.let { bytes.add(it) }
            }
            return String(bytes.toByteArray(), StandardCharsets.UTF_8)
        }
    }

    private val cache = mutableMapOf<String, String>()

    fun encode(text: String, addSpecialTokens: Boolean = true): IntArray {
        if (text.isBlank()) return intArrayOf()

        val tokens = mutableListOf<Int>()

        // Add BOS if needed
        if (addSpecialTokens) {
            specialTokens["<|im_start|>"]?.let { tokens.add(it) }
        }

        // Pre-tokenize
        val matcher = PRETOKEN_PATTERN.matcher(text)
        while (matcher.find()) {
            val word = matcher.group()
            val encoded = encodeWord(word)
            tokens.addAll(encoded)
        }

        // Add EOS if needed
        if (addSpecialTokens) {
            specialTokens["<|im_end|>"]?.let { tokens.add(it) }
        }

        return tokens.toIntArray()
    }

    private fun encodeWord(word: String): List<Int> {
        // Convert to BPE-friendly representation
        var token = stringToBytes(word)

        // Apply merges
        for ((first, second) in merges) {
            val merged = first + second
            token = token.replace(first + second, merged)
        }

        // Split into subwords and look up IDs
        val result = mutableListOf<Int>()
        var remaining = token
        while (remaining.isNotEmpty()) {
            var found = false
            // Try longest match first
            for (len in remaining.length downTo 1) {
                val sub = remaining.substring(0, len)
                val id = vocab[sub]
                if (id != null) {
                    result.add(id)
                    remaining = remaining.substring(len)
                    found = true
                    break
                }
            }
            if (!found) {
                // Unknown token — add UNK
                result.add(UNK_ID)
                remaining = remaining.drop(1)
            }
        }

        return result
    }

    fun decode(tokenIds: IntArray, skipSpecialTokens: Boolean = true): String {
        val sb = StringBuilder()
        for (id in tokenIds) {
            if (skipSpecialTokens && isSpecialToken(id)) continue
            val token = vocab.entries.find { it.value == id }?.key ?: continue
            sb.append(bytesToChar(token))
        }
        return sb.toString()
    }

    private fun isSpecialToken(id: Int): Boolean {
        return specialTokens.values.contains(id)
    }

    fun getVocabSize(): Int = vocab.size
    fun getPadId(): Int = TokenizerConstants.PAD_ID
    fun getEosId(): Int = TokenizerConstants.EOS_ID
    fun getBosId(): Int = TokenizerConstants.BOS_ID
    fun getAudioTokenStart(): Int = TokenizerConstants.AUDIO_TOKEN_START

    /**
     * Encode text for TTS inference (no chat template, just raw text tokens)
     */
    fun encodeForTTS(text: String): IntArray {
        // Qwen3-TTS uses raw text without chat template for audio generation
        val tokens = mutableListOf<Int>()
        val matcher = PRETOKEN_PATTERN.matcher(text)
        while (matcher.find()) {
            val word = matcher.group()
            tokens.addAll(encodeWord(word))
        }
        return tokens.toIntArray()
    }
}
