package com.qwen3.tts.util

import android.content.Context
import android.util.Log
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.LinkedBlockingDeque

/**
 * In-process TTS logger.
 *
 * Keeps the last [MAX_LINES] log lines in a circular in-memory queue and
 * appends them to a rolling log file in the app's cache directory so they
 * survive process death and can be shared via [LogShareHelper].
 *
 * Thread-safe: all public methods may be called from any thread.
 */
class TTSLogger private constructor(private val context: Context) {

    companion object {
        private const val TAG = "TTSLogger"
        private const val MAX_LINES = 2000
        private const val LOG_FILE = "reciter_tts.log"
        /** Shared-preference key written before each heavy native load stage. */
        private const val PREF_LOAD_STAGE = "load_stage_in_progress"
        private const val PREFS = "tts_logger_prefs"

        @Volatile
        private var instance: TTSLogger? = null

        fun getInstance(context: Context): TTSLogger =
            instance ?: synchronized(this) {
                instance ?: TTSLogger(context.applicationContext).also { instance = it }
            }
    }

    private val lines = LinkedBlockingDeque<String>(MAX_LINES)
    private val fmt = SimpleDateFormat("MM-dd HH:mm:ss.SSS", Locale.US)
    private val logFile: File = File(context.cacheDir, LOG_FILE)

    // ── Public logging API ────────────────────────────────────────────────────

    fun d(tag: String, msg: String) = log("D", tag, msg)
    fun i(tag: String, msg: String) = log("I", tag, msg)
    fun w(tag: String, msg: String) = log("W", tag, msg)
    fun e(tag: String, msg: String, throwable: Throwable? = null) {
        log("E", tag, msg)
        throwable?.let { log("E", tag, it.stackTraceToString().take(800)) }
    }

    // ── Load-stage crash detection ────────────────────────────────────────────

    /**
     * Call immediately before starting a native model-load stage.
     * The stage name is persisted so that if the process is killed by a
     * native crash (no JVM exception), the next launch can report it.
     */
    fun beginLoadStage(stage: String) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .edit().putString(PREF_LOAD_STAGE, stage).commit()
        i(TAG, ">> $stage")
    }

    /** Call immediately after the load stage succeeds. Clears the breadcrumb. */
    fun endLoadStage() {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .edit().remove(PREF_LOAD_STAGE).commit()
    }

    /**
     * Returns the stale load-stage name if a previous run crashed mid-load,
     * then clears it. Returns null if the previous run finished cleanly.
     */
    fun consumeStaleLoadCrash(): String? {
        val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val stage = prefs.getString(PREF_LOAD_STAGE, null) ?: return null
        prefs.edit().remove(PREF_LOAD_STAGE).apply()
        return stage
    }

    // ── Download progress ─────────────────────────────────────────────────────

    fun logDownload(url: String, downloaded: Long, total: Long, speedKbps: Float) {
        val mb = downloaded / 1024 / 1024
        val totalMb = total / 1024 / 1024
        val pct = if (total > 0) (downloaded * 100 / total).toInt() else 0
        d(TAG, "Download $pct% — ${mb}MB/${totalMb}MB @ ${speedKbps.toInt()} KB/s  [$url]")
    }

    // ── Crash logging ─────────────────────────────────────────────────────────

    fun logCrash(throwable: Throwable) {
        e(TAG, "CRASH: ${throwable.javaClass.simpleName}: ${throwable.message}", throwable)
    }

    // ── Log access ────────────────────────────────────────────────────────────

    /** Returns the latest [n] log lines (oldest first). */
    fun getLatestLogs(n: Int = MAX_LINES): List<String> {
        val snapshot = lines.toArray().mapNotNull { it as? String }
        return if (snapshot.size <= n) snapshot else snapshot.takeLast(n)
    }

    fun clearLogs() {
        lines.clear()
        try { logFile.delete() } catch (_: Exception) {}
    }

    /** Returns the log file for sharing. May be null if writing failed. */
    fun getLogFile(): File? = if (logFile.exists()) logFile else null

    // ── Internals ─────────────────────────────────────────────────────────────

    private fun log(level: String, tag: String, msg: String) {
        val line = "${fmt.format(Date())} $level/$tag: $msg"
        // Make space if full
        while (!lines.offerLast(line)) lines.pollFirst()
        // Mirror to file (best-effort, non-blocking write)
        try {
            logFile.appendText("$line\n", Charsets.UTF_8)
        } catch (_: Exception) {}
    }
}
