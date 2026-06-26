package com.qwen3.tts.util

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicBoolean

class TTSLogger private constructor(private val context: Context) {
    companion object {
        private const val TAG = "TTSLogger"
        private const val MAX_LOG_FILE_SIZE = 5 * 1024 * 1024 // 5 MB
        private const val MAX_LOG_FILES = 5
        private const val LOG_DIR = "logs"
        private const val LOG_FILE = "qwen3_tts.log"
        private const val CRASH_FILE = "crash.log"
        private const val CHECKPOINT_FILE = "load_checkpoint.txt"

        @Volatile
        private var instance: TTSLogger? = null

        fun getInstance(context: Context): TTSLogger {
            return instance ?: synchronized(this) {
                instance ?: TTSLogger(context.applicationContext).also { instance = it }
            }
        }
    }

    enum class Level { VERBOSE, DEBUG, INFO, WARNING, ERROR, FATAL }

    private val logQueue = ConcurrentLinkedQueue<LogEntry>()
    private val isRunning = AtomicBoolean(false)
    private val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US)
    private val logDateFormat = SimpleDateFormat("yyyyMMdd", Locale.US)

    private val logsDir: File
        get() = File(context.getExternalFilesDir(null), LOG_DIR).apply { if (!exists()) mkdirs() }

    private val currentLogFile: File
        get() = File(logsDir, LOG_FILE)

    data class LogEntry(
        val timestamp: Long,
        val level: Level,
        val tag: String,
        val message: String,
        val throwable: Throwable? = null
    )

    init {
        startWriter()
    }

    private fun startWriter() {
        if (isRunning.getAndSet(true)) return
        Thread {
            while (isRunning.get()) {
                val entry = logQueue.poll()
                if (entry != null) {
                    writeToFile(entry)
                } else {
                    Thread.sleep(100)
                }
            }
        }.apply { isDaemon = true; start() }
    }

    private fun writeToFile(entry: LogEntry) {
        try {
            rotateLogIfNeeded()
            val timeStr = dateFormat.format(Date(entry.timestamp))
            val threadName = Thread.currentThread().name
            val levelStr = entry.level.name.padEnd(7)
            val line = buildString {
                append("[$timeStr] ")
                append("[$levelStr] ")
                append("[${entry.tag}] ")
                append("[$threadName] ")
                append(entry.message)
                if (entry.throwable != null) {
                    append("\n")
                    append(Log.getStackTraceString(entry.throwable))
                }
                append("\n")
            }
            FileWriter(currentLogFile, true).use { it.write(line) }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to write log", e)
        }
    }

    private fun rotateLogIfNeeded() {
        val file = currentLogFile
        if (!file.exists() || file.length() < MAX_LOG_FILE_SIZE) return

        val backup = File(logsDir, "${LOG_FILE}.${logDateFormat.format(Date())}")
        if (!file.renameTo(backup)) {
            Log.w(TAG, "Failed to rotate log file to ${backup.name}")
        }

        // Keep only MAX_LOG_FILES
        logsDir.listFiles { _, name -> name.startsWith(LOG_FILE) && name != LOG_FILE }
            ?.sortedByDescending { it.lastModified() }
            ?.drop(MAX_LOG_FILES)
            ?.forEach { it.delete() }
    }

    fun v(tag: String, message: String) = log(Level.VERBOSE, tag, message)
    fun d(tag: String, message: String) = log(Level.DEBUG, tag, message)
    fun i(tag: String, message: String) = log(Level.INFO, tag, message)
    fun w(tag: String, message: String) = log(Level.WARNING, tag, message)
    fun e(tag: String, message: String, throwable: Throwable? = null) = log(Level.ERROR, tag, message, throwable)
    fun f(tag: String, message: String, throwable: Throwable? = null) = log(Level.FATAL, tag, message, throwable)

    private fun log(level: Level, tag: String, message: String, throwable: Throwable? = null) {
        // Also log to Android logcat
        when (level) {
            Level.VERBOSE -> Log.v(tag, message, throwable)
            Level.DEBUG -> Log.d(tag, message, throwable)
            Level.INFO -> Log.i(tag, message, throwable)
            Level.WARNING -> Log.w(tag, message, throwable)
            Level.ERROR -> Log.e(tag, message, throwable)
            Level.FATAL -> Log.e(tag, "FATAL: $message", throwable)
        }

        logQueue.offer(LogEntry(System.currentTimeMillis(), level, tag, message, throwable))
    }

    fun logInference(
        modelName: String,
        inputShape: String,
        outputShape: String,
        durationMs: Long,
        device: String
    ) {
        d("INFERENCE", "Model: $modelName | Input: $inputShape | Output: $outputShape | Time: ${durationMs}ms | Device: $device")
    }

    fun logAudio(
        sampleRate: Int,
        channels: Int,
        samples: Int,
        durationSec: Float,
        bufferSize: Int
    ) {
        d("AUDIO", "SR=$sampleRate Hz, CH=$channels, Samples=$samples, Duration=${String.format("%.2f", durationSec)}s, Buffer=$bufferSize")
    }

    fun logDownload(
        url: String,
        bytesDownloaded: Long,
        totalBytes: Long,
        speedKbps: Float
    ) {
        val percent = if (totalBytes > 0) (bytesDownloaded * 100 / totalBytes) else 0
        val sanitizedUrl = try { java.net.URL(url).let { "${it.protocol}://${it.host}${it.path}" } } catch (_: Exception) { "(invalid)" }
        d("DOWNLOAD", "URL: $sanitizedUrl | Progress: $percent% | ${bytesDownloaded/1024/1024}MB/${totalBytes/1024/1024}MB | Speed: ${String.format("%.1f", speedKbps)} KB/s")
    }

    suspend fun getLogs(): String = withContext(Dispatchers.IO) {
        val file = currentLogFile
        if (!file.exists()) return@withContext "No logs yet."
        file.readText()
    }
    
    suspend fun getLatestLogs(limit: Int = 100): List<String> = withContext(Dispatchers.IO) {
        val file = currentLogFile
        if (!file.exists()) return@withContext emptyList()
        try {
            file.readLines().takeLast(limit)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to read log file", e)
            listOf("[Error reading logs: ${e.message}]")
        }
    }

    suspend fun getLogFiles(): List<File> = withContext(Dispatchers.IO) {
        logsDir.listFiles { f -> f.name.endsWith(".log") }?.toList() ?: emptyList()
    }

    fun clearLogs() {
        logQueue.clear()
        logsDir.listFiles()?.forEach { it.delete() }
        i(TAG, "Logs cleared")
    }

    fun getLogFileForSharing(): File? {
        val file = currentLogFile
        return if (file.exists() && file.length() > 0) file else null
    }

    fun getCrashLog(): File? {
        val file = File(logsDir, CRASH_FILE)
        return if (file.exists()) file else null
    }

    // ── Crash-safe load breadcrumbs ──────────────────────────
    //
    // Native crashes inside ONNX Runtime (SIGSEGV / SIGABRT) terminate the
    // process without ever reaching the JVM crash handler or flushing the
    // async log queue — the user sees the app "instantly close with nothing in
    // the logs". To make those crashes diagnosable we write a tiny checkpoint
    // file *synchronously* right before each risky native call and delete it on
    // success. If the file is still there on next launch, the previous run died
    // exactly at that stage.

    private val checkpointFile: File
        get() = File(logsDir, CHECKPOINT_FILE)

    /** Synchronously record the stage we are about to enter (survives a native crash). */
    fun beginLoadStage(stage: String) {
        try {
            FileWriter(checkpointFile, false).use {
                it.write("${dateFormat.format(Date())}|$stage")
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to write load checkpoint", e)
        }
        i("LOAD", "▶ $stage")
    }

    /** Clear the checkpoint once the risky stage completed without crashing. */
    fun endLoadStage() {
        try {
            if (checkpointFile.exists()) checkpointFile.delete()
        } catch (_: Exception) {}
    }

    /**
     * If a previous run left a checkpoint behind, the process died (almost
     * always a native ONNX Runtime crash). Return a human-readable description
     * and clear it. Returns null if the last run shut down cleanly.
     */
    fun consumeStaleLoadCrash(): String? {
        val file = checkpointFile
        if (!file.exists()) return null
        val content = try { file.readText() } catch (_: Exception) { "" }
        try { file.delete() } catch (_: Exception) {}
        val parts = content.split("|", limit = 2)
        val stage = parts.getOrNull(1) ?: content
        return if (stage.isBlank()) null else stage
    }

    /** Synchronous, queue-bypassing write — guaranteed on disk before we return. */
    fun logSync(level: Level, tag: String, message: String) {
        log(level, tag, message)
        try {
            writeToFile(LogEntry(System.currentTimeMillis(), level, tag, message, null))
        } catch (_: Exception) {}
    }

    fun logCrash(throwable: Throwable) {
        f("CRASH", "Application crashed", throwable)
        try {
            FileWriter(File(logsDir, CRASH_FILE), true).use {
                it.write("${dateFormat.format(Date())}: ${Log.getStackTraceString(throwable)}\n\n")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to write crash log", e)
        }
    }

    fun getLogStats(): LogStats {
        val files = logsDir.listFiles { f -> f.name.endsWith(".log") } ?: emptyArray()
        val totalSize = files.sumOf { it.length() }
        val totalLines = files.sumOf { file ->
            try { file.readLines().size } catch (e: Exception) {
                Log.w(TAG, "Failed to count lines in ${file.name}", e)
                0
            }
        }
        return LogStats(files.size, totalSize, totalLines, currentLogFile.length())
    }

    data class LogStats(
        val fileCount: Int,
        val totalSizeBytes: Long,
        val totalLines: Int,
        val currentFileSize: Long
    )

    fun shutdown() {
        isRunning.set(false)
        // Flush remaining
        while (logQueue.isNotEmpty()) {
            logQueue.poll()?.let { writeToFile(it) }
        }
    }
}
