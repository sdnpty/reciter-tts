package com.qwen3.tts.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.net.Uri
import android.os.Binder
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import androidx.core.app.NotificationCompat
import com.qwen3.tts.R
import com.qwen3.tts.ui.MainActivity
import com.qwen3.tts.util.ModelConfig
import com.qwen3.tts.util.TTSLogger
import kotlinx.coroutines.*
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URL
import java.util.zip.ZipInputStream
import kotlin.math.roundToInt

class ModelDownloadService : Service() {
    companion object {
        const val EXTRA_URL = "url"
        const val EXTRA_ZIP_URI = "zip_uri"
        const val EXTRA_FORCE = "force"

        private const val TAG = "ModelDownload"
        private const val NOTIFICATION_ID = 1001
        private const val CHANNEL_ID = "qwen3_model_download"
        private const val WAKELOCK_TAG = "Reciter::ModelDownload"
        private const val BUFFER_SIZE = 65536  // 64 KB buffer for better performance
        private const val CONNECT_TIMEOUT = 30000
        private const val READ_TIMEOUT = 60000
    }

    private val binder = LocalBinder()
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private lateinit var logger: TTSLogger
    private var wakeLock: PowerManager.WakeLock? = null
    private var downloadJob: Job? = null
    private var isCancelled = false

    var isDownloading = false
    var isImporting = false
    var currentPercent = 0
    var currentDownloaded = 0L
    var currentTotal = 0L
    var currentSpeed = 0f
    var currentTitle = ""

    var onProgress: ((Int, Long, Long, Float) -> Unit)? = null
    var onComplete: ((Boolean, String?) -> Unit)? = null

    inner class LocalBinder : Binder() {
        fun getService(): ModelDownloadService = this@ModelDownloadService
    }

    override fun onCreate() {
        super.onCreate()
        logger = TTSLogger.getInstance(this)
        createNotificationChannel()
        logger.i(TAG, "Service created")
    }

    override fun onBind(intent: Intent): IBinder = binder

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val url = intent?.getStringExtra(EXTRA_URL)
        val zipUri = intent?.getStringExtra(EXTRA_ZIP_URI)
        val force = intent?.getBooleanExtra(EXTRA_FORCE, false) == true
        if (url == null && zipUri == null) return START_NOT_STICKY

        isCancelled = false

        val initialText = if (zipUri != null) {
            isImporting = true
            isDownloading = false
            currentTitle = "Preparing model import…"
            "Preparing model import…"
        } else {
            isDownloading = true
            isImporting = false
            currentTitle = "Starting download…"
            "Starting download…"
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(
                NOTIFICATION_ID,
                createNotification(initialText, 0, 0),
                android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
            )
        } else {
            startForeground(NOTIFICATION_ID, createNotification(initialText, 0, 0))
        }
        acquireWakeLock()

        downloadJob = scope.launch {
            try {
                if (zipUri != null) {
                    importLocalZip(zipUri)
                } else {
                    downloadAndExtract(url ?: return@launch, force)
                }
            } catch (e: CancellationException) {
                logger.w(TAG, "Model transfer cancelled")
                onComplete?.invoke(false, "Cancelled")
            } catch (e: OutOfMemoryError) {
                logger.e(TAG, "Out of memory during model transfer", Exception(e))
                onComplete?.invoke(false, "Недостаточно памяти. Освободите место и попробуйте снова.")
            } catch (e: Exception) {
                logger.e(TAG, "Model transfer failed", e)
                onComplete?.invoke(false, e.message)
            } finally {
                isDownloading = false
                isImporting = false
                releaseWakeLock()
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf()
            }
        }

        return START_REDELIVER_INTENT
    }

    private suspend fun downloadAndExtract(url: String, force: Boolean) = withContext(Dispatchers.IO) {
        if (!url.startsWith("https://")) {
            throw IllegalArgumentException("Only HTTPS URLs are allowed for model downloads")
        }
        logger.i(TAG, "Starting model download")

        val root = ModelConfig.getModelsDirOrCreate(this@ModelDownloadService)
        val staging = File(root, "_incoming").apply { deleteRecursively(); mkdirs() }

        if (!isWifiConnected()) {
            logger.w(TAG, "Wi-Fi not connected, warning user about mobile data")
        }

        val tempFile = File(cacheDir, "models_temp.zip")
        try {
            val connection = URL(url).openConnection() as HttpURLConnection
            connection.apply {
                connectTimeout = CONNECT_TIMEOUT
                readTimeout = READ_TIMEOUT
                setRequestProperty("Accept", "application/zip, application/octet-stream")
                setRequestProperty("User-Agent", "Reciter-TTS-Android/1.0")
            }

            val responseCode = connection.responseCode
            if (responseCode !in 200..299) {
                val errorBody = try {
                    connection.errorStream?.bufferedReader()?.readText()?.take(500) ?: "no body"
                } catch (_: Exception) { "unreadable" }
                connection.disconnect()
                throw java.io.IOException("HTTP $responseCode from $url: $errorBody")
            }

            val totalSize = connection.contentLengthLong
            logger.i(TAG, "Total size: ${totalSize / 1024 / 1024} MB")

            var downloaded = 0L
            var lastUpdate = System.currentTimeMillis()
            var lastDownloaded = 0L

            connection.inputStream.use { input ->
                FileOutputStream(tempFile).use { output ->
                    val buffer = ByteArray(BUFFER_SIZE)
                    var bytesRead: Int

                    while (input.read(buffer).also { bytesRead = it } != -1) {
                        if (isCancelled) throw CancellationException()
                        output.write(buffer, 0, bytesRead)
                        downloaded += bytesRead

                        val now = System.currentTimeMillis()
                        if (now - lastUpdate > 500) {
                            val elapsed = (now - lastUpdate) / 1000f
                            val speed = if (elapsed > 0) (downloaded - lastDownloaded) / 1024f / elapsed else 0f
                            val percent = if (totalSize > 0) ((downloaded * 100) / totalSize).toInt() else 0
                            currentPercent = percent
                            currentDownloaded = downloaded
                            currentTotal = totalSize
                            currentSpeed = speed
                            currentTitle = "Downloading..."

                            logger.logDownload(url, downloaded, totalSize, speed)
                            updateNotification(percent, downloaded, totalSize, speed)
                            onProgress?.invoke(percent, downloaded, totalSize, speed)

                            lastUpdate = now
                            lastDownloaded = downloaded
                        }
                    }
                }
            }

            connection.disconnect()
            logger.i(TAG, "Download complete: ${downloaded / 1024 / 1024} MB")

            updateNotificationText("Extracting models…")
            extractZipStreaming(tempFile, staging)
        } finally {
            tempFile.delete()
        }

        val slotId = finalizeSlot(staging)
        val missing = ModelConfig.missingOnnxModels(this@ModelDownloadService)
        if (missing.isNotEmpty()) {
            throw IllegalStateException("Missing ONNX files after extraction: ${missing.joinToString()}")
        }

        logger.i(TAG, "Extraction complete into slot '$slotId', all models verified")
        onComplete?.invoke(true, null)
    }

    /**
     * Move freshly-extracted files from the staging dir into a model slot named
     * after the manifest id (or a generated id), and make it the active model.
     */
    private fun finalizeSlot(staging: File): String {
        val root = ModelConfig.getModelsDir(this)
        val manifestFile = File(staging, ModelConfig.MANIFEST_FILE)
        val rawId = if (manifestFile.exists()) {
            try {
                org.json.JSONObject(manifestFile.readText()).optString("id").takeIf { it.isNotBlank() }
            } catch (_: Exception) { null }
        } else null

        val slotId = (rawId ?: "model_${System.currentTimeMillis()}")
            .replace(Regex("[^A-Za-z0-9._-]"), "_")
        val target = File(root, slotId)
        if (target.exists()) target.deleteRecursively()

        if (!staging.renameTo(target)) {
            staging.copyRecursively(target, overwrite = true)
            staging.deleteRecursively()
        }

        ModelConfig.setActiveModel(this, slotId)
        logger.i(TAG, "Installed model into slot '$slotId'")
        return slotId
    }

    private suspend fun importLocalZip(uriString: String) = withContext(Dispatchers.IO) {
        val uri = Uri.parse(uriString)
        val root = ModelConfig.getModelsDirOrCreate(this@ModelDownloadService)
        val staging = File(root, "_incoming").apply { deleteRecursively(); mkdirs() }
        logger.i(TAG, "Starting local ZIP import from: $uri")

        // First pass: count entries
        var totalOnnxFiles = 0
        try {
            contentResolver.openInputStream(uri)?.use { input ->
                totalOnnxFiles = countZipEntries(input)
            }
        } catch (e: Exception) {
            logger.w(TAG, "Failed to count ZIP entries: ${e.message}")
        }
        logger.i(TAG, "Found $totalOnnxFiles ONNX files in ZIP")

        // Second pass: extract into staging
        try {
            contentResolver.openInputStream(uri)?.use { input ->
                extractZipStreaming(input, staging, totalOnnxFiles)
            } ?: throw IllegalStateException("Could not open input stream for $uri")

            logger.i(TAG, "ZIP extraction finished successfully")
        } catch (e: Exception) {
            logger.e(TAG, "Error during ZIP extraction", e)
            throw e
        }

        val slotId = finalizeSlot(staging)
        val missing = ModelConfig.missingOnnxModels(this@ModelDownloadService)
        if (missing.isNotEmpty()) {
            logger.w(TAG, "Still missing models after import: ${missing.joinToString()}")
            throw IllegalStateException("Import finished but some models are missing: ${missing.joinToString()}")
        }

        logger.i(TAG, "Local model import complete into slot '$slotId' and verified")
        onComplete?.invoke(true, null)
    }

    private fun countZipEntries(input: InputStream): Int {
        var count = 0
        try {
            ZipInputStream(input.buffered(BUFFER_SIZE)).use { zis ->
                var entry = zis.nextEntry
                while (entry != null) {
                    val baseName = entry.name.substringAfterLast("/")
                    val isOnnx = baseName.endsWith(".onnx", ignoreCase = true)
                    val isTokenizer = baseName == "vocab.json" || baseName == "merges.txt"
                    
                    if (!entry.isDirectory && (isOnnx || isTokenizer)) {
                        count++
                    }
                    zis.closeEntry()
                    entry = zis.nextEntry
                }
            }
        } catch (e: Exception) {
            // Partial count is fine
        }
        return count
    }

    /**
     * Streaming ZIP extraction — never loads a full model into memory.
     * Works for both File input and InputStream (URI) input.
     */
    private fun extractZipStreaming(source: File, destDir: File, totalEntries: Int = 0) {
        extractZipStreaming(source.inputStream().buffered(BUFFER_SIZE), destDir, totalEntries)
    }

    private fun extractZipStreaming(input: InputStream, destDir: File, totalEntries: Int = 0) {
        val renameMap = ModelConfig.ARCHIVE_RENAME_MAP
        val canonicalDestDir = destDir.canonicalFile
        var extracted = 0
        val writeBuffer = ByteArray(BUFFER_SIZE)

        ZipInputStream(if (input is java.io.BufferedInputStream) input else input.buffered(BUFFER_SIZE)).use { zis ->
            var entry = zis.nextEntry
            while (entry != null) {
                if (isCancelled) {
                    logger.w(TAG, "Extraction cancelled after $extracted files")
                    break
                }

                if (!entry.isDirectory) {
                    logger.i(TAG, "Processing ZIP entry: ${entry.name}")

                    val baseName = entry.name.substringAfterLast("/")

                    // Extract ONNX models, tokenizer files, and the model manifest
                    val isOnnx = baseName.endsWith(".onnx", ignoreCase = true)
                    val isTokenizer = baseName == "vocab.json" || baseName == "merges.txt"
                    val isManifest = baseName == ModelConfig.MANIFEST_FILE

                    if (!isOnnx && !isTokenizer && !isManifest) {
                        logger.d(TAG, "Skipping non-relevant file: ${entry.name}")
                        zis.closeEntry()
                        entry = zis.nextEntry
                        continue
                    }

                    // Skip invalid names silently
                    if (baseName.isEmpty() || baseName.contains("..")) {
                        zis.closeEntry()
                        entry = zis.nextEntry
                        continue
                    }

                    val destName = renameMap[baseName] ?: baseName
                    val outFile = File(canonicalDestDir, destName)

                    // Security: verify destination is inside destDir
                    if (!outFile.canonicalPath.startsWith(canonicalDestDir.canonicalPath)) {
                        logger.w(TAG, "ZIP path traversal blocked: ${entry.name}")
                        zis.closeEntry()
                        entry = zis.nextEntry
                        continue
                    }

                    outFile.parentFile?.mkdirs()

                    logger.i(TAG, "Extracting: ${entry.name} → ${outFile.name}")

                    try {
                        FileOutputStream(outFile).use { fos ->
                            var bytesRead: Int
                            while (zis.read(writeBuffer).also { bytesRead = it } != -1) {
                                if (isCancelled) break
                                fos.write(writeBuffer, 0, bytesRead)
                            }
                        }
                        // If cancelled mid-file, delete partial file
                        if (isCancelled) {
                            outFile.delete()
                            break
                        }
                    } catch (e: OutOfMemoryError) {
                        outFile.delete()
                        logger.e(TAG, "OOM while extracting ${entry.name}", Exception(e))
                        throw Exception("Out of memory extracting ${entry.name}. Free up device storage and try again.")
                    } catch (e: Exception) {
                        outFile.delete()
                        logger.e(TAG, "Failed to extract ${entry.name}: ${e.message}", e)
                        // Continue with next entry rather than aborting entire ZIP
                        zis.closeEntry()
                        entry = zis.nextEntry
                        continue
                    }

                    extracted++

                    if (totalEntries > 0) {
                        val percent = (extracted * 100) / totalEntries
                        currentPercent = percent
                        currentDownloaded = extracted.toLong()
                        currentTotal = totalEntries.toLong()
                        currentTitle = "Importing models… $extracted / $totalEntries"
                        updateNotification(currentTitle, percent, 100)
                        onProgress?.invoke(percent, extracted.toLong(), totalEntries.toLong(), 0f)
                    } else {
                        // Indeterminate progress
                        currentTitle = "Importing models… $extracted files"
                        updateNotificationText(currentTitle)
                        onProgress?.invoke(0, extracted.toLong(), 0L, 0f)
                    }
                }

                try {
                    zis.closeEntry()
                } catch (_: Exception) {}
                entry = try { zis.nextEntry } catch (_: Exception) { null }
            }
        }
        logger.i(TAG, "Extraction finished. Total extracted: $extracted")
    }

    fun cancelDownload() {
        isCancelled = true
        downloadJob?.cancel()
        logger.i(TAG, "Download cancellation requested")
    }

    private fun acquireWakeLock() {
        if (wakeLock?.isHeld == true) return
        // Defensive: without the WAKE_LOCK permission acquire() throws a
        // SecurityException that crashes the whole service start before the ZIP
        // is even unpacked. The import works without the wake lock (just less
        // resilient to doze), so degrade gracefully instead of crashing.
        try {
            val powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
            wakeLock = powerManager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, WAKELOCK_TAG).apply {
                acquire(60L * 60L * 1000L)
            }
        } catch (t: Throwable) {
            logger.w(TAG, "Wake lock unavailable, continuing without it: ${t.message}")
            wakeLock = null
        }
    }

    private fun releaseWakeLock() {
        try {
            val lock = wakeLock
            if (lock?.isHeld == true) lock.release()
        } catch (_: Throwable) {}
        wakeLock = null
    }

    private fun isWifiConnected(): Boolean {
        val connectivityManager = getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        val network = connectivityManager.activeNetwork ?: return false
        val capabilities = connectivityManager.getNetworkCapabilities(network) ?: return false
        return capabilities.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) ||
            capabilities.hasTransport(NetworkCapabilities.TRANSPORT_ETHERNET)
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Model Downloads",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Downloads ONNX model files for Reciter TTS"
                setShowBadge(false)
            }
            val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            nm.createNotificationChannel(channel)
        }
    }

    private fun createNotification(text: String, progress: Int, max: Int): Notification {
        val pendingIntent = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE
        )

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Reciter TTS — Downloading Models")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setProgress(max, progress, progress == 0 && max == 0)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .build()
    }

    private fun updateNotification(percent: Int, downloaded: Long, total: Long, speed: Float) {
        val text = "${percent}% • ${downloaded / 1024 / 1024}MB / ${total / 1024 / 1024}MB • ${speed.roundToInt()} KB/s"
        updateNotification(text, percent, 100)
    }

    private fun updateNotification(text: String, progress: Int, max: Int) {
        val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        nm.notify(NOTIFICATION_ID, createNotification(text, progress, max))
    }

    private fun updateNotificationText(text: String) {
        val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        nm.notify(NOTIFICATION_ID, createNotification(text, 0, 0))
    }

    override fun onDestroy() {
        super.onDestroy()
        scope.cancel()
        releaseWakeLock()
        logger.i(TAG, "Service destroyed")
    }
}
