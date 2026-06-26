package com.qwen3.tts.util

import android.content.Context
import android.content.Intent
import android.widget.Toast
import androidx.core.content.FileProvider
import com.qwen3.tts.R

/**
 * Utility for sharing the in-process TTS log file via the Android share sheet.
 *
 * The log file is exposed through the app's FileProvider so it can be sent
 * to external apps (e-mail, messaging, etc.) without storage permissions.
 */
object LogShareHelper {

    /**
     * Attempts to share the log file via [Intent.ACTION_SEND]. Falls back to a
     * [Toast] if the file is empty, does not exist, or no app can receive it.
     */
    fun shareOrToast(context: Context, logger: TTSLogger) {
        val file = logger.getLogFile()
        if (file == null || !file.exists() || file.length() == 0L) {
            Toast.makeText(context, R.string.toast_no_logs, Toast.LENGTH_SHORT).show()
            return
        }

        try {
            val uri = FileProvider.getUriForFile(
                context,
                "${context.packageName}.fileprovider",
                file
            )
            val intent = Intent(Intent.ACTION_SEND).apply {
                type = "text/plain"
                putExtra(Intent.EXTRA_STREAM, uri)
                putExtra(Intent.EXTRA_SUBJECT, "Reciter TTS Log")
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            context.startActivity(Intent.createChooser(intent, "Share log").apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            })
        } catch (e: Exception) {
            Toast.makeText(context, R.string.toast_share_failed, Toast.LENGTH_SHORT).show()
        }
    }
}
