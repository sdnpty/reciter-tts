package com.qwen3.tts.util

import android.content.Context
import android.content.Intent
import androidx.core.content.FileProvider
import java.io.File

object LogShareHelper {

    fun createShareIntent(context: Context, logFile: File): Intent {
        val uri = FileProvider.getUriForFile(
            context, "${context.packageName}.fileprovider", logFile
        )

        return Intent(Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(Intent.EXTRA_STREAM, uri)
            putExtra(Intent.EXTRA_SUBJECT, "Reciter TTS Logs")
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
    }

    fun shareOrToast(context: Context, logger: TTSLogger, chooserTitle: String = "Share Logs") {
        val logFile = logger.getLogFileForSharing()
        if (logFile == null) {
            android.widget.Toast.makeText(context, com.qwen3.tts.R.string.toast_no_logs, android.widget.Toast.LENGTH_SHORT).show()
            return
        }
        val intent = createShareIntent(context, logFile)
        context.startActivity(Intent.createChooser(intent, chooserTitle))
    }
}
