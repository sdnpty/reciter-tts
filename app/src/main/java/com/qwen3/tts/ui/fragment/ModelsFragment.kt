package com.qwen3.tts.ui.fragment

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.media.AudioFormat
import android.media.AudioTrack
import android.net.Uri
import android.os.Bundle
import android.os.IBinder
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.lifecycleScope
import com.google.android.material.chip.Chip
import com.qwen3.tts.R
import com.qwen3.tts.databinding.FragmentModelsBinding
import com.qwen3.tts.service.ModelDownloadService
import com.qwen3.tts.ui.TTSViewModel
import com.qwen3.tts.util.AudioHelper
import com.qwen3.tts.util.ModelConfig
import com.qwen3.tts.util.TTSLogger
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.text.DecimalFormat
import kotlin.math.PI
import kotlin.math.sin

class ModelsFragment : Fragment(R.layout.fragment_models) {
    companion object {
        private const val TAG = "ModelsFragment"
        private const val SAMPLE_RATE = 24000
    }

    private var _binding: FragmentModelsBinding? = null
    private val binding get() = _binding!!

    private val viewModel by lazy {
        ViewModelProvider(requireActivity())[TTSViewModel::class.java]
    }
    private lateinit var logger: TTSLogger

    private var downloadService: ModelDownloadService? = null
    private var isBound = false
    private var isLocalImport = false
    private var audioTrack: AudioTrack? = null

    private val pickZipLauncher = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri ->
        uri?.let { startLocalImport(it) }
    }

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            val binder = service as ModelDownloadService.LocalBinder
            downloadService = binder.getService()
            isBound = true
            setupDownloadCallbacks()
            restoreProgressIfActive()
        }
        override fun onServiceDisconnected(name: ComponentName?) {
            downloadService = null
            isBound = false
        }
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        _binding = FragmentModelsBinding.bind(view)
        logger = TTSLogger.getInstance(requireContext())

        refreshModelStatus()

        binding.btnDownload.setOnClickListener { showDownloadDialog() }
        binding.btnPickZip.setOnClickListener { pickZipLauncher.launch(arrayOf("application/zip")) }
        binding.btnDeleteModels.setOnClickListener { showDeleteDialog() }
        binding.btnStopDownload.setOnClickListener { stopDownload() }
        binding.btnTestSine.setOnClickListener { playSineWave() }
        binding.btnTestSweep.setOnClickListener { playFrequencySweep() }
        binding.btnStopAudio.setOnClickListener { stopAudio() }
    }

    // chipId -> model slot id
    private val modelChips = mutableMapOf<Int, String>()

    /**
     * Builds a chip per installed model so the user can pick which one is
     * active right on the Models screen. The active model drives the status
     * card below; downloading/importing an archive adds (and activates) a slot.
     */
    private fun setupModelPicker() {
        val ctx = context ?: return
        if (_binding == null) return
        val installed = ModelConfig.installedModels(ctx)
        binding.chipGroupModels.setOnCheckedStateChangeListener(null)
        binding.chipGroupModels.removeAllViews()
        modelChips.clear()

        if (installed.isEmpty()) {
            binding.tvNoModels.visibility = View.VISIBLE
            return
        }
        binding.tvNoModels.visibility = View.GONE

        val activeId = ModelConfig.activeModel(ctx)?.id
        installed.forEach { model ->
            val chip = layoutInflater.inflate(R.layout.item_voice_chip, binding.chipGroupModels, false) as Chip
            chip.id = View.generateViewId()
            val langs = model.profile.languages.joinToString(",")
            val ready = isModelReady(model)
            val name = if (langs.isBlank()) model.profile.displayName else "${model.profile.displayName} · $langs"
            chip.text = if (ready) "✓ $name" else "⬇ $name"
            chip.isChecked = model.id == activeId
            binding.chipGroupModels.addView(chip)
            modelChips[chip.id] = model.id
        }

        binding.chipGroupModels.setOnCheckedStateChangeListener { _, checkedIds ->
            val id = checkedIds.firstOrNull()?.let { modelChips[it] } ?: return@setOnCheckedStateChangeListener
            if (id != ModelConfig.activeModel(requireContext())?.id) {
                ModelConfig.setActiveModel(requireContext(), id)
                com.qwen3.tts.engine.tokenizer.Qwen3Tokenizer.reset()
                logger.i(TAG, "Active model switched to '$id'")
                Toast.makeText(requireContext(), R.string.model_switch_hint, Toast.LENGTH_LONG).show()
                // Rebuild chips + status after this callback returns, so we don't
                // mutate the ChipGroup from inside its own selection callback.
                binding.chipGroupModels.post { refreshModelStatus() }
            }
        }
    }

    /** True when every required ONNX file of [model]'s profile exists in its slot dir. */
    private fun isModelReady(model: ModelConfig.InstalledModel): Boolean =
        model.profile.modelFiles
            .filter { it.requiredForSynthesis }
            .all { File(model.dir, it.filename).exists() }

    override fun onStart() {
        super.onStart()
        val intent = Intent(requireContext(), ModelDownloadService::class.java)
        requireContext().bindService(intent, serviceConnection, Context.BIND_AUTO_CREATE)
    }

    override fun onStop() {
        super.onStop()
        if (isBound) {
            requireContext().unbindService(serviceConnection)
            isBound = false
        }
    }

    override fun onResume() {
        super.onResume()
        refreshModelStatus()
        restoreProgressIfActive()
    }

    override fun onHiddenChanged(hidden: Boolean) {
        super.onHiddenChanged(hidden)
        if (!hidden) {
            refreshModelStatus()
            restoreProgressIfActive()
        }
    }

    private fun restoreProgressIfActive() {
        val service = downloadService ?: return
        if (_binding == null) return
        if (service.isDownloading || service.isImporting) {
            isLocalImport = service.isImporting
            showProgress(service.currentTitle)
            binding.progressBar.isIndeterminate = false
            binding.progressBar.progress = service.currentPercent
            
            val percent = service.currentPercent
            val downloaded = service.currentDownloaded
            val total = service.currentTotal
            
            binding.tvDownloadStatus.text = if (isLocalImport) {
                getString(R.string.extracting_progress, downloaded.toInt(), total.toInt())
            } else {
                getString(R.string.dialog_download_progress, percent, "${downloaded / 1024 / 1024}MB", "${total / 1024 / 1024}MB")
            }
            binding.btnStopDownload.visibility = View.VISIBLE
            if (isLocalImport) {
                binding.tvCurrentFile.visibility = View.VISIBLE
            } else {
                binding.tvCurrentFile.visibility = View.GONE
            }
        } else {
            hideProgress()
        }
    }

    private fun refreshModelStatus() {
        val modelsDir = ModelConfig.activeModelDir(requireContext())
        val profile = ModelConfig.activeProfile(requireContext())
        val installed = ModelConfig.installedModels(requireContext())
        val df = DecimalFormat("#,##0")

        var totalSize = 0L
        var allPresent = true
        val sb = StringBuilder()

        sb.appendLine("model: ${profile.displayName}")
        if (installed.size > 1) sb.appendLine("(${installed.size} installed — pick in Settings)")
        sb.appendLine()

        for (model in profile.modelFiles) {
            val file = File(modelsDir, model.filename)
            val sizeMb = file.length() / (1024 * 1024)
            totalSize += file.length()

            val shortName = model.filename.removeSuffix("_android.onnx").removeSuffix(".onnx")
            if (file.exists()) {
                val check = if (sizeMb >= model.expectedSizeMb * 0.8) "OK" else "WARN"
                sb.appendLine("$check  $shortName: ${df.format(sizeMb)} MB")
            } else {
                allPresent = false
                sb.appendLine("--  $shortName: MISSING")
            }
        }

        for (filename in profile.tokenizerFiles) {
            val file = File(modelsDir, filename)
            if (file.exists()) {
                sb.appendLine("OK  $filename")
            } else {
                sb.appendLine("--  $filename (optional)")
            }
        }

        sb.appendLine()
        sb.appendLine(
            if (tokenizerAvailable()) "OK  tokenizer: BPE (vocab+merges)"
            else "!!  tokenizer: byte fallback — add vocab.json + merges.txt"
        )

        binding.tvModelStatus.text = sb.toString()
        binding.tvTotalSize.text = "${df.format(totalSize / (1024 * 1024))} / ${df.format(profile.modelFiles.sumOf { it.expectedSizeMb })} MB"

        binding.btnDownload.isEnabled = !allPresent
        binding.btnDeleteModels.isEnabled = totalSize > 0

        setupModelPicker()
        viewModel.refreshModelStatus()
    }

    /** True if real BPE tokenizer files are resolvable (models dir or assets). */
    private fun tokenizerAvailable(): Boolean {
        val dir = ModelConfig.activeModelDir(requireContext())
        if (File(dir, "vocab.json").exists() && File(dir, "merges.txt").exists()) return true
        return try {
            val assets = requireContext().assets.list("tokenizer") ?: emptyArray()
            assets.contains("vocab.json") && assets.contains("merges.txt")
        } catch (_: Exception) {
            false
        }
    }

    private fun showDownloadDialog() {
        val prefs = requireContext().getSharedPreferences("qwen3_tts_prefs", Context.MODE_PRIVATE)
        val savedUrl = prefs.getString("model_url", "") ?: ""

        val input = android.widget.EditText(requireContext()).apply {
            setText(savedUrl)
            hint = "https://huggingface.co/.../models.zip"
            inputType = android.text.InputType.TYPE_TEXT_VARIATION_URI
            setPadding(48, 32, 48, 16)
            setTextColor(resources.getColor(R.color.black, null))
            setHintTextColor(resources.getColor(R.color.slate_400, null))
        }

        AlertDialog.Builder(requireContext())
            .setTitle(R.string.dialog_download_title)
            .setMessage(R.string.dialog_download_message)
            .setView(input)
            .setPositiveButton(R.string.dialog_yes) { _, _ ->
                val url = input.text.toString().trim()
                if (url.isEmpty()) {
                    Toast.makeText(requireContext(), R.string.error_empty_url, Toast.LENGTH_SHORT).show()
                    return@setPositiveButton
                }
                prefs.edit().putString("model_url", url).apply()
                startDownload(url)
            }
            .setNegativeButton(R.string.dialog_no, null)
            .show()
    }

    private fun startDownload(url: String) {
        if (!url.startsWith("https://")) {
            Toast.makeText(requireContext(), R.string.error_https_only, Toast.LENGTH_LONG).show()
            return
        }

        isLocalImport = false
        showProgress(getString(R.string.status_downloading))
        binding.progressBar.isIndeterminate = false
        binding.progressBar.progress = 0
        binding.tvCurrentFile.visibility = View.GONE
        binding.btnStopDownload.visibility = View.VISIBLE

        val intent = Intent(requireContext(), ModelDownloadService::class.java).apply {
            putExtra(ModelDownloadService.EXTRA_URL, url)
            putExtra(ModelDownloadService.EXTRA_FORCE, true)
        }
        requireContext().startService(intent)
    }

    private fun startLocalImport(uri: Uri) {
        isLocalImport = true
        showProgress(getString(R.string.dialog_extracting))
        binding.progressBar.isIndeterminate = false
        binding.progressBar.progress = 0
        binding.tvDownloadStatus.text = getString(R.string.dialog_extracting)
        binding.tvCurrentFile.visibility = View.VISIBLE
        binding.tvCurrentFile.text = uri.lastPathSegment ?: uri.toString()
        binding.btnStopDownload.visibility = View.VISIBLE

        val intent = Intent(requireContext(), ModelDownloadService::class.java).apply {
            data = uri
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            putExtra(ModelDownloadService.EXTRA_ZIP_URI, uri.toString())
        }
        requireContext().startService(intent)
    }

    private fun setupDownloadCallbacks() {
        downloadService?.onProgress = { percent, downloaded, total, _ ->
            activity?.runOnUiThread {
                if (_binding == null) return@runOnUiThread
                binding.progressBar.progress = percent
                binding.tvDownloadStatus.text = if (isLocalImport) {
                    getString(R.string.extracting_progress, downloaded.toInt(), total.toInt())
                } else {
                    getString(R.string.dialog_download_progress, percent, "${downloaded / 1024 / 1024}MB", "${total / 1024 / 1024}MB")
                }
            }
        }

        downloadService?.onComplete = { success, error ->
            activity?.runOnUiThread {
                val context = context ?: return@runOnUiThread
                if (_binding == null) return@runOnUiThread
                hideProgress()
                binding.tvCurrentFile.visibility = View.GONE
                ModelConfig.invalidateProfileCache()
                if (success) {
                    Toast.makeText(context, R.string.dialog_download_complete, Toast.LENGTH_LONG).show()
                } else {
                    Toast.makeText(context, getString(R.string.dialog_download_error, error ?: "Unknown"), Toast.LENGTH_LONG).show()
                }
                refreshModelStatus()
            }
        }
    }

    private fun stopDownload() {
        downloadService?.cancelDownload()
        hideProgress()
    }

    private fun showDeleteDialog() {
        AlertDialog.Builder(requireContext())
            .setTitle(R.string.dialog_delete_title)
            .setMessage(R.string.dialog_delete_message)
            .setPositiveButton(R.string.btn_delete_models) { _, _ -> deleteModels() }
            .setNegativeButton(R.string.dialog_cancel, null)
            .show()
    }

    private fun deleteModels() {
        val appContext = requireContext().applicationContext
        lifecycleScope.launch(Dispatchers.IO) {
            val modelsDir = ModelConfig.getModelsDir(appContext)
            modelsDir.listFiles()?.forEach { it.deleteRecursively() }
            ModelConfig.invalidateProfileCache()
            com.qwen3.tts.engine.tokenizer.Qwen3Tokenizer.reset()
            withContext(Dispatchers.Main) {
                if (_binding != null) {
                    Toast.makeText(appContext, R.string.toast_models_deleted, Toast.LENGTH_SHORT).show()
                    refreshModelStatus()
                }
            }
        }
    }

    private fun showProgress(title: String) {
        binding.cardProgress.visibility = View.VISIBLE
        binding.tvProgressTitle.text = title
    }

    private fun hideProgress() {
        binding.cardProgress.visibility = View.GONE
        binding.btnStopDownload.visibility = View.GONE
        binding.progressBar.isIndeterminate = false
    }

    // Audio test helpers
    private fun playSineWave() {
        stopAudio()
        val samples = SAMPLE_RATE * 2
        val audioData = FloatArray(samples) { i ->
            (sin(2 * PI * 440.0 * i / SAMPLE_RATE) * 0.5).toFloat()
        }
        playFloatAudio(audioData, "440 Hz")
    }

    private fun playFrequencySweep() {
        stopAudio()
        val samples = SAMPLE_RATE * 2
        val audioData = FloatArray(samples) { i ->
            val t = i.toFloat() / SAMPLE_RATE
            val freq = 200f + (1800f * t / 2)
            (sin(2 * PI * freq * i / SAMPLE_RATE) * 0.3).toFloat()
        }
        playFloatAudio(audioData, "200-2000 Hz")
    }

    private fun playFloatAudio(audioData: FloatArray, description: String) {
        try {
            val minBuffer = AudioTrack.getMinBufferSize(
                SAMPLE_RATE, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT
            )
            val track = AudioHelper.buildSpeechAudioTrack(
                sampleRate = SAMPLE_RATE,
                encoding = AudioFormat.ENCODING_PCM_FLOAT,
                bufferSizeInBytes = minBuffer.coerceAtLeast(audioData.size * 4)
            )
            track.write(audioData, 0, audioData.size, AudioTrack.WRITE_BLOCKING)
            track.play()
            audioTrack = track
            Toast.makeText(requireContext(), description, Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Toast.makeText(requireContext(), getString(R.string.audio_playback_failed, e.message ?: ""), Toast.LENGTH_SHORT).show()
        }
    }

    private fun stopAudio() {
        audioTrack?.let {
            try { it.stop(); it.release() } catch (_: Exception) {}
        }
        audioTrack = null
    }

    override fun onDestroyView() {
        super.onDestroyView()
        stopAudio()
        if (isBound) {
            requireContext().unbindService(serviceConnection)
            isBound = false
        }
        _binding = null
    }
}
