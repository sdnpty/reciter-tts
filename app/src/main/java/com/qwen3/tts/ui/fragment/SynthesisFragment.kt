package com.qwen3.tts.ui.fragment

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.provider.Settings
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.view.View
import android.widget.EditText
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import com.qwen3.tts.R
import com.google.android.material.chip.Chip
import com.qwen3.tts.databinding.FragmentSynthesisBinding
import com.qwen3.tts.ui.TTSViewModel
import com.qwen3.tts.util.CustomVoiceStore
import com.qwen3.tts.util.ModelConfig
import com.qwen3.tts.util.VoiceCloner
import com.qwen3.tts.util.VoiceRecorder
import java.io.File
import java.util.Locale

class SynthesisFragment : Fragment(R.layout.fragment_synthesis) {
    private var _binding: FragmentSynthesisBinding? = null
    private val binding get() = _binding!!
    private var tts: TextToSpeech? = null
    private var exportTts: TextToSpeech? = null
    private var isSynthesizing = false

    private val recorder = VoiceRecorder()
    private var pendingClip: File? = null

    private val micPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) toggleRecording()
            else Toast.makeText(requireContext(), R.string.record_permission_needed, Toast.LENGTH_SHORT).show()
        }

    /**
     * Creates a [TextToSpeech] bound to THIS app's own engine (QwenTTSEngine)
     * instead of the system default (which is usually Google TTS). Without the
     * explicit engine package the preview always plays Google's voice and never
     * touches the imported Qwen model.
     */
    private fun newAppTts(onInit: (Int) -> Unit): TextToSpeech =
        TextToSpeech(requireContext(), onInit, requireContext().packageName)

    private val viewModel by lazy {
        ViewModelProvider(requireActivity())[TTSViewModel::class.java]
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        _binding = FragmentSynthesisBinding.bind(view)

        setupModelPicker()
        setupVoiceChips()
        observeModelStatus()

        binding.btnSynthesize.setOnClickListener {
            if (isSynthesizing) stopSynthesis() else startSynthesis()
        }

        binding.btnSetDefault.setOnClickListener { openTtsSettings() }
        binding.btnSaveWav.setOnClickListener { saveToWav() }
        binding.btnRecordVoice.setOnClickListener { onRecordClicked() }
    }

    // ── Mic recording → custom voice ──────────────────────────────

    private fun onRecordClicked() {
        if (recorder.isRecording) { toggleRecording(); return }
        val granted = ContextCompat.checkSelfPermission(
            requireContext(), Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED
        if (granted) toggleRecording() else micPermission.launch(Manifest.permission.RECORD_AUDIO)
    }

    private fun toggleRecording() {
        if (recorder.isRecording) {
            val seconds = recorder.stop()
            binding.btnRecordVoice.setText(R.string.btn_record_voice)
            if (seconds < 1.5f) {
                pendingClip?.delete()
                Toast.makeText(requireContext(), R.string.record_too_short, Toast.LENGTH_SHORT).show()
            } else {
                promptVoiceName()
            }
        } else {
            val tmp = File(requireContext().cacheDir, "voice_rec_${System.currentTimeMillis()}.wav")
            try {
                recorder.start(tmp)
                pendingClip = tmp
                binding.btnRecordVoice.setText(R.string.btn_stop_record)
                Toast.makeText(requireContext(), R.string.record_in_progress, Toast.LENGTH_SHORT).show()
            } catch (e: Exception) {
                Toast.makeText(requireContext(), R.string.record_failed, Toast.LENGTH_SHORT).show()
            }
        }
    }

    /**
     * Computes the speaker x-vector for a freshly recorded clip off the UI thread,
     * using the active model's `speaker_encoder.onnx`. When the encoder is present
     * the cloned voice becomes selectable for synthesis; otherwise the clip is
     * kept and will be cloned once a model with an encoder is active.
     */
    private fun cloneInBackground(voiceId: String, clip: File) {
        val ctx = requireContext().applicationContext
        Thread {
            val cloner = VoiceCloner.forModel(ModelConfig.activeModelDir(ctx)) ?: return@Thread
            val vec = cloner.clone(clip)
            cloner.release()
            if (vec != null && vec.isNotEmpty()) {
                CustomVoiceStore.saveXVector(ctx, voiceId, vec)
                activity?.runOnUiThread {
                    if (isAdded) Toast.makeText(ctx, R.string.voice_cloned, Toast.LENGTH_SHORT).show()
                }
            }
        }.start()
    }

    private fun promptVoiceName() {
        val clip = pendingClip ?: return
        val input = EditText(requireContext()).apply {
            hint = getString(R.string.dialog_voice_name)
            setText(getString(R.string.label_my_voices) + " " + (CustomVoiceStore.list(requireContext()).size + 1))
        }
        AlertDialog.Builder(requireContext())
            .setTitle(R.string.dialog_voice_name)
            .setView(input)
            .setPositiveButton(R.string.dialog_save) { _, _ ->
                val name = input.text?.toString()?.trim().orEmpty()
                val voice = CustomVoiceStore.add(requireContext(), name, getSelectedLocale().toLanguageTag())
                clip.copyTo(File(voice.clipPath), overwrite = true)
                clip.delete()
                pendingClip = null
                setupVoiceChips()
                Toast.makeText(
                    requireContext(),
                    getString(R.string.record_saved, voice.displayName), Toast.LENGTH_SHORT
                ).show()
                cloneInBackground(voice.id, File(voice.clipPath))
            }
            .setNegativeButton(R.string.dialog_cancel) { _, _ ->
                clip.delete(); pendingClip = null
            }
            .show()
    }

    private fun saveToWav() {
        val text = binding.etInput.text?.toString()?.trim()
        if (text.isNullOrEmpty()) {
            Toast.makeText(requireContext(), R.string.hint_text, Toast.LENGTH_SHORT).show()
            return
        }
        val dir = File(requireContext().cacheDir, "exports").apply { mkdirs() }
        val outFile = File(dir, "reciter_${System.currentTimeMillis()}.wav")
        Toast.makeText(requireContext(), R.string.wav_saving, Toast.LENGTH_SHORT).show()

        exportTts?.shutdown()
        exportTts = newAppTts { status ->
            if (status != TextToSpeech.SUCCESS) {
                activity?.runOnUiThread { toastWavFailed() }
                return@newAppTts
            }
            exportTts?.language = getSelectedLocale()
            selectedVoice()?.id?.let { voiceId ->
                exportTts?.voices?.firstOrNull { it.name == voiceId }?.let { exportTts?.voice = it }
            }
            exportTts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                override fun onStart(utteranceId: String?) {}
                override fun onDone(utteranceId: String?) {
                    activity?.runOnUiThread { shareWav(outFile) }
                }
                override fun onError(utteranceId: String?) {
                    activity?.runOnUiThread { toastWavFailed() }
                }
            })
            val uid = "export_${System.currentTimeMillis()}"
            val result = exportTts?.synthesizeToFile(text, android.os.Bundle(), outFile, uid)
            if (result != TextToSpeech.SUCCESS) {
                activity?.runOnUiThread { toastWavFailed() }
            }
        }
    }

    private fun shareWav(file: File) {
        val ctx = context ?: return
        if (!file.exists() || file.length() == 0L) {
            toastWavFailed()
            return
        }
        val uri = FileProvider.getUriForFile(ctx, "${ctx.packageName}.fileprovider", file)
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "audio/x-wav"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        startActivity(Intent.createChooser(intent, getString(R.string.wav_share_title)))
    }

    private fun toastWavFailed() {
        if (isAdded) Toast.makeText(requireContext(), R.string.wav_failed, Toast.LENGTH_SHORT).show()
    }

    private fun observeModelStatus() {
        viewModel.modelsReady.observe(viewLifecycleOwner) { ready ->
            if (ready) {
                binding.tvEngineStatus.text = getString(R.string.status_models_present)
                tintStatusDot(R.color.green)
                binding.btnSynthesize.isEnabled = true
            } else {
                val missing = ModelConfig.missingSynthesisModels(requireContext())
                    .take(3).joinToString()
                binding.tvEngineStatus.text = getString(R.string.status_models_missing_details, missing)
                tintStatusDot(R.color.amber)
                binding.btnSynthesize.isEnabled = false
            }
        }
    }

    private fun tintStatusDot(colorRes: Int) {
        binding.statusIndicator.backgroundTintList = android.content.res.ColorStateList.valueOf(
            ContextCompat.getColor(requireContext(), colorRes)
        )
    }

    // chipId -> installed model id, for the model selector.
    private val modelChips = mutableMapOf<Int, String>()

    /**
     * Builds the model-selector chip row from every installed model so the user
     * can switch the active model right on the Synthesis tab (not only on the
     * Models tab). Hidden when fewer than two models are installed.
     */
    private fun setupModelPicker() {
        val models = ModelConfig.installedModels(requireContext())
        val activeId = ModelConfig.activeModel(requireContext())?.id
        binding.chipGroupModel.removeAllViews()
        modelChips.clear()

        if (models.size < 2) {
            binding.cardModelPicker.visibility = View.GONE
            return
        }
        binding.cardModelPicker.visibility = View.VISIBLE

        models.forEach { m ->
            val chip = layoutInflater.inflate(
                R.layout.item_voice_chip, binding.chipGroupModel, false
            ) as Chip
            chip.id = View.generateViewId()
            val ready = if (m.synthesisReady()) "✓ " else "⬇ "
            chip.text = ready + m.profile.displayName
            chip.isChecked = m.id == activeId
            binding.chipGroupModel.addView(chip)
            modelChips[chip.id] = m.id
        }

        binding.chipGroupModel.setOnCheckedStateChangeListener { _, checkedIds ->
            val id = checkedIds.firstOrNull()?.let { modelChips[it] } ?: return@setOnCheckedStateChangeListener
            if (id == ModelConfig.activeModel(requireContext())?.id) return@setOnCheckedStateChangeListener
            ModelConfig.setActiveModel(requireContext(), id)
            setupVoiceChips()
            viewModel.refreshModelStatus()
        }
    }

    // chipId -> VoiceSpec, rebuilt from the active model profile/manifest.
    private val voiceChips = mutableMapOf<Int, ModelConfig.VoiceSpec>()

    private fun setupVoiceChips() {
        val profile = ModelConfig.activeProfile(requireContext())
        binding.chipGroupLang.removeAllViews()
        voiceChips.clear()

        val customVoices = CustomVoiceStore.list(requireContext()).map { it.toVoiceSpec() }
        val allVoices = profile.voices + customVoices

        allVoices.forEachIndexed { index, spec ->
            val chip = layoutInflater.inflate(
                R.layout.item_voice_chip, binding.chipGroupLang, false
            ) as Chip
            chip.id = View.generateViewId()
            chip.text = spec.displayName
            chip.isChecked = index == 0
            binding.chipGroupLang.addView(chip)
            voiceChips[chip.id] = spec
        }

        profile.voices.firstOrNull()?.let { binding.etInput.setText(sampleTextFor(it.languageTag())) }

        binding.chipGroupLang.setOnCheckedStateChangeListener { _, checkedIds ->
            val spec = checkedIds.firstOrNull()?.let { voiceChips[it] } ?: return@setOnCheckedStateChangeListener
            binding.etInput.setText(sampleTextFor(spec.languageTag()))
        }
    }

    private fun sampleTextFor(lang: String): String = when (lang.lowercase()) {
        "ru" -> getString(R.string.sample_text_ru)
        "en" -> getString(R.string.sample_text_en)
        "zh" -> getString(R.string.sample_text_zh)
        else -> getString(R.string.sample_text_default)
    }

    private fun selectedVoice(): ModelConfig.VoiceSpec? =
        voiceChips[binding.chipGroupLang.checkedChipId] ?: voiceChips.values.firstOrNull()

    private fun getSelectedLocale(): Locale = selectedVoice()?.toLocale() ?: Locale("ru", "RU")

    private fun startSynthesis() {
        val text = binding.etInput.text?.toString()?.trim()
        if (text.isNullOrEmpty()) {
            Toast.makeText(requireContext(), R.string.hint_text, Toast.LENGTH_SHORT).show()
            return
        }

        isSynthesizing = true
        binding.btnSynthesize.text = getString(R.string.btn_stop_audio)
        binding.progressSynth.visibility = View.VISIBLE

        tts?.shutdown()
        tts = null
        tts = newAppTts { status ->
            if (status == TextToSpeech.SUCCESS) {
                tts?.language = getSelectedLocale()
                // Prefer the exact model voice matching the selected chip.
                selectedVoice()?.id?.let { voiceId ->
                    tts?.voices?.firstOrNull { it.name == voiceId }?.let { tts?.voice = it }
                }
                tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                    override fun onStart(utteranceId: String?) {}
                    override fun onDone(utteranceId: String?) {
                        activity?.runOnUiThread { resetSynthesisState() }
                    }
                    override fun onError(utteranceId: String?) {
                        activity?.runOnUiThread {
                            resetSynthesisState()
                            Toast.makeText(requireContext(), R.string.status_error, Toast.LENGTH_SHORT).show()
                        }
                    }
                })
                tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "synth_${System.currentTimeMillis()}")
            } else {
                activity?.runOnUiThread {
                    resetSynthesisState()
                    Toast.makeText(requireContext(), R.string.status_error, Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    private fun stopSynthesis() {
        tts?.stop()
        resetSynthesisState()
    }

    private fun resetSynthesisState() {
        isSynthesizing = false
        binding.btnSynthesize.text = getString(R.string.btn_synthesize)
        binding.progressSynth.visibility = View.GONE
    }

    private fun openTtsSettings() {
        val intent = Intent("com.android.settings.TTS_SETTINGS")
        try {
            startActivity(intent)
        } catch (_: Exception) {
            try {
                startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS))
            } catch (_: Exception) {
                Toast.makeText(requireContext(), R.string.toast_default_tts_set, Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onResume() {
        super.onResume()
        viewModel.refreshModelStatus()
        setupModelPicker()
        // Rebuild chips only if the active model's voice set (or custom voices) changed.
        val current = voiceChips.values.map { it.id }.toSet()
        val latest = (ModelConfig.activeProfile(requireContext()).voices.map { it.id } +
            CustomVoiceStore.list(requireContext()).map { it.id }).toSet()
        if (current != latest) setupVoiceChips()
    }

    override fun onDestroyView() {
        super.onDestroyView()
        recorder.cancel()
        tts?.shutdown()
        tts = null
        exportTts?.shutdown()
        exportTts = null
        _binding = null
    }
}
