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

    private val pickAudio =
        registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
            if (uri != null) importAudio(uri)
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
        binding.btnUploadVoice.setOnClickListener {
            pickAudio.launch(arrayOf("audio/*"))
        }
    }

    /**
     * Decodes a user-picked audio file to a 16 kHz mono WAV off the UI thread and
     * reuses the same naming/cloning flow as a mic recording. Lets the user try a
     * cloned voice from an existing clip instead of recording one live.
     */
    private fun importAudio(uri: android.net.Uri) {
        val ctx = requireContext().applicationContext
        Toast.makeText(ctx, R.string.upload_decoding, Toast.LENGTH_SHORT).show()
        val tmp = File(ctx.cacheDir, "voice_import_${System.currentTimeMillis()}.wav")
        Thread {
            val seconds = com.qwen3.tts.util.AudioImport.decodeToWav16kMono(ctx, uri, tmp)
            activity?.runOnUiThread {
                if (!isAdded) return@runOnUiThread
                if (seconds < 1.5f) {
                    tmp.delete()
                    Toast.makeText(requireContext(), R.string.upload_failed, Toast.LENGTH_SHORT).show()
                } else {
                    pendingClip = tmp
                    promptVoiceName()
                }
            }
        }.start()
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
            val ctx = requireContext()
            val hasModel = ModelConfig.installedModels(ctx).isNotEmpty()
            when {
                ready -> {
                    // Show the active (selected) model by name, nothing else.
                    binding.tvEngineStatus.text =
                        getString(R.string.status_model_ready, ModelConfig.activeProfile(ctx).displayName)
                    tintStatusDot(R.color.green)
                    binding.btnSynthesize.isEnabled = true
                }
                !hasModel -> {
                    // No model installed: the missing-file list is just noise here.
                    binding.tvEngineStatus.text = getString(R.string.status_no_model)
                    tintStatusDot(R.color.amber)
                    binding.btnSynthesize.isEnabled = false
                }
                else -> {
                    // A model is selected but still being installed/loaded.
                    val missing = ModelConfig.missingSynthesisModels(ctx).take(3).joinToString()
                    binding.tvEngineStatus.text = getString(R.string.status_models_missing_details, missing)
                    tintStatusDot(R.color.amber)
                    binding.btnSynthesize.isEnabled = false
                }
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

    // chipId -> locale string
    private val langChips = mutableMapOf<Int, String>()
    // chipId -> VoiceSpec, rebuilt from the active model profile/manifest.
    private val voiceChips = mutableMapOf<Int, ModelConfig.VoiceSpec>()
    private var lastVoiceIds = emptySet<String>()

    private fun setupVoiceChips() {
        val profile = ModelConfig.activeProfile(requireContext())
        binding.chipGroupLang.removeAllViews()
        langChips.clear()

        val customVoices = CustomVoiceStore.list(requireContext()).map { it.toVoiceSpec() }
        // No installed model => activeProfile() is the compile-time Qwen3 fallback;
        // don't show its voices as if they were usable.
        val profileVoices =
            if (ModelConfig.installedModels(requireContext()).isEmpty()) emptyList()
            else profile.voices
        val allVoices = profileVoices + customVoices
        lastVoiceIds = allVoices.map { it.id }.toSet()

        val distinctLocales = allVoices.map { it.locale }.distinct()
        if (distinctLocales.isEmpty()) {
            binding.chipGroupVoice.removeAllViews()
            voiceChips.clear()
            return
        }

        distinctLocales.forEachIndexed { index, loc ->
            val chip = layoutInflater.inflate(
                R.layout.item_voice_chip, binding.chipGroupLang, false
            ) as Chip
            chip.id = View.generateViewId()
            
            val parts = loc.split('-', '_')
            val locale = if (parts.size >= 2) Locale(parts[0], parts[1]) else Locale(parts[0])
            val displayName = locale.getDisplayLanguage(Locale.getDefault()).replaceFirstChar { it.uppercase() }
            
            chip.text = displayName
            chip.isChecked = index == 0
            binding.chipGroupLang.addView(chip)
            langChips[chip.id] = loc
        }

        val initialLoc = distinctLocales.firstOrNull() ?: "ru-RU"
        updateVoiceChipsForLocale(initialLoc, allVoices)

        val initialLangCode = initialLoc.split('-', '_').firstOrNull()?.lowercase() ?: initialLoc
        binding.etInput.setText(sampleTextFor(initialLangCode))

        binding.chipGroupLang.setOnCheckedStateChangeListener { _, checkedIds ->
            val loc = checkedIds.firstOrNull()?.let { langChips[it] } ?: return@setOnCheckedStateChangeListener
            updateVoiceChipsForLocale(loc, allVoices)
            val langCode = loc.split('-', '_').firstOrNull()?.lowercase() ?: loc
            binding.etInput.setText(sampleTextFor(langCode))
        }
    }

    private fun updateVoiceChipsForLocale(locale: String, allVoices: List<ModelConfig.VoiceSpec>) {
        binding.chipGroupVoice.setOnCheckedStateChangeListener(null)
        binding.chipGroupVoice.removeAllViews()
        voiceChips.clear()

        val stored = ModelConfig.defaultVoiceId(requireContext())
        val filteredVoices = allVoices.filter { it.locale.equals(locale, ignoreCase = true) }
        val checkedIndex = filteredVoices.indexOfFirst { it.id == stored }.coerceAtLeast(0)
        filteredVoices.forEachIndexed { index, spec ->
            val chip = layoutInflater.inflate(
                R.layout.item_voice_chip, binding.chipGroupVoice, false
            ) as Chip
            chip.id = View.generateViewId()
            chip.text = spec.displayName
            chip.isChecked = index == checkedIndex
            binding.chipGroupVoice.addView(chip)
            voiceChips[chip.id] = spec
        }

        // Persist the selection: reader apps don't set a voice on their requests,
        // so the service uses this stored id as the default (see QwenTTSEngine).
        binding.chipGroupVoice.setOnCheckedStateChangeListener { _, checkedIds ->
            val spec = checkedIds.firstOrNull()?.let { voiceChips[it] } ?: return@setOnCheckedStateChangeListener
            ModelConfig.setDefaultVoiceId(requireContext(), spec.id)
        }
    }

    private fun sampleTextFor(lang: String): String = when (lang.lowercase()) {
        "ru" -> getString(R.string.sample_text_ru)
        "en" -> getString(R.string.sample_text_en)
        "zh" -> getString(R.string.sample_text_zh)
        else -> getString(R.string.sample_text_default)
    }

    private fun selectedVoice(): ModelConfig.VoiceSpec? =
        voiceChips[binding.chipGroupVoice.checkedChipId] ?: voiceChips.values.firstOrNull()

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
        refreshVoicesAndStatus()
    }

    // MainActivity switches tabs with show/hide, so onResume does NOT fire when
    // the user returns from the Models tab after installing a model — voices
    // looked like they only appeared after an app restart.
    override fun onHiddenChanged(hidden: Boolean) {
        super.onHiddenChanged(hidden)
        if (!hidden && _binding != null) refreshVoicesAndStatus()
    }

    private fun refreshVoicesAndStatus() {
        viewModel.refreshModelStatus()
        setupModelPicker()
        // Rebuild chips only if the active model's voice set (or custom voices) changed.
        // Mirror setupVoiceChips(): profile voices only count when a model is installed.
        val profileIds =
            if (ModelConfig.installedModels(requireContext()).isEmpty()) emptyList()
            else ModelConfig.activeProfile(requireContext()).voices.map { it.id }
        val latest = (profileIds + CustomVoiceStore.list(requireContext()).map { it.id }).toSet()
        if (lastVoiceIds != latest) setupVoiceChips()
        updateCloneButtons()
    }

    /**
     * Voice cloning (mic recording / audio upload) only works when the active
     * model actually has a speaker encoder (Qwen family). Hide the buttons for
     * models that can't clone (sherpa Kokoro/Piper have fixed voice banks).
     */
    private fun updateCloneButtons() {
        val ctx = requireContext()
        val supported = ModelConfig.installedModels(ctx).isNotEmpty() &&
            ModelConfig.activeProfile(ctx).modelFiles
                .any { it.role == ModelConfig.Role.SPEAKER_ENCODER }
        binding.btnRecordVoice.visibility = if (supported) View.VISIBLE else View.GONE
        binding.btnUploadVoice.visibility = if (supported) View.VISIBLE else View.GONE
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
