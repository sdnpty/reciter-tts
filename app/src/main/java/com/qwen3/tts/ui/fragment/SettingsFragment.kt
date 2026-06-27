package com.qwen3.tts.ui.fragment

import android.content.Context
import android.content.ClipData
import android.content.ClipboardManager
import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.lifecycleScope
import com.google.android.material.chip.Chip
import com.qwen3.tts.BuildConfig
import com.qwen3.tts.R
import com.qwen3.tts.databinding.FragmentSettingsBinding
import com.qwen3.tts.engine.tokenizer.Qwen3Tokenizer
import com.qwen3.tts.ui.TTSViewModel
import com.qwen3.tts.util.LogShareHelper
import com.qwen3.tts.util.ModelConfig
import com.qwen3.tts.util.TTSLogger
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class SettingsFragment : Fragment(R.layout.fragment_settings) {
    private var _binding: FragmentSettingsBinding? = null
    private val binding get() = _binding!!

    private val viewModel by lazy {
        ViewModelProvider(requireActivity())[TTSViewModel::class.java]
    }
    private lateinit var logger: TTSLogger

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        _binding = FragmentSettingsBinding.bind(view)
        logger = TTSLogger.getInstance(requireContext())

        val prefs = requireContext().getSharedPreferences("qwen3_tts_prefs", Context.MODE_PRIVATE)

        // Load saved values
        binding.sliderSpeed.value = prefs.getInt("speed", 100).toFloat()
        binding.switchLogging.isChecked = prefs.getBoolean("detailed_logging", true)

        val savedDevice = prefs.getString("device", "cpu") ?: "cpu"
        if (savedDevice == "nnapi") {
            binding.chipNnapi.isChecked = true
        } else {
            binding.chipCpu.isChecked = true
        }

        // Listeners
        binding.sliderSpeed.addOnChangeListener { _, value, _ ->
            prefs.edit().putInt("speed", value.toInt()).apply()
            binding.tvSpeedValue.text = "${value.toInt()}%"
            logger.i(TAG, "Speed: ${value.toInt()}%")
        }
        binding.tvSpeedValue.text = "${binding.sliderSpeed.value.toInt()}%"

        binding.chipGroupDevice.setOnCheckedStateChangeListener { _, checkedIds ->
            val device = if (checkedIds.contains(R.id.chipNnapi)) "nnapi" else "cpu"
            prefs.edit().putString("device", device).apply()
            logger.i(TAG, "Device: $device")
        }

        binding.switchLogging.setOnCheckedChangeListener { _, isChecked ->
            prefs.edit().putBoolean("detailed_logging", isChecked).apply()
            logger.i(TAG, "Detailed logging: $isChecked")
        }

        binding.btnShareLogs.setOnClickListener {
            LogShareHelper.shareOrToast(requireContext(), logger)
        }

        binding.btnClearLogs.setOnClickListener {
            logger.clearLogs()
            binding.tvLogContent.text = getString(R.string.log_empty)
            Toast.makeText(requireContext(), R.string.toast_logs_cleared, Toast.LENGTH_SHORT).show()
        }

        binding.btnCopyLogs.setOnClickListener {
            val clipboard = requireContext().getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
            val clip = ClipData.newPlainText("Logs", binding.tvLogContent.text)
            clipboard.setPrimaryClip(clip)
            Toast.makeText(requireContext(), R.string.toast_logs_copied, Toast.LENGTH_SHORT).show()
        }

        binding.tvAppVersion.text = getString(R.string.app_version_format, BuildConfig.VERSION_NAME, BuildConfig.VERSION_CODE)

        setupModelPicker()
        startLogWatcher()
    }

    // chipId -> model slot id
    private val modelChips = mutableMapOf<Int, String>()

    private fun setupModelPicker() {
        val ctx = context ?: return
        val installed = ModelConfig.installedModels(ctx)
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
            chip.text = if (langs.isBlank()) model.profile.displayName else "${model.profile.displayName} · $langs"
            chip.isChecked = model.id == activeId
            binding.chipGroupModels.addView(chip)
            modelChips[chip.id] = model.id
        }

        binding.chipGroupModels.setOnCheckedStateChangeListener { _, checkedIds ->
            val id = checkedIds.firstOrNull()?.let { modelChips[it] } ?: return@setOnCheckedStateChangeListener
            if (id != ModelConfig.activeModel(requireContext())?.id) {
                ModelConfig.setActiveModel(requireContext(), id)
                Qwen3Tokenizer.reset()
                logger.i(TAG, "Active model switched to '$id'")
                Toast.makeText(requireContext(), R.string.model_switch_hint, Toast.LENGTH_LONG).show()
            }
        }
    }

    override fun onResume() {
        super.onResume()
        setupModelPicker()
    }

    private fun startLogWatcher() {
        viewLifecycleOwner.lifecycleScope.launch {
            while (isActive) {
                val logs = withContext(Dispatchers.IO) {
                    logger.getLatestLogs(100).joinToString("\n")
                }
                val b = _binding
                if (b != null) {
                    b.tvLogContent.text = logs.ifEmpty { getString(R.string.log_empty) }
                    b.scrollLogs.post {
                        // Runs later on the handler; the view may be gone by then.
                        _binding?.scrollLogs?.fullScroll(View.FOCUS_DOWN)
                    }
                }
                delay(3000)
            }
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }

    companion object {
        private const val TAG = "Settings"
    }
}
