package com.qwen3.tts.ui

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import com.qwen3.tts.R
import com.qwen3.tts.databinding.ActivityMainBinding
import com.qwen3.tts.ui.fragment.ModelsFragment
import com.qwen3.tts.ui.fragment.SettingsFragment
import com.qwen3.tts.ui.fragment.SynthesisFragment

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    val viewModel by lazy { ViewModelProvider(this)[TTSViewModel::class.java] }

    private val synthesisFragment by lazy { SynthesisFragment() }
    private val modelsFragment by lazy { ModelsFragment() }
    private val settingsFragment by lazy { SettingsFragment() }

    private var activeFragment: Fragment? = null

    private val requestNotificationPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { /* No-op: download still works, just without a progress notification. */ }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        maybeRequestNotificationPermission()

        if (savedInstanceState == null) {
            showFragment(synthesisFragment)
        }

        binding.bottomNav.setOnItemSelectedListener { item ->
            when (item.itemId) {
                R.id.nav_synthesis -> {
                    showFragment(synthesisFragment)
                    true
                }
                R.id.nav_models -> {
                    showFragment(modelsFragment)
                    true
                }
                R.id.nav_settings -> {
                    showFragment(settingsFragment)
                    true
                }
                else -> false
            }
        }
    }

    private fun maybeRequestNotificationPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            val granted = ContextCompat.checkSelfPermission(
                this, Manifest.permission.POST_NOTIFICATIONS
            ) == PackageManager.PERMISSION_GRANTED
            if (!granted) {
                requestNotificationPermission.launch(Manifest.permission.POST_NOTIFICATIONS)
            }
        }
    }

    private fun showFragment(fragment: Fragment) {
        if (fragment === activeFragment) return

        val transaction = supportFragmentManager.beginTransaction()

        activeFragment?.let { transaction.hide(it) }

        if (!fragment.isAdded) {
            transaction.add(R.id.fragment_container, fragment)
        } else {
            transaction.show(fragment)
        }

        transaction.commit()
        activeFragment = fragment
    }

    override fun onResume() {
        super.onResume()
        viewModel.refreshModelStatus()
    }
}
