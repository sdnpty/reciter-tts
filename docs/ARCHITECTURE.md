# Architecture

## Overview

Reciter TTS is an Android TTS engine that runs Qwen3-TTS inference locally
via ONNX Runtime. It registers as a system `TextToSpeechService`, so any app
that uses Android's TTS API will use it automatically once set as default.

```
Android System TTS API
         |
  QwenTTSEngine (TextToSpeechService)
         |
  QwenInferenceEngine
    |         |           |            |
  Talker  CodePredictor  Code2Wav  SpeakerEncoder
  (ONNX)    (ONNX)       (ONNX)     (ONNX)
         |
     PCM-16 audio @ 24 kHz
         |
  SynthesisCallback -> AudioTrack
```

## TTS Pipeline

1. **Tokenize** -- `Qwen3Tokenizer` (BPE, vocab.json + merges.txt) converts
   text to token IDs. Falls back to a byte-level Cyrillic tokenizer if vocab
   files are missing.

2. **Talker** (autoregressive) -- Feeds token IDs through the transformer and
   greedily decodes audio token IDs until EOS. Audio tokens start at ID 151936
   (`TokenizerConstants.AUDIO_TOKEN_START`). Produces coarse codec codes
   (quantizer 0).

3. **CodePredictor** -- Takes coarse codes and predicts fine codes for
   quantizers 1..15 via argmax over logits. If unavailable, coarse codes are
   duplicated across all quantizers (degraded quality, still functional).

4. **Code2Wav** (vocoder) -- Converts the full `[1, 16, seq]` codec tensor to
   a raw audio waveform `[1, 1, samples]` at 24 kHz.

5. **PCM conversion** -- `AudioHelper.floatToPcm16()` converts float samples
   to 16-bit PCM bytes streamed via `SynthesisCallback`.

## Module Map

```
com.qwen3.tts/
  QwenTTSApplication         -- Global crash handler, logger init

  engine/
    QwenTTSEngine             -- TextToSpeechService implementation
    CheckVoiceData            -- Android TTS voice data check Activity
    GetSampleText             -- Android TTS sample text Activity
    InstallVoiceData          -- Android TTS voice data install Activity
    inference/
      QwenInferenceEngine     -- ONNX session management + inference
    tokenizer/
      Qwen3Tokenizer          -- BPE tokenizer (singleton)

  service/
    ModelDownloadService      -- Foreground service for downloading/importing models

  ui/
    MainActivity              -- Single Activity with BottomNavigationView
    TTSViewModel              -- Shared ViewModel (model status, logger, tokenizer)
    fragment/
      SynthesisFragment       -- Text input, language selection, TTS playback
      ModelsFragment          -- Model status, download/import/delete, audio tests
      SettingsFragment        -- Speaker ID, speed, device, logging, log viewer

  util/
    ModelConfig               -- Model profiles, file paths, readiness checks
    TokenizerConstants        -- Vocab size, special token IDs
    AudioHelper               -- PCM conversion, AudioTrack builder
    TTSLogger                 -- File-based logger with rotation
    LogShareHelper            -- Log file sharing via FileProvider
```

## Model Slots (multiple installed models)

Several models can be installed side by side. Each lives in its own slot
directory under the models root:

```
<external-files>/models/
  <slotId-A>/  talker_base_android.onnx … vocab.json merges.txt model.json
  <slotId-B>/  …
  talker_base_android.onnx …            # legacy flat layout = "default" slot
```

- Import/download extracts into `models/_incoming/`, then `finalizeSlot()`
  moves it to `models/<id>/` (id taken from `model.json`, else generated) and
  marks it active.
- `ModelConfig.installedModels()` scans slot dirs (+ the legacy flat layout).
- The active slot id is stored in SharedPreferences (`active_model_id`) and
  chosen in Settings → Active model. `activeModelDir()` / `activeProfile()`
  resolve everything (engine paths, tokenizer, presence checks) against it.
- Switching models resets the tokenizer; the TTS engine picks up the new model
  on its next (re)load.

## Multi-Model Architecture

`ModelConfig` supports multiple model profiles through `ModelProfile`:

```kotlin
data class ModelProfile(
    val id: String,              // "qwen3-tts-12hz-0.6b"
    val displayName: String,     // shown in UI
    val family: String,          // "qwen3-tts" -- groups related models
    val sampleRateHz: Int,       // 24000
    val codecFrameRateHz: Int,   // 12
    val modelFiles: List<ModelFile>,
    val tokenizerFiles: List<String>
)
```

Each `ModelFile` has a `Role` (TALKER, CODE_PREDICTOR, VOCODER, SPEAKER_ENCODER)
and metadata (`expectedSizeMb`, `requiredForSynthesis`).

To add a new model:

1. Export ONNX files (see [MODEL_BUILD_PIPELINE.md](MODEL_BUILD_PIPELINE.md))
2. Add a `ModelProfile` to `ModelConfig.SUPPORTED_PROFILES`
3. The engine resolves files by `Role`, so different model sizes and filenames
   work without engine code changes

Currently `ACTIVE_PROFILE` is compile-time. To make it runtime-selectable:
- Store the selected profile ID in SharedPreferences
- Read it in `QwenTTSEngine.initEngine()` and `TTSViewModel`
- Restart the TTS service when the profile changes

## ONNX Runtime Configuration

- **CPU by default**: intra-op threads = `availableProcessors()` clamped to
  1..4, 1 inter-op thread, `ALL_OPT` graph optimization, memory-pattern
  pre-allocation disabled to keep the native footprint small and predictable.
- **NNAPI is opt-in** (Settings → Engine → NNAPI). It can accelerate the
  vocoder, but INT8-quantized models abort *natively* (SIGABRT) on many device
  drivers — that is the usual cause of the engine "instantly closing" with
  nothing in the logs. CPU is the stable default everywhere.
- **Off-main-thread load**: the TTS service loads sessions on a background
  thread; the first synthesis request waits on a latch instead of failing.
- **Crash breadcrumbs**: `TTSLogger.beginLoadStage()` writes a synchronous
  checkpoint before each native `createSession`. If the process dies natively,
  the next launch reads the leftover checkpoint and reports exactly where it
  crashed (`consumeStaleLoadCrash()`), so native crashes are no longer silent.
- **Runtime version**: 1.26.0 (Maven Central). Compatible with INT8 quantized
  models at opset 18.

## Security

- **HTTPS-only downloads**: `ModelDownloadService` rejects non-HTTPS URLs
- **Path traversal protection**: ZIP extraction validates canonical paths
- **Network security config**: `network_security_config.xml` restricts
  cleartext traffic
- **No secrets in code**: Download URLs are user-provided at runtime

## Build & CI

- **Gradle**: Kotlin DSL, AGP 8.4.0, Kotlin 1.9.24
- **SDK**: minSdk 26, targetSdk 34, compileSdk 34, Java 17
- **CI**: GitHub Actions (`build.yml`) -- JDK 17, Android SDK 34, Gradle cache
  - Push to main/master: debug build + unit tests
  - Tag `v*`: release build + GitHub Release
  - Manual dispatch: debug or release + custom ONNX Runtime version
- **ProGuard**: Enabled for release (`isMinifyEnabled = true`,
  `isShrinkResources = true`)
