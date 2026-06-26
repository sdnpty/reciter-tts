# Reciter TTS

Android TTS engine powered by Qwen3-TTS with on-device ONNX Runtime inference.

Registers as a system `TextToSpeechService` -- any app that uses Android's TTS
API (TalkBack, e-readers, assistants) uses it once set as default.

## Features

- **Fully local** -- no internet needed after model download
- **Multi-language** -- Russian, English, Chinese
- **System TTS integration** -- works with any Android app via standard TTS API
- **Material 3 dark UI** -- single Activity with 3-tab navigation
- **Extensible** -- architecture supports adding new Qwen model variants

## Requirements

- Android 8.0+ (API 26)
- ARM64 (arm64-v8a)
- ~800 MB free storage (models + cache)
- 4+ GB RAM recommended

## Architecture

```
Text -> Qwen3Tokenizer (BPE) -> Talker (autoregressive, 429 MB)
     -> CodePredictor (77 MB) -> Code2Wav vocoder (210 MB)
     -> PCM-16 audio @ 24 kHz
```

4 ONNX models, INT8 quantized, running on ONNX Runtime 1.26.0.
Total model size: ~724 MB.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full module map and
multi-model scaling guide.

## Quick Start

### 1. Build

```bash
./gradlew assembleDebug
```

JDK 17 required. ONNX Runtime 1.26.0 is pulled from Maven Central automatically.

### 2. Install Models

Models are **not** bundled in the APK. Install them via one of:

**Option A: Download in-app**

Open the app -> Models tab -> Download Models -> paste your HTTPS URL to a
ZIP archive containing the 4 ONNX files.

**Option B: Pick local ZIP**

Models tab -> Pick Local ZIP -> select a ZIP file from device storage.

**Option C: Manual copy**

```
adb push models/ /sdcard/Android/data/com.qwen3.tts/files/models/
```

Expected files:
```
talker_base_android.onnx          428 MB
code_predictor_base_android.onnx   77 MB
code2wav_android.onnx             210 MB
speaker_encoder_android.onnx        9 MB
```

### 3. Tokenizer

The BPE tokenizer loads `vocab.json` + `merges.txt` from
`app/src/main/assets/tokenizer/` (bundled in APK) or from the models directory.

To download tokenizer files:
```bash
pip install huggingface-hub
huggingface-cli download Qwen/Qwen3-0.6B vocab.json merges.txt --local-dir app/src/main/assets/tokenizer/
```

### 4. Set as Default TTS

Open the app -> Synthesis tab -> "Set as Default TTS" -> select Reciter TTS
in system settings.

## Building Models from Scratch

**Easiest:** open [`tools/qwen3_tts_export_colab.ipynb`](tools/qwen3_tts_export_colab.ipynb)
in Google Colab (GPU runtime) and run all cells. It performs every step and
downloads a ready-to-import ZIP:

1. Install deps → 2. Load Qwen3-TTS → 3. Export **Talker with `lm_head`**
(logits) → 4. Export **Code Predictor with head** → 5. Code2Wav → 6. Speaker
Encoder → 7. INT8 quantize → 8. Android optimize → 9. tokenizer
`vocab.json`+`merges.txt` → 10. **`model.json` manifest** → 11. ZIP.

The same logic lives in [`tools/export_onnx_corrected.py`](tools/export_onnx_corrected.py).
Then in the app: Models → Pick Local ZIP.

See [docs/MODEL_BUILD_PIPELINE.md](docs/MODEL_BUILD_PIPELINE.md) for the full
pipeline notes: PyTorch export -> ONNX -> INT8 quantization -> Android optimization.

## Adding New Model Variants

The app discovers a model's **languages, voices, file roles, sample rate and
token offsets** at runtime from an optional `model.json` manifest shipped inside
the models ZIP. No app code change is needed to add a model of the family —
just include a manifest.

### `model.json` format

Place this file at the root of the models ZIP (next to the `.onnx` files):

```json
{
  "schemaVersion": 1,
  "id": "qwen3-tts-12hz-0.6b",
  "displayName": "Qwen3 TTS 0.6B",
  "family": "qwen3-tts",
  "architecture": "qwen3-codec",
  "sampleRateHz": 24000,
  "codecFrameRateHz": 12,
  "audioTokenStart": 151936,
  "eosTokenId": 151645,
  "tokenizer": { "type": "qwen-bpe", "files": ["vocab.json", "merges.txt"] },
  "files": [
    { "filename": "talker_base_android.onnx",         "role": "TALKER",          "sizeMb": 428, "required": true },
    { "filename": "code_predictor_base_android.onnx", "role": "CODE_PREDICTOR",  "sizeMb": 77,  "required": false },
    { "filename": "code2wav_android.onnx",            "role": "VOCODER",         "sizeMb": 210, "required": true },
    { "filename": "speaker_encoder_android.onnx",     "role": "SPEAKER_ENCODER", "sizeMb": 9,   "required": false }
  ],
  "voices": [
    { "id": "qwen3_ru", "locale": "ru-RU", "displayName": "Русский", "speakerId": 0 },
    { "id": "qwen3_en", "locale": "en-US", "displayName": "English", "speakerId": 1 },
    { "id": "qwen3_zh", "locale": "zh-CN", "displayName": "中文",     "speakerId": 2 }
  ]
}
```

- `architecture` selects the inference engine: `qwen3-codec` (default) or
  `vits`. Other families (cosyvoice, f5, xtts) are planned — see
  [docs/MODELS.md](docs/MODELS.md).
- `audioTokenStart` / `eosTokenId` let the autoregressive decoder match the
  exact vocab layout of your export (critical — see ARCHITECTURE).
- `voices` drives the system TTS voice list, language availability, and the
  in-app language chips.
- Any field omitted falls back to the bundled profile in `ModelConfig.kt`.

If no manifest is present, the compile-time `ModelConfig.ACTIVE_PROFILE` is used.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

## CI / GitHub Actions

- Push to `main` -> debug build + unit tests
- Tag `v*` -> release build + GitHub Release
- Workflow dispatch -> debug/release + custom ONNX Runtime version

### Release Signing (optional)

Add these secrets to your repository:
- `KEYSTORE_BASE64` -- base64-encoded keystore.jks
- `KEYSTORE_PASSWORD`
- `KEY_ALIAS`
- `KEY_PASSWORD`

## ONNX Runtime Version

The project uses **ONNX Runtime 1.26.0** from Maven Central -- the latest
version with an Android AAR. Version 1.27.0 is Python/C++ only.

To use a local AAR instead:
```bash
wget https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/1.26.0/onnxruntime-android-1.26.0.aar
mv onnxruntime-android-1.26.0.aar app/libs/
```
Then in `app/build.gradle.kts`, replace the Maven dependency with:
```kotlin
implementation(files("libs/onnxruntime-android-1.26.0.aar"))
```

## Project Structure

```
app/src/main/java/com/qwen3/tts/
  engine/              TTS service + ONNX inference + tokenizer
  service/             Model download foreground service
  ui/                  MainActivity + 3 Fragments + ViewModel
  util/                ModelConfig, AudioHelper, TTSLogger
```

## License

Apache 2.0
