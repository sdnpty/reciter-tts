# Reciter TTS

Android TTS-движок на базе Qwen3-TTS с локальным инференсом через ONNX Runtime.

Регистрируется как системный `TextToSpeechService` — любое приложение,
использующее Android TTS API (TalkBack, читалки, ассистенты), работает с ним,
как только он выбран движком по умолчанию.

## Возможности

- **Полностью офлайн** — интернет нужен только для загрузки модели
- **Мультиязычность** — русский, английский, китайский
- **Интеграция с системным TTS** — работает с любым Android-приложением через стандартный TTS API
- **Тёмный UI на Material 3** — одна Activity с навигацией по 3 вкладкам
- **Расширяемость** — архитектура позволяет добавлять новые варианты моделей Qwen

## Требования

- Android 8.0+ (API 26)
- ARM64 (arm64-v8a)
- ~800 МБ свободного места (модели + кэш)
- Рекомендуется 4+ ГБ ОЗУ

## Архитектура

```
Текст -> Qwen3Tokenizer (BPE) -> Talker (авторегрессионный, 429 МБ)
      -> CodePredictor (77 МБ) -> Вокодер Code2Wav (210 МБ)
      -> PCM-16 аудио @ 24 кГц
```

4 ONNX-модели, квантованные в INT8, работают на ONNX Runtime 1.26.0.
Суммарный размер моделей: ~724 МБ.

Полная карта модулей и руководство по масштабированию на несколько моделей —
в [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Быстрый старт

### 1. Сборка

```bash
./gradlew assembleDebug
```

Нужен JDK 17. ONNX Runtime 1.26.0 подтягивается из Maven Central автоматически.

### 2. Установка моделей

Модели **не** входят в APK. Установить их можно одним из способов:

**Вариант A: загрузка в приложении**

Откройте приложение -> вкладка «Модели» -> «Скачать модели» -> вставьте свой
HTTPS-URL на ZIP-архив с 4 ONNX-файлами.

**Вариант B: локальный ZIP**

Вкладка «Модели» -> «Выбрать локальный ZIP» -> выберите ZIP-файл из памяти устройства.

**Вариант C: ручное копирование**

```
adb push models/ /sdcard/Android/data/com.qwen3.tts/files/models/
```

Ожидаемые файлы:
```
talker_base_android.onnx          428 МБ
code_predictor_base_android.onnx   77 МБ
code2wav_android.onnx             210 МБ
speaker_encoder_android.onnx        9 МБ
```

### 3. Токенизатор

BPE-токенизатор загружает `vocab.json` + `merges.txt` из
`app/src/main/assets/tokenizer/` (входят в APK) или из каталога моделей.

Скачать файлы токенизатора:
```bash
pip install huggingface-hub
huggingface-cli download Qwen/Qwen3-0.6B vocab.json merges.txt --local-dir app/src/main/assets/tokenizer/
```

### 4. Выбор движком по умолчанию

Откройте приложение -> вкладка «Синтез» -> «Сделать TTS по умолчанию» ->
выберите Reciter TTS в системных настройках.

## Сборка моделей с нуля

**Проще всего:** открыть [`tools/qwen3_tts_export_colab.ipynb`](tools/qwen3_tts_export_colab.ipynb)
в Google Colab (GPU-рантайм) и выполнить все ячейки. Ноутбук проходит все шаги
и скачивает готовый к импорту ZIP:

1. Установка зависимостей → 2. Загрузка Qwen3-TTS → 3. Экспорт **Talker с `lm_head`**
(логиты) → 4. Экспорт **Code Predictor с головой** → 5. Code2Wav → 6. Speaker
Encoder → 7. INT8-квантование → 8. Оптимизация под Android → 9. Токенизатор
`vocab.json`+`merges.txt` → 10. **Манифест `model.json`** → 11. ZIP.

Та же логика — в [`tools/export_onnx_corrected.py`](tools/export_onnx_corrected.py).
Затем в приложении: «Модели» → «Выбрать локальный ZIP».

Полные заметки по пайплайну (экспорт из PyTorch -> ONNX -> INT8-квантование ->
оптимизация под Android) — в [docs/MODEL_BUILD_PIPELINE.md](docs/MODEL_BUILD_PIPELINE.md).

## Добавление новых вариантов моделей

Приложение определяет **языки, голоса, роли файлов, частоту дискретизации и
смещения токенов** модели во время выполнения — из необязательного манифеста
`model.json` внутри ZIP-архива моделей. Чтобы добавить модель того же семейства,
менять код приложения не нужно — достаточно вложить манифест.

### Формат `model.json`

Положите файл в корень ZIP-архива моделей (рядом с `.onnx`-файлами):

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

- `architecture` выбирает движок инференса: `qwen3-codec` (по умолчанию) или
  `vits`. Другие семейства (cosyvoice, f5, xtts) — в планах, см.
  [docs/MODELS.md](docs/MODELS.md).
- `audioTokenStart` / `eosTokenId` позволяют авторегрессионному декодеру
  подстроиться под точную раскладку словаря вашего экспорта (критично — см. ARCHITECTURE).
- `voices` формирует список голосов системного TTS, доступность языков и
  чипы языков в приложении.
- Любое пропущенное поле берётся из встроенного профиля в `ModelConfig.kt`.

Если манифеста нет, используется compile-time-профиль `ModelConfig.ACTIVE_PROFILE`.

Подробности — в [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## CI / GitHub Actions

- Push в `main` -> debug-сборка + юнит-тесты
- Тег `v*` -> release-сборка + GitHub Release
- Ручной запуск (workflow dispatch) -> debug/release + произвольная версия ONNX Runtime

### Подпись release-сборок (опционально)

Добавьте в репозиторий секреты:
- `KEYSTORE_BASE64` — keystore.jks в base64
- `KEYSTORE_PASSWORD`
- `KEY_ALIAS`
- `KEY_PASSWORD`

## Версия ONNX Runtime

Проект использует **ONNX Runtime 1.26.0** из Maven Central — последнюю версию
с Android AAR. Версия 1.27.0 существует только для Python/C++.

Чтобы использовать локальный AAR:
```bash
wget https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/1.26.0/onnxruntime-android-1.26.0.aar
mv onnxruntime-android-1.26.0.aar app/libs/
```
Затем в `app/build.gradle.kts` замените Maven-зависимость на:
```kotlin
implementation(files("libs/onnxruntime-android-1.26.0.aar"))
```

## Структура проекта

```
app/src/main/java/com/qwen3/tts/
  engine/              TTS-сервис + ONNX-инференс + токенизатор
  service/             Foreground-сервис загрузки моделей
  ui/                  MainActivity + 3 фрагмента + ViewModel
  util/                ModelConfig, AudioHelper, TTSLogger
```

## Лицензия

Apache 2.0
