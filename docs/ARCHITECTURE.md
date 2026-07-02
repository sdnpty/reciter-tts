# Архитектура

## Обзор

Reciter TTS — Android TTS-движок, выполняющий инференс Qwen3-TTS локально
через ONNX Runtime. Он регистрируется как системный `TextToSpeechService`,
поэтому любое приложение, использующее Android TTS API, автоматически работает
с ним, как только он выбран движком по умолчанию.

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
     PCM-16 аудио @ 24 кГц
         |
  SynthesisCallback -> AudioTrack
```

## TTS-пайплайн

1. **Токенизация** — `Qwen3Tokenizer` (BPE, vocab.json + merges.txt)
   преобразует текст в ID токенов. Если файлы словаря отсутствуют, откатывается
   на байтовый кириллический токенизатор.

2. **Talker** (авторегрессионный) — пропускает ID токенов через трансформер и
   жадно декодирует ID аудио-токенов до EOS. Аудио-токены начинаются с ID 151936
   (`TokenizerConstants.AUDIO_TOKEN_START`). Выдаёт грубые кодек-коды
   (квантователь 0).

3. **CodePredictor** — по грубым кодам предсказывает тонкие коды для
   квантователей 1..15 через argmax по логитам. Если модуль недоступен, грубые
   коды дублируются по всем квантователям (качество ниже, но работает).

4. **Code2Wav** (вокодер) — превращает полный кодек-тензор `[1, 16, seq]` в
   сырую аудио-волну `[1, 1, samples]` на 24 кГц.

5. **Преобразование в PCM** — `AudioHelper.floatToPcm16()` конвертирует
   float-сэмплы в 16-битные PCM-байты, которые стримятся через `SynthesisCallback`.

## Карта модулей

```
com.qwen3.tts/
  QwenTTSApplication         -- глобальный обработчик крэшей, инициализация логгера

  engine/
    QwenTTSEngine             -- реализация TextToSpeechService
    CheckVoiceData            -- Activity проверки голосовых данных Android TTS
    GetSampleText             -- Activity выдачи образца текста Android TTS
    InstallVoiceData          -- Activity установки голосовых данных Android TTS
    inference/
      QwenInferenceEngine     -- управление ONNX-сессиями + инференс
    tokenizer/
      Qwen3Tokenizer          -- BPE-токенизатор (синглтон)

  service/
    ModelDownloadService      -- foreground-сервис загрузки/импорта моделей

  ui/
    MainActivity              -- одна Activity с BottomNavigationView
    TTSViewModel              -- общая ViewModel (статус модели, логгер, токенизатор)
    fragment/
      SynthesisFragment       -- ввод текста, выбор языка, воспроизведение TTS
      ModelsFragment          -- статус моделей, загрузка/импорт/удаление, аудиотесты
      SettingsFragment        -- speaker ID, скорость, устройство, логирование, просмотр логов

  util/
    ModelConfig               -- профили моделей, пути к файлам, проверки готовности
    TokenizerConstants        -- размер словаря, ID специальных токенов
    AudioHelper               -- конвертация PCM, конструктор AudioTrack
    TTSLogger                 -- файловый логгер с ротацией
    LogShareHelper            -- шаринг лог-файлов через FileProvider
```

## Слоты моделей (несколько установленных моделей)

Несколько моделей могут быть установлены одновременно. Каждая живёт в своём
слоте-каталоге внутри корня моделей:

```
<external-files>/models/
  <slotId-A>/  talker_base_android.onnx … vocab.json merges.txt model.json
  <slotId-B>/  …
  talker_base_android.onnx …            # старая плоская раскладка = слот "default"
```

- Импорт/загрузка распаковывается в `models/_incoming/`, затем `finalizeSlot()`
  переносит его в `models/<id>/` (id берётся из `model.json`, иначе генерируется)
  и делает активным.
- `ModelConfig.installedModels()` сканирует каталоги-слоты (+ старую плоскую раскладку).
- ID активного слота хранится в SharedPreferences (`active_model_id`) и
  выбирается в «Настройки → Активная модель». `activeModelDir()` / `activeProfile()`
  разрешают всё (пути движка, токенизатор, проверки наличия) относительно него.
- Переключение модели сбрасывает токенизатор; TTS-движок подхватывает новую
  модель при следующей (пере)загрузке.

## Мультимодельная архитектура

`ModelConfig` поддерживает несколько профилей моделей через `ModelProfile`:

```kotlin
data class ModelProfile(
    val id: String,              // "qwen3-tts-12hz-0.6b"
    val displayName: String,     // отображается в UI
    val family: String,          // "qwen3-tts" -- группирует родственные модели
    val sampleRateHz: Int,       // 24000
    val codecFrameRateHz: Int,   // 12
    val modelFiles: List<ModelFile>,
    val tokenizerFiles: List<String>
)
```

У каждого `ModelFile` есть `Role` (TALKER, CODE_PREDICTOR, VOCODER, SPEAKER_ENCODER)
и метаданные (`expectedSizeMb`, `requiredForSynthesis`).

Чтобы добавить новую модель:

1. Экспортируйте ONNX-файлы (см. [MODEL_BUILD_PIPELINE.md](MODEL_BUILD_PIPELINE.md))
2. Добавьте `ModelProfile` в `ModelConfig.SUPPORTED_PROFILES`
3. Движок находит файлы по `Role`, поэтому другие размеры и имена файлов
   работают без изменения кода движка

Сейчас `ACTIVE_PROFILE` задаётся на этапе компиляции. Чтобы сделать его
переключаемым во время выполнения:
- Храните ID выбранного профиля в SharedPreferences
- Читайте его в `QwenTTSEngine.initEngine()` и `TTSViewModel`
- Перезапускайте TTS-сервис при смене профиля

## Конфигурация ONNX Runtime

- **CPU по умолчанию**: intra-op-потоки = `availableProcessors()` с ограничением
  1..4, 1 inter-op-поток, оптимизация графа `ALL_OPT`, преаллокация memory-pattern
  отключена, чтобы нативный след памяти оставался небольшим и предсказуемым.
- **NNAPI включается вручную** (Настройки → Движок → NNAPI). Он может ускорить
  вокодер, но INT8-квантованные модели на многих драйверах устройств падают
  *нативно* (SIGABRT) — это обычная причина «мгновенного закрытия» движка без
  записей в логах. CPU — стабильный вариант по умолчанию везде.
- **Загрузка вне главного потока**: TTS-сервис загружает сессии в фоновом
  потоке; первый запрос синтеза ждёт на latch вместо того, чтобы падать.
- **«Хлебные крошки» крэшей**: `TTSLogger.beginLoadStage()` синхронно пишет
  чекпоинт перед каждым нативным `createSession`. Если процесс умирает нативно,
  следующий запуск читает оставшийся чекпоинт и сообщает, где именно произошёл
  крэш (`consumeStaleLoadCrash()`) — нативные падения больше не «молчаливые».
- **Версия рантайма**: 1.26.0 (Maven Central). Совместима с INT8-квантованными
  моделями на opset 18.

## Безопасность

- **Только HTTPS-загрузки**: `ModelDownloadService` отклоняет не-HTTPS URL
- **Защита от path traversal**: распаковка ZIP проверяет канонические пути
- **Network security config**: `network_security_config.xml` ограничивает
  незашифрованный трафик
- **Никаких секретов в коде**: URL загрузки задаёт пользователь во время работы

## Сборка и CI

- **Gradle**: Kotlin DSL, AGP 8.4.0, Kotlin 1.9.24
- **SDK**: minSdk 26, targetSdk 34, compileSdk 34, Java 17
- **CI**: GitHub Actions (`build.yml`) — JDK 17, Android SDK 34, кэш Gradle
  - Push в main/master: debug-сборка + юнит-тесты
  - Тег `v*`: release-сборка + GitHub Release
  - Ручной запуск: debug или release + произвольная версия ONNX Runtime
- **ProGuard**: включён для release (`isMinifyEnabled = true`,
  `isShrinkResources = true`)
