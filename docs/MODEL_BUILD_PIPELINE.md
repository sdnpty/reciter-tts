# Пайплайн сборки моделей

Как экспортировать, квантовать и оптимизировать ONNX-модели Qwen3-TTS для Android.

Пайплайн выполняется в Google Colab (GPU желателен, но для экспорта не обязателен).

## Окружение

| Пакет          | Версия   |
|----------------|----------|
| PyTorch        | 2.5.1+cu121 |
| Transformers   | 4.57.3   |
| ONNX           | 1.22.0   |
| ONNX Runtime   | 1.27.0   |
| ONNX Optimizer | 0.4.2    |
| qwen-tts       | latest   |

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.57.3 accelerate sentencepiece protobuf huggingface-hub
pip install onnx onnxruntime onnxoptimizer safetensors numpy scipy librosa soundfile
pip install qwen-tts
```

## Шаг 1. Загрузка моделей

```python
import torch, os
from qwen_tts import Qwen3TTSModel
from transformers import AutoModel

MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

# Основная модель (talker + speaker_encoder)
model_wrapper = Qwen3TTSModel.from_pretrained(MODEL_NAME, device_map="cpu", dtype=torch.float16)
model = model_wrapper.model
model.eval()

talker = model.talker
speaker_encoder = model.speaker_encoder

# Модель-токенизатор (encoder + decoder/code2wav)
tokenizer_model = AutoModel.from_pretrained(
    "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    trust_remote_code=True,
    dtype=torch.float32,
    device_map="cpu"
)
tokenizer_model.eval()
```

### Размеры компонентов (FP32)

| Компонент         | Параметры    | Источник |
|-------------------|-------------|----------|
| talker            | 905 788 672 | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` |
| speaker_encoder   | 8 854 336   | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` |
| encoder (не используется) | 39 391 520 | `Qwen/Qwen3-TTS-Tokenizer-12Hz` |
| decoder (code2wav)| 114 323 137 | `Qwen/Qwen3-TTS-Tokenizer-12Hz` |

## Шаг 2. Экспорт в ONNX

Все экспорты используют **opset 18** с динамическими осями для переменной длины последовательности.

### Talker Base

Оборачивает эмбеддинг + слои трансформера (без lm_head), чтобы получить скрытые состояния (hidden states):

```python
class BaseTransformerWrapper(torch.nn.Module):
    def __init__(self, embed, model):
        super().__init__()
        self.embed = embed
        self.model = model

    def forward(self, input_ids, attention_mask):
        inputs_embeds = self.embed(input_ids)
        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return outputs.last_hidden_state

embed = talker.model.embed_tokens  # либо найти через named_modules
wrapper = BaseTransformerWrapper(embed, talker.model).float().eval()

dummy_ids = torch.randint(0, embed.num_embeddings, (1, 128), dtype=torch.long)
dummy_mask = torch.ones(1, 128, dtype=torch.long)

torch.onnx.export(
    wrapper, (dummy_ids, dummy_mask), "talker_base.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["hidden_states"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "hidden_states": {0: "batch_size", 1: "seq_len"}
    },
    opset_version=18, do_constant_folding=True
)
# Результат: ~1695 МБ
```

### Code Predictor Base

Тот же подход, что и для talker, но используется внутренний трансформер code predictor:

```python
code_predictor = talker.code_predictor
cp_model = code_predictor.model.float().eval()
cp_embed = cp_model.embed_tokens

# vocab_size обычно 2048 (размер кодовой книги)
dummy_cp_ids = torch.randint(0, cp_embed.num_embeddings, (1, 128), dtype=torch.long)

torch.onnx.export(
    CPBaseWrapper(cp_embed, cp_model),
    (dummy_cp_ids, torch.ones(1, 128, dtype=torch.long)),
    "code_predictor_base.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["hidden_states"],
    dynamic_axes={"input_ids": {0: "batch_size", 1: "seq_len"}, ...},
    opset_version=18, do_constant_folding=True
)
# Результат: ~308 МБ
```

### Code2Wav (вокодер)

Компонент-декодер из модели-токенизатора превращает кодек-токены в аудио:

```python
code2wav = tokenizer_model.decoder  # Qwen3TTSTokenizerV2Decoder
c2w_fp32 = code2wav.float().eval()

dummy_codes = torch.randint(0, 2048, (1, 16, 32), dtype=torch.long)

torch.onnx.export(
    c2w_fp32, (dummy_codes,), "code2wav.onnx",
    input_names=["codec_tokens"],
    output_names=["audio"],
    dynamic_axes={
        "codec_tokens": {0: "batch_size", 2: "seq_len"},
        "audio": {0: "batch_size", 2: "samples"}
    },
    opset_version=18, do_constant_folding=True
)
# Результат: ~436 МБ
```

Форма входа: `[batch, 16_quantizers, seq_len]` -> Форма выхода: `[batch, 1, samples]`

### Speaker Encoder

```python
se_fp32 = speaker_encoder.float().eval()
dummy_mel = torch.randn(1, 300, 128)  # мел-спектрограмма

torch.onnx.export(
    se_fp32, dummy_mel, "speaker_encoder.onnx",
    input_names=["mel_spectrogram"],
    output_names=["speaker_embedding"],
    dynamic_axes={"mel_spectrogram": {0: "batch"}, "speaker_embedding": {0: "batch"}},
    opset_version=18, do_constant_folding=True
)
# Результат: ~34 МБ
```

## Шаг 3. INT8-квантование

Динамическое квантование уменьшает размер моделей на 52–75 %:

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_int8(input_path, output_path, per_channel=False):
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        reduce_range=False
    )

quantize_int8("talker_base.onnx",          "talker_base_int8.onnx",          per_channel=True)
quantize_int8("code_predictor_base.onnx",  "code_predictor_base_int8.onnx",  per_channel=False)
quantize_int8("code2wav.onnx",             "code2wav_int8.onnx",             per_channel=False)
quantize_int8("speaker_encoder.onnx",      "speaker_encoder_int8.onnx",      per_channel=False)
```

| Модель             | FP32    | INT8   | Сжатие    |
|--------------------|---------|--------|-----------|
| talker_base        | 1695 МБ | 429 МБ | 75 %      |
| code_predictor_base| 308 МБ  | 77 МБ  | 75 %      |
| code2wav           | 436 МБ  | 210 МБ | 52 %      |
| speaker_encoder    | 34 МБ   | 9 МБ   | 74 %      |
| **Итого**          | **2473 МБ** | **724 МБ** | **71 %** |

## Шаг 4. Оптимизация под Android

```python
import onnx, onnxoptimizer

def optimize_android(input_path, output_path):
    model = onnx.load(input_path)
    passes = [
        "eliminate_identity",
        "fuse_consecutive_transposes",
        "extract_constant_to_initializer",
        "eliminate_unused_initializer"
    ]
    model = onnxoptimizer.optimize(model, passes)
    onnx.save(model, output_path)

optimize_android("talker_base_int8.onnx",          "talker_base_android.onnx")
optimize_android("code_predictor_base_int8.onnx",  "code_predictor_base_android.onnx")
optimize_android("code2wav_int8.onnx",             "code2wav_android.onnx")
optimize_android("speaker_encoder_int8.onnx",      "speaker_encoder_android.onnx")
```

## Шаг 5. Упаковка

```python
import zipfile

with zipfile.ZipFile("qwen3-tts-android.zip", 'w', zipfile.ZIP_DEFLATED) as zf:
    for name in ["talker_base", "code_predictor_base", "code2wav", "speaker_encoder"]:
        zf.write(f"{name}_android.onnx", f"models/{name}.onnx")
# Архив: ~605 МБ
```

Сервис `ModelDownloadService` в приложении ожидает ZIP-архив с файлами `.onnx`.
`ModelConfig.ARCHIVE_RENAME_MAP` сопоставляет имена в архиве (без суффикса `_android`)
с именами файлов, ожидаемыми на устройстве.

## Итоговые артефакты

```
/sdcard/Android/data/com.qwen3.tts/files/models/
  talker_base_android.onnx          428 МБ
  code_predictor_base_android.onnx   77 МБ
  code2wav_android.onnx             210 МБ
  speaker_encoder_android.onnx        9 МБ
  vocab.json                         (из токенизатора HuggingFace)
  merges.txt                         (из токенизатора HuggingFace)
```

## Добавление нового варианта модели

Чтобы добавить новую модель Qwen (например, `Qwen3-TTS-12Hz-1.5B-Base`):

1. Замените `MODEL_NAME` в Шаге 1 на новый чекпойнт
2. Прогоните весь пайплайн (шаги 2–5)
3. Добавьте новый `ModelProfile` в `ModelConfig.kt`:

```kotlin
val QWEN3_TTS_12HZ_15B = ModelProfile(
    id = "qwen3-tts-12hz-1.5b",
    displayName = "Qwen3 TTS 12Hz 1.5B",
    family = "qwen3-tts",
    sampleRateHz = 24_000,
    codecFrameRateHz = 12,
    modelFiles = listOf(
        ModelFile("talker_1.5b_android.onnx", 850L, Role.TALKER),
        // ... остальные файлы с подходящими размерами
    ),
    tokenizerFiles = listOf("vocab.json", "merges.txt")
)
```

4. Добавьте профиль в `SUPPORTED_PROFILES` и задайте `ACTIVE_PROFILE`
