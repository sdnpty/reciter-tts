# Model Build Pipeline

How to export, quantize, and optimize Qwen3-TTS ONNX models for Android.

The pipeline runs in Google Colab (GPU recommended but not required for export).

## Environment

| Package        | Version  |
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

## Step 1: Load Models

```python
import torch, os
from qwen_tts import Qwen3TTSModel
from transformers import AutoModel

MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

# Main model (talker + speaker_encoder)
model_wrapper = Qwen3TTSModel.from_pretrained(MODEL_NAME, device_map="cpu", dtype=torch.float16)
model = model_wrapper.model
model.eval()

talker = model.talker
speaker_encoder = model.speaker_encoder

# Tokenizer model (encoder + decoder/code2wav)
tokenizer_model = AutoModel.from_pretrained(
    "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    trust_remote_code=True,
    dtype=torch.float32,
    device_map="cpu"
)
tokenizer_model.eval()
```

### Component sizes (FP32)

| Component        | Parameters   | Source |
|------------------|-------------|--------|
| talker           | 905,788,672 | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` |
| speaker_encoder  | 8,854,336   | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` |
| encoder (unused) | 39,391,520  | `Qwen/Qwen3-TTS-Tokenizer-12Hz` |
| decoder (code2wav)| 114,323,137| `Qwen/Qwen3-TTS-Tokenizer-12Hz` |

## Step 2: Export to ONNX

All exports use **opset 18** with dynamic axes for variable sequence lengths.

### Talker Base

Wraps embedding + transformer layers (without lm_head) to get hidden states:

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

embed = talker.model.embed_tokens  # or search via named_modules
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
# Result: ~1695 MB
```

### Code Predictor Base

Same pattern as talker, using the code predictor's internal transformer:

```python
code_predictor = talker.code_predictor
cp_model = code_predictor.model.float().eval()
cp_embed = cp_model.embed_tokens

# vocab_size typically 2048 (codebook size)
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
# Result: ~308 MB
```

### Code2Wav (Vocoder)

The decoder component from the tokenizer model converts codec tokens to audio:

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
# Result: ~436 MB
```

Input shape: `[batch, 16_quantizers, seq_len]` -> Output: `[batch, 1, samples]`

### Speaker Encoder

```python
se_fp32 = speaker_encoder.float().eval()
dummy_mel = torch.randn(1, 300, 128)  # mel spectrogram

torch.onnx.export(
    se_fp32, dummy_mel, "speaker_encoder.onnx",
    input_names=["mel_spectrogram"],
    output_names=["speaker_embedding"],
    dynamic_axes={"mel_spectrogram": {0: "batch"}, "speaker_embedding": {0: "batch"}},
    opset_version=18, do_constant_folding=True
)
# Result: ~34 MB
```

## Step 3: INT8 Quantization

Dynamic quantization reduces model size by 52-75%:

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

| Model              | FP32    | INT8   | Reduction |
|--------------------|---------|--------|-----------|
| talker_base        | 1695 MB | 429 MB | 75%       |
| code_predictor_base| 308 MB  | 77 MB  | 75%       |
| code2wav           | 436 MB  | 210 MB | 52%       |
| speaker_encoder    | 34 MB   | 9 MB   | 74%       |
| **Total**          | **2473 MB** | **724 MB** | **71%** |

## Step 4: Optimize for Android

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

## Step 5: Package

```python
import zipfile

with zipfile.ZipFile("qwen3-tts-android.zip", 'w', zipfile.ZIP_DEFLATED) as zf:
    for name in ["talker_base", "code_predictor_base", "code2wav", "speaker_encoder"]:
        zf.write(f"{name}_android.onnx", f"models/{name}.onnx")
# Archive: ~605 MB
```

The app's `ModelDownloadService` expects a ZIP archive with `.onnx` files.
`ModelConfig.ARCHIVE_RENAME_MAP` maps archive names (without `_android` suffix)
to the filenames expected on device.

## Final artifacts

```
/sdcard/Android/data/com.qwen3.tts/files/models/
  talker_base_android.onnx          428 MB
  code_predictor_base_android.onnx   77 MB
  code2wav_android.onnx             210 MB
  speaker_encoder_android.onnx        9 MB
  vocab.json                         (from HuggingFace tokenizer)
  merges.txt                         (from HuggingFace tokenizer)
```

## Adding a New Model Variant

To add a new Qwen model (e.g., `Qwen3-TTS-12Hz-1.5B-Base`):

1. Replace `MODEL_NAME` in Step 1 with the new checkpoint
2. Run the full pipeline (steps 2-5)
3. Add a new `ModelProfile` in `ModelConfig.kt`:

```kotlin
val QWEN3_TTS_12HZ_15B = ModelProfile(
    id = "qwen3-tts-12hz-1.5b",
    displayName = "Qwen3 TTS 12Hz 1.5B",
    family = "qwen3-tts",
    sampleRateHz = 24_000,
    codecFrameRateHz = 12,
    modelFiles = listOf(
        ModelFile("talker_1.5b_android.onnx", 850L, Role.TALKER),
        // ... other files with appropriate sizes
    ),
    tokenizerFiles = listOf("vocab.json", "merges.txt")
)
```

4. Add the profile to `SUPPORTED_PROFILES` and set `ACTIVE_PROFILE`
