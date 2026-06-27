# Qwen3-TTS on-device pipeline (validated)

The whole chain (text → prefill → autoregressive codec generation → code2wav →
24 kHz audio) is reproduced purely from exported ONNX + raw tables and matches
PyTorch `model.generate` **bit-exact** (codes 100%, audio identical). Voices are
baked x-vectors from Russian reference clips; the app can also record new
reference clips from the mic (see §6) to add user voices on-device.

## 1. Build the model files (Google Colab, GPU)

```python
# deps (clean; NO onnxscript — legacy export is used)
!pip install -q torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
!pip install -q transformers==4.57.3 accelerate sentencepiece protobuf huggingface-hub qwen-tts onnx onnxruntime

import torch
from qwen_tts import Qwen3TTSModel
model_wrapper = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base", device_map="cpu", dtype=torch.float32)
model = model_wrapper.model; model.eval()

# export every block + tables (writes to /content/qwen3-tts-onnx/android)
exec(open(download("tools/export_talker_ar.py")).read())
```

Outputs (the on-device model set):

| file | what |
|---|---|
| `talker_step.onnx` | 28-layer talker, 1 step, KV cache → logits[3072] + hidden + KV |
| `subtalker_step.onnx` | 5-layer code predictor, 1 step, KV cache → hidden + KV |
| `codec_embed.onnx` | code0 (0..3071) → embedding[1024] |
| `code2wav.onnx` | codes[1,16,T] → audio |
| `text_cond_table.f16` | [151936,1024] = text_projection(text_embedding) |
| `subtalker_codec_embed.f16` | [15,2048,1024] residual embeddings |
| `subtalker_heads.f16` | [15,2048,1024] residual head weights (no bias) |

The three big ONNX are quantized to INT8 automatically at the end of
`export_talker_ar.py` (`quantize_for_android()`, QInt8, per_channel for the
talker) → ~430/80/210 MB, cutting the set from ~1.7 GB to ~700 MB.

## 2. Bake voices (x-vector path)

For each Russian reference clip (3–10 s clean speech; the file name becomes the
voice name): `generate_speaker_prompt(..., x_vector_only_mode=True)` = a 1024-d
x-vector. Save them concatenated as `baked_voices.bin` + `voices.json`. The
Colab notebook `tools/qwen3_tts_export_colab.ipynb` does this (cells 4–5) and
lets you upload your own Russian clips.

## 3. On-device algorithm (ported in QwenArEngine.kt)

Constants: H=1024, talker layers=28, subtalker layers=5, kv_heads=8, head_dim=128,
groups=15, EOS=2150, codebook0 range [0,2048), suppress 2048..3071 except EOS,
repetition_penalty=1.05. tts_bos/eos/pad=151672/151673/151671. codec
pad/bos/think/think_bos/think_eos=2148/2149/2154/2156/2157, russian lang=2069.

**Prefill** (build from input_ids = role(3)+text+suffix(5), and the voice x-vector):
```
role   = text_cond(input_ids[:3])                              # 3
codec0 = codec_embed([think, think_bos, lang, think_eos])      # 4
codec1 = codec_embed([codec_pad, codec_bos])                   # 2
codec_input = [codec0, xvector, codec1]                        # 7
tie    = [tts_pad*5, tts_bos] + codec_input[:6]                # 6
prefill = [role, tie, text_cond(input_ids[3]) + codec_input[6]]  # 10
trailing_text = [text_cond(input_ids[4:-5]), tts_eos]
```
Feed the 10 prefill vectors through `talker_step` one at a time (positions 0..9,
3D position_ids = arange, empty→growing KV). The last step's logits give code0[0]
and its hidden is `past_hidden`.

**AR loop** per frame (until code0==EOS):
```
last_id = codec_embed(code0)
# subtalker: feed past_hidden(pos0), last_id(pos1) → head[0] → code_1
#   then for g in 1..14: emb = subtalker_codec_embed[g-1][code_g]; step; head[g] → code_(g+1)
residual = 15 codes
frame = [code0] + residual
s = last_id + Σ subtalker_codec_embed[g][residual[g]]  (g=0..14)
s += (trailing_text[step] if step < len else tts_pad)
logits, hidden = talker_step(s, pos++, KV)
code0 = select(logits)   # repetition_penalty on code0 history + suppress specials
past_hidden = hidden; step++
```
Stack frames → `[1,16,T]` → `code2wav.onnx` → PCM.

Reference implementations: `tools/infer_onnx_reference.py` (AR loop) and
`tools/build_prefill.py` (prefill) — both validated against PyTorch.

## 4. Asset layout on device

`<filesDir>/models/qwen3-ar/` containing the files from step 1 (INT8) +
`baked_voices.bin` (concatenated float32 x-vectors) + `voices.json` (names) +
`vocab.json`, `merges.txt` for the tokenizer.

## 5. inputIdsFor (from ar_config.json, captured)

```
role_tokens   = [151644, 77091, 198]                 # <|im_start|>assistant\n
suffix_tokens = [151645, 198, 151644, 77091, 198]    # <|im_end|>\n<|im_start|>assistant\n
inputIdsFor(text) = role_tokens + Qwen3Tokenizer.encodeForTTS(text) + suffix_tokens
```
Voices baked: Russian reference clips (one x-vector each, named by file).

## 6. On-device voice cloning (mic)

The app records a 16 kHz mono reference clip (`VoiceRecorder`), stores it via
`CustomVoiceStore` under `filesDir/custom_voices/<id>.wav`, and shows it as a
voice chip on the Synthesis tab. When the AR engine (`QwenArEngine`) is wired
into the runtime, the speaker encoder turns each stored clip into an x-vector at
load time, so user voices behave like baked ones. Until then the clips are
persisted losslessly.
