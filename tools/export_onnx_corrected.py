#!/usr/bin/env python3
"""
Corrected ONNX export script for Qwen3-TTS 0.6B → Android.

CRITICAL FIXES vs the original script:
  1. Talker is exported WITH lm_head (not just backbone) — required for token generation
  2. Code Predictor is exported WITH its head — required for fine code prediction
  3. ZIP archive names match what Android expects (*_android.onnx)
  4. vocab.json + merges.txt are included in the ZIP for the Android tokenizer
  5. Code2Wav input shape documented: (batch, num_quantizers=16, seq_len)
  6. Emits model.json manifest (languages/voices + audioTokenStart/eosTokenId)
     so the app auto-detects the model's capabilities — see docs/model.example.json

Output ZIP layout (extracted into the app's models dir):
  models/talker_base_android.onnx
  models/code_predictor_base_android.onnx
  models/code2wav_android.onnx
  models/speaker_encoder_android.onnx
  models/vocab.json
  models/merges.txt
  models/model.json

Run in Google Colab with GPU runtime.
"""

import os
import sys
import zipfile

# ══════════════════════════════════════════════════════════════
# CELL 1: Install dependencies
# ══════════════════════════════════════════════════════════════
# !pip install -q torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
# !pip install -q transformers==4.57.3 accelerate sentencepiece protobuf huggingface-hub
# !pip install -q onnx onnxruntime onnxoptimizer safetensors numpy scipy librosa soundfile
# !pip install -q qwen-tts

import torch
import onnx
import onnxoptimizer
from onnxruntime.quantization import quantize_dynamic, QuantType

WORK_DIR = "/content/qwen3-tts-onnx"
ONNX_DIR = f"{WORK_DIR}/onnx"
QUANT_DIR = f"{WORK_DIR}/quantized"
ANDROID_DIR = f"{WORK_DIR}/android"

for d in [WORK_DIR, ONNX_DIR, QUANT_DIR, ANDROID_DIR]:
    os.makedirs(d, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# CELL 2: Load models
# ══════════════════════════════════════════════════════════════
MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

print("Loading Qwen3-TTS-0.6B...")
from qwen_tts import Qwen3TTSModel
try:
    model_wrapper = Qwen3TTSModel.from_pretrained(
        MODEL_NAME, device_map="cpu", dtype=torch.float16
    )
except TypeError:
    model_wrapper = Qwen3TTSModel.from_pretrained(MODEL_NAME, dtype=torch.float16)
    model_wrapper.to("cpu")

model = model_wrapper.model
model.eval()
print("Main model loaded")

print("Loading Qwen3-TTS-Tokenizer-12Hz...")
from transformers import AutoModel
tokenizer_model = AutoModel.from_pretrained(
    "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    trust_remote_code=True, dtype=torch.float32, device_map="cpu"
)
tokenizer_model.eval()
print("Tokenizer model loaded")

talker = model.talker if hasattr(model, 'talker') else model
speaker_encoder = model.speaker_encoder if hasattr(model, 'speaker_encoder') else None

# ══════════════════════════════════════════════════════════════
# CELL 3: Export Talker WITH lm_head (CRITICAL FIX)
# ══════════════════════════════════════════════════════════════
print("\n=== Exporting Talker (with lm_head) ===")

# Find embedding layer
embed = None
for attr in ['embed_tokens', 'word_embeddings', 'token_embeddings', 'wte', 'embeddings']:
    if hasattr(talker, attr):
        embed = getattr(talker, attr)
        break
    elif hasattr(talker.model, attr):
        embed = getattr(talker.model, attr)
        break
if embed is None:
    for name, module in talker.named_modules():
        if isinstance(module, torch.nn.Embedding):
            embed = module
            break

# Find lm_head
lm_head = None
for attr in ['lm_head', 'output_projection', 'classifier']:
    if hasattr(talker, attr):
        lm_head = getattr(talker, attr)
        print(f"  Found talker.{attr}: {type(lm_head).__name__}")
        break

if lm_head is None:
    print("  WARNING: No lm_head found! Exporting backbone only (will NOT support generation)")
elif isinstance(lm_head, torch.nn.ModuleList):
    print(f"  lm_head is a ModuleList with {len(lm_head)} heads")

# Diagnostic: list talker heads so we can confirm we export the AUDIO head.
# If lm_head.out_features == text vocab size, the talker emits text tokens and
# the audio codec head is elsewhere (codec_head / audio_head / mtp / thinker).
print("  talker children:", [n for n, _ in talker.named_children()])
for cand in ['lm_head', 'codec_head', 'audio_head', 'audio_lm_head',
             'output_projection', 'mtp', 'classifier']:
    if hasattr(talker, cand):
        h = getattr(talker, cand)
        of = getattr(h, 'out_features', None)
        if of is None and isinstance(h, torch.nn.ModuleList) and len(h) > 0:
            of = f"ModuleList[{len(h)}] x out_features={getattr(h[0], 'out_features', '?')}"
        print(f"    talker.{cand}: {type(h).__name__} out_features={of}")


def apply_head(head, hidden):
    """Apply an output head to hidden states, returning logits.

    Handles a single Linear head as well as a ModuleList of per-quantizer
    heads (Qwen3-TTS code predictor): in that case logits from every head are
    concatenated on the last dim → [B, T, num_heads * codebook], which the
    Android decoder splits back into per-quantizer chunks.
    """
    if head is None:
        return hidden
    if isinstance(head, torch.nn.ModuleList):
        return torch.cat([m(hidden) for m in head], dim=-1)
    return head(hidden)


class TalkerWithHead(torch.nn.Module):
    """Exports the full talker: embed → backbone → lm_head → logits."""
    def __init__(self, embed, backbone, lm_head):
        super().__init__()
        self.embed = embed
        self.backbone = backbone
        self.lm_head = lm_head

    def forward(self, input_ids, attention_mask):
        inputs_embeds = self.embed(input_ids)
        outputs = self.backbone(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        return apply_head(self.lm_head, hidden)


wrapper = TalkerWithHead(embed, talker.model, lm_head).float().eval()

vocab_size = embed.num_embeddings if hasattr(embed, 'num_embeddings') else 32000
dummy_ids = torch.randint(0, vocab_size, (1, 128), dtype=torch.long)
dummy_mask = torch.ones(1, 128, dtype=torch.long)

talker_path = f"{ONNX_DIR}/talker_base.onnx"
with torch.no_grad():
    out = wrapper(dummy_ids, dummy_mask)
    print(f"  Forward OK: {out.shape}")
    # Width of the talker logits — audio codec tokens live in the upper range,
    # used below to compute audioTokenStart for the manifest.
    TALKER_OUTPUT_DIM = int(out.shape[-1])
    output_name = "logits" if lm_head is not None else "hidden_states"
    torch.onnx.export(
        wrapper, (dummy_ids, dummy_mask), talker_path,
        input_names=["input_ids", "attention_mask"],
        output_names=[output_name],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            output_name: {0: "batch", 1: "seq"}
        },
        opset_version=17,
        do_constant_folding=True
    )
print(f"  Talker: {os.path.getsize(talker_path) / 1024**2:.0f} MB")

# ══════════════════════════════════════════════════════════════
# CELL 4: Export Code Predictor WITH head (CRITICAL FIX)
# ══════════════════════════════════════════════════════════════
print("\n=== Exporting Code Predictor (with head) ===")

code_predictor = talker.code_predictor
cp_model = code_predictor.model
cp_model.eval()

# Find CP embedding
cp_embed = None
for attr in ['embed_tokens', 'word_embeddings', 'token_embeddings', 'wte', 'embeddings']:
    if hasattr(code_predictor, attr):
        cp_embed = getattr(code_predictor, attr)
        break
    elif hasattr(cp_model, attr):
        cp_embed = getattr(cp_model, attr)
        break
if cp_embed is None:
    for name, module in code_predictor.named_modules():
        if isinstance(module, torch.nn.Embedding):
            cp_embed = module
            break

# Find CP head
cp_head = None
for attr in ['lm_head', 'output_projection', 'code_head', 'classifier']:
    if hasattr(code_predictor, attr):
        cp_head = getattr(code_predictor, attr)
        print(f"  Found code_predictor.{attr}: {type(cp_head).__name__}")
        break

if cp_head is None:
    print("  WARNING: No head found for code_predictor!")
elif isinstance(cp_head, torch.nn.ModuleList):
    print(f"  code_predictor head is a ModuleList with {len(cp_head)} heads "
          f"(one per residual quantizer) — logits will be concatenated")


def apply_head(head, hidden):
    """Apply an output head (single Linear or a ModuleList of per-quantizer
    heads) to hidden states. ModuleList outputs are concatenated on the last
    dim → [B, T, num_heads * codebook]."""
    if head is None:
        return hidden
    if isinstance(head, torch.nn.ModuleList):
        return torch.cat([m(hidden) for m in head], dim=-1)
    return head(hidden)


class CPWithHead(torch.nn.Module):
    def __init__(self, embed, backbone, head):
        super().__init__()
        self.embed = embed
        self.backbone = backbone
        self.head = head

    def forward(self, input_ids, attention_mask):
        inputs_embeds = self.embed(input_ids)
        outputs = self.backbone(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        return apply_head(self.head, hidden)


cp_wrapper = CPWithHead(cp_embed, cp_model, cp_head).float().eval()

cp_vocab = cp_embed.num_embeddings if hasattr(cp_embed, 'num_embeddings') else 2048
dummy_cp = torch.randint(0, cp_vocab, (1, 128), dtype=torch.long)
dummy_cp_mask = torch.ones(1, 128, dtype=torch.long)

cp_path = f"{ONNX_DIR}/code_predictor_base.onnx"
with torch.no_grad():
    out = cp_wrapper(dummy_cp, dummy_cp_mask)
    print(f"  Forward OK: {out.shape}")
    cp_out_name = "logits" if cp_head is not None else "hidden_states"
    torch.onnx.export(
        cp_wrapper, (dummy_cp, dummy_cp_mask), cp_path,
        input_names=["input_ids", "attention_mask"],
        output_names=[cp_out_name],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            cp_out_name: {0: "batch", 1: "seq"}
        },
        opset_version=17,
        do_constant_folding=True
    )
print(f"  Code Predictor: {os.path.getsize(cp_path) / 1024**2:.0f} MB")

# ══════════════════════════════════════════════════════════════
# CELL 5: Export Code2Wav (decoder)
# ══════════════════════════════════════════════════════════════
print("\n=== Exporting Code2Wav ===")

# Diagnostic: show the tokenizer model structure to locate the decoder.
print("  tokenizer_model children:")
for name, module in tokenizer_model.named_children():
    n = sum(p.numel() for p in module.parameters())
    print(f"    - {name}: {type(module).__name__} ({n:,} params)")

code2wav = None
c2w_name = None
for attr in ['decoder', 'code2wav', 'vocoder', 'generator', 'wave_decoder',
             'codec_decoder', 'acoustic_decoder', 'dac', 'quantizer_decoder', 'wavenet']:
    if hasattr(tokenizer_model, attr):
        code2wav = getattr(tokenizer_model, attr); c2w_name = attr; break

# Fallback: pick the largest decoder-ish child module.
if code2wav is None:
    best, best_params = None, 0
    for name, module in tokenizer_model.named_children():
        if any(k in name.lower() for k in ['dec', 'wav', 'vocod', 'gen']):
            p = sum(pp.numel() for pp in module.parameters())
            if p > best_params:
                best, best_params, c2w_name = module, p, name
    code2wav = best

code2wav_path = None
if code2wav is None:
    print("  Code2Wav NOT found — share the children list above so it can be located.")
else:
    print(f"  Using tokenizer_model.{c2w_name}: {type(code2wav).__name__}")
    c2w = code2wav.float().eval()
    code2wav_path = f"{ONNX_DIR}/code2wav.onnx"

    # The codec decoder may expect codes as [B, Q, T] or [B, T, Q], int64/int32.
    attempts = [
        ("codes[1,16,32] int64", torch.randint(0, 2048, (1, 16, 32), dtype=torch.long), 2),
        ("codes[1,32,16] int64", torch.randint(0, 2048, (1, 32, 16), dtype=torch.long), 1),
        ("codes[1,16,32] int32", torch.randint(0, 2048, (1, 16, 32), dtype=torch.int32), 2),
    ]
    exported = False
    for desc, dummy, seq_ax in attempts:
        try:
            with torch.no_grad():
                out = c2w(dummy)
            shape = tuple(out.shape) if hasattr(out, "shape") else type(out).__name__
            print(f"  Forward OK with {desc}: {shape}")
            with torch.no_grad():
                torch.onnx.export(
                    c2w, (dummy,), code2wav_path,
                    input_names=["codec_tokens"],
                    output_names=["audio"],
                    dynamic_axes={
                        "codec_tokens": {0: "batch", seq_ax: "seq_len"},
                        "audio": {0: "batch", 2: "samples"}
                    },
                    opset_version=17,
                    do_constant_folding=True
                )
            print(f"  Code2Wav: {os.path.getsize(code2wav_path) / 1024**2:.0f} MB")
            exported = True
            break
        except Exception as e:
            print(f"  attempt '{desc}' failed: {repr(e)[:200]}")
    if not exported:
        code2wav_path = None
        print("  Code2Wav export FAILED for all input shapes — share the errors above.")

# ══════════════════════════════════════════════════════════════
# CELL 6: Export Speaker Encoder
# ══════════════════════════════════════════════════════════════
print("\n=== Exporting Speaker Encoder ===")

se_path = None
if speaker_encoder is not None:
    speaker_encoder.eval()
    se = speaker_encoder.float()

    for shape in [(1, 300, 128), (1, 80, 300), (1, 128, 300)]:
        try:
            test = torch.randn(*shape)
            with torch.no_grad():
                out = se(test)
            print(f"  Input shape: {shape}, Output: {out.shape}")
            se_path = f"{ONNX_DIR}/speaker_encoder.onnx"
            with torch.no_grad():
                torch.onnx.export(
                    se, test, se_path,
                    input_names=["mel_spectrogram"],
                    output_names=["speaker_embedding"],
                    dynamic_axes={
                        "mel_spectrogram": {0: "batch"},
                        "speaker_embedding": {0: "batch"}
                    },
                    opset_version=17,
                    do_constant_folding=True
                )
            print(f"  Speaker Encoder: {os.path.getsize(se_path) / 1024**2:.0f} MB")
            break
        except:
            continue
else:
    print("  Speaker Encoder not found")

# ══════════════════════════════════════════════════════════════
# CELL 7: INT8 Quantization
# ══════════════════════════════════════════════════════════════
print("\n=== INT8 Quantization ===")

def quantize_int8(input_path, output_path, per_channel=False):
    if not input_path or not os.path.exists(input_path):
        return None
    quantize_dynamic(input_path, output_path,
                     weight_type=QuantType.QInt8,
                     per_channel=per_channel, reduce_range=False)
    orig = os.path.getsize(input_path) / 1024**2
    quant = os.path.getsize(output_path) / 1024**2
    print(f"  {os.path.basename(input_path)}: {orig:.0f} MB -> {quant:.0f} MB ({(1 - quant/orig)*100:.0f}%)")
    return output_path

quantized = {}
q = quantize_int8(talker_path, f"{QUANT_DIR}/talker_base_int8.onnx", per_channel=True)
if q: quantized['talker_base'] = q
q = quantize_int8(cp_path, f"{QUANT_DIR}/code_predictor_base_int8.onnx")
if q: quantized['code_predictor_base'] = q
if code2wav_path:
    q = quantize_int8(code2wav_path, f"{QUANT_DIR}/code2wav_int8.onnx")
    if q: quantized['code2wav'] = q
if se_path:
    q = quantize_int8(se_path, f"{QUANT_DIR}/speaker_encoder_int8.onnx")
    if q: quantized['speaker_encoder'] = q

# ══════════════════════════════════════════════════════════════
# CELL 8: Optimize for Android
# ══════════════════════════════════════════════════════════════
print("\n=== Android Optimization ===")

def optimize_android(input_path, output_path):
    if not os.path.exists(input_path):
        return None
    m = onnx.load(input_path)
    try:
        passes = ["eliminate_identity", "fuse_consecutive_transposes",
                  "extract_constant_to_initializer", "eliminate_unused_initializer"]
        m = onnxoptimizer.optimize(m, passes)
    except Exception as e:
        print(f"  Optimization skipped: {e}")
    onnx.save(m, output_path)
    print(f"  {os.path.basename(output_path)}: {os.path.getsize(output_path) / 1024**2:.0f} MB")
    return output_path

android_files = {}
for name, path in quantized.items():
    # FIX: Output files end with _android.onnx to match Android app expectation
    out = f"{ANDROID_DIR}/{name}_android.onnx"
    result = optimize_android(path, out)
    if result:
        android_files[name] = result

# ══════════════════════════════════════════════════════════════
# CELL 9: Copy tokenizer files (CRITICAL FIX)
# ══════════════════════════════════════════════════════════════
print("\n=== Copying tokenizer files ===")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Save vocab.json and merges.txt for Android
tokenizer.save_pretrained(f"{ANDROID_DIR}/tokenizer_tmp")

import shutil
import json

# Copy vocab.json
vocab_src = f"{ANDROID_DIR}/tokenizer_tmp/vocab.json"
if os.path.exists(vocab_src):
    shutil.copy(vocab_src, f"{ANDROID_DIR}/vocab.json")
    vocab_size = len(json.load(open(vocab_src)))
    print(f"  vocab.json: {vocab_size} tokens")
else:
    # Try to extract from tokenizer.json
    tj = f"{ANDROID_DIR}/tokenizer_tmp/tokenizer.json"
    if os.path.exists(tj):
        data = json.load(open(tj))
        vocab = data.get("model", {}).get("vocab", {})
        with open(f"{ANDROID_DIR}/vocab.json", "w") as f:
            json.dump(vocab, f)
        print(f"  vocab.json extracted: {len(vocab)} tokens")

# Copy merges.txt
merges_src = f"{ANDROID_DIR}/tokenizer_tmp/merges.txt"
if os.path.exists(merges_src):
    shutil.copy(merges_src, f"{ANDROID_DIR}/merges.txt")
    with open(merges_src) as f:
        n = sum(1 for line in f if line.strip() and not line.startswith('#'))
    print(f"  merges.txt: {n} rules")
else:
    # Extract from tokenizer.json
    tj = f"{ANDROID_DIR}/tokenizer_tmp/tokenizer.json"
    if os.path.exists(tj):
        data = json.load(open(tj))
        merges = data.get("model", {}).get("merges", [])
        with open(f"{ANDROID_DIR}/merges.txt", "w") as f:
            f.write("#version: 0.2\n")
            for m in merges:
                f.write(m + "\n")
        print(f"  merges.txt extracted: {len(merges)} rules")

shutil.rmtree(f"{ANDROID_DIR}/tokenizer_tmp", ignore_errors=True)

# ══════════════════════════════════════════════════════════════
# CELL 9.5: Generate model.json manifest (CRITICAL for the new app)
# ══════════════════════════════════════════════════════════════
# The app reads languages/voices, file roles, sample rate and the decode
# constants (audioTokenStart / eosTokenId) from this manifest. Without it the
# app falls back to compile-time defaults, which may not match this export.
print("\n=== Generating model.json manifest ===")

vocab_path = f"{ANDROID_DIR}/vocab.json"
text_vocab_size = len(json.load(open(vocab_path))) if os.path.exists(vocab_path) else 151936

# Audio codec tokens begin right after the text vocabulary in the talker logits.
audio_token_start = text_vocab_size
talker_dim = int(globals().get("TALKER_OUTPUT_DIM", 0))

try:
    eos_token_id = int(getattr(tokenizer, "eos_token_id", None) or 151645)
except Exception:
    eos_token_id = 151645

print(f"  talker logits dim : {talker_dim}")
print(f"  text vocab size   : {text_vocab_size}")
print(f"  audioTokenStart   : {audio_token_start}")
print(f"  eosTokenId        : {eos_token_id}")
if talker_dim and audio_token_start >= talker_dim:
    print("  WARNING: audioTokenStart >= talker logits width — verify the vocab "
          "layout and adjust 'audioTokenStart' in model.json before shipping.")
else:
    print(f"  => {max(talker_dim - audio_token_start, 0)} audio codec tokens above the offset")

_file_roles = [
    ("talker_base", "talker_base_android.onnx", "TALKER", True),
    ("code_predictor_base", "code_predictor_base_android.onnx", "CODE_PREDICTOR", False),
    ("code2wav", "code2wav_android.onnx", "VOCODER", True),
    ("speaker_encoder", "speaker_encoder_android.onnx", "SPEAKER_ENCODER", False),
]
files_meta = []
for key, fname, role, required in _file_roles:
    if key in android_files:
        size_mb = round(os.path.getsize(android_files[key]) / 1024**2)
        files_meta.append({"filename": fname, "role": role, "sizeMb": size_mb, "required": required})

manifest = {
    "schemaVersion": 1,
    "id": "qwen3-tts-12hz-0.6b",
    "displayName": "Qwen3 TTS 0.6B",
    "family": "qwen3-tts",
    "sampleRateHz": 24000,
    "codecFrameRateHz": 12,
    "audioTokenStart": audio_token_start,
    "eosTokenId": eos_token_id,
    "tokenizer": {"type": "qwen-bpe", "files": ["vocab.json", "merges.txt"]},
    "files": files_meta,
    "voices": [
        {"id": "qwen3_ru", "locale": "ru-RU", "displayName": "Русский", "speakerId": 0},
        {"id": "qwen3_en", "locale": "en-US", "displayName": "English", "speakerId": 1},
        {"id": "qwen3_zh", "locale": "zh-CN", "displayName": "中文", "speakerId": 2},
    ],
}
manifest_path = f"{ANDROID_DIR}/model.json"
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
print(f"  Wrote {manifest_path}")

# ══════════════════════════════════════════════════════════════
# CELL 10: Create ZIP archive (FIXED naming)
# ══════════════════════════════════════════════════════════════
print("\n=== Creating archive ===")

total = 0
for name, path in android_files.items():
    size = os.path.getsize(path) / 1024**2
    total += size
    print(f"  {os.path.basename(path)}: {size:.0f} MB")
print(f"  Total: {total:.0f} MB")

zip_path = f"{WORK_DIR}/qwen3-tts-android.zip"
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    # FIX: Archive uses _android.onnx names matching Android app expectations
    for name, path in android_files.items():
        archive_name = f"models/{name}_android.onnx"
        zf.write(path, archive_name)
        print(f"  Added: {archive_name}")

    # FIX: Include tokenizer files
    vocab_file = f"{ANDROID_DIR}/vocab.json"
    merges_file = f"{ANDROID_DIR}/merges.txt"
    if os.path.exists(vocab_file):
        zf.write(vocab_file, "models/vocab.json")
        print("  Added: models/vocab.json")
    if os.path.exists(merges_file):
        zf.write(merges_file, "models/merges.txt")
        print("  Added: models/merges.txt")

    # FIX: Include the capability manifest so the app auto-detects languages/voices
    manifest_file = f"{ANDROID_DIR}/model.json"
    if os.path.exists(manifest_file):
        zf.write(manifest_file, "models/model.json")
        print("  Added: models/model.json")

print(f"\nArchive: {os.path.getsize(zip_path) / 1024**2:.0f} MB")

# Download (Colab only)
try:
    from google.colab import files
    files.download(zip_path)
except ImportError:
    print(f"Archive saved to: {zip_path}")
