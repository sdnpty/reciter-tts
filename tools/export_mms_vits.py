#!/usr/bin/env python3
"""
Export a VITS TTS model (facebook/mms-tts-rus or compatible) → ONNX → Android.

Produces a ZIP with:
  models/vits_rus_android.onnx   (input_ids, attention_mask → waveform)
  models/vocab.json              (char → id, for the on-device tokenizer)
  models/model.json              (architecture = "vits")

Import it in the app (Models → Pick Local ZIP). The app's VITS engine
(architecture "vits") runs it. NOTE: MMS uses a char vocab; if the tokenizer
reports is_uroman=True the language needs romanization that the on-device
tokenizer does not do yet — pick a model whose vocab already covers the script.

Run in Google Colab (GPU optional — VITS is small).
"""

import os
import json
import zipfile

# ══════════════════════════════════════════════════════════════
# CELL 1: Install dependencies
# ══════════════════════════════════════════════════════════════
# !pip install -q torch transformers onnx onnxruntime numpy

import torch
from transformers import VitsModel, VitsTokenizer

MODEL_NAME = "facebook/mms-tts-rus"
WORK_DIR = "/content/vits-onnx"
os.makedirs(WORK_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# CELL 2: Load model + tokenizer
# ══════════════════════════════════════════════════════════════
print(f"Loading {MODEL_NAME} …")
model = VitsModel.from_pretrained(MODEL_NAME).eval()
tokenizer = VitsTokenizer.from_pretrained(MODEL_NAME)

sample_rate = int(getattr(model.config, "sampling_rate", 16000))
print(f"  sample_rate = {sample_rate}")
print(f"  is_uroman   = {getattr(tokenizer, 'is_uroman', None)} "
      f"(True ⇒ needs romanization, not supported on-device)")
print(f"  vocab size  = {tokenizer.vocab_size}")

# ══════════════════════════════════════════════════════════════
# CELL 3: Export to ONNX (input_ids, attention_mask → waveform)
# ══════════════════════════════════════════════════════════════
class VitsWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        return self.m(input_ids=input_ids, attention_mask=attention_mask).waveform


wrapper = VitsWrapper(model).eval()

sample = tokenizer("привет, это проверка синтеза речи", return_tensors="pt")
dummy_ids = sample["input_ids"]
dummy_mask = sample.get("attention_mask", torch.ones_like(dummy_ids))

onnx_path = f"{WORK_DIR}/vits_rus_android.onnx"
with torch.no_grad():
    wav = wrapper(dummy_ids, dummy_mask)
    print(f"  Forward OK: waveform {tuple(wav.shape)}")
    torch.onnx.export(
        wrapper, (dummy_ids, dummy_mask), onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["waveform"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "waveform": {0: "batch", 1: "samples"},
        },
        opset_version=17, do_constant_folding=True,
    )
print(f"  ONNX: {os.path.getsize(onnx_path) / 1024**2:.0f} MB")

# ══════════════════════════════════════════════════════════════
# CELL 4: INT8 quantization (optional, smaller download)
# ══════════════════════════════════════════════════════════════
try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    q_path = f"{WORK_DIR}/vits_rus_android_int8.onnx"
    quantize_dynamic(onnx_path, q_path, weight_type=QuantType.QInt8)
    if os.path.getsize(q_path) < os.path.getsize(onnx_path):
        os.replace(q_path, onnx_path)
        print(f"  Quantized: {os.path.getsize(onnx_path) / 1024**2:.0f} MB")
except Exception as e:
    print(f"  Quantization skipped: {e}")

# ══════════════════════════════════════════════════════════════
# CELL 5: Write vocab.json (char → id) for the on-device tokenizer
# ══════════════════════════════════════════════════════════════
vocab = tokenizer.get_vocab()  # token(str) → id(int)
with open(f"{WORK_DIR}/vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False)
print(f"  vocab.json: {len(vocab)} symbols (sample: {list(vocab)[:12]})")

# ══════════════════════════════════════════════════════════════
# CELL 6: Generate model.json manifest (architecture = vits)
# ══════════════════════════════════════════════════════════════
manifest = {
    "schemaVersion": 1,
    "id": "mms-tts-rus",
    "displayName": "MMS-TTS Russian (VITS)",
    "family": "vits",
    "architecture": "vits",
    "sampleRateHz": sample_rate,
    "tokenizer": {"type": "vits-char", "files": ["vocab.json"]},
    "files": [
        {"filename": "vits_rus_android.onnx", "role": "VOCODER", "sizeMb": round(os.path.getsize(onnx_path) / 1024**2), "required": True}
    ],
    "voices": [
        {"id": "mms_ru", "locale": "ru-RU", "displayName": "Русский (MMS)", "speakerId": 0}
    ],
}
with open(f"{WORK_DIR}/model.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
print("  Wrote model.json")

# ══════════════════════════════════════════════════════════════
# CELL 7: Create ZIP + download
# ══════════════════════════════════════════════════════════════
zip_path = f"{WORK_DIR}/mms-tts-rus-android.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(onnx_path, "models/vits_rus_android.onnx")
    zf.write(f"{WORK_DIR}/vocab.json", "models/vocab.json")
    zf.write(f"{WORK_DIR}/model.json", "models/model.json")
print(f"Archive: {os.path.getsize(zip_path) / 1024**2:.0f} MB")

try:
    from google.colab import files
    files.download(zip_path)
except ImportError:
    print(f"Archive saved to: {zip_path}")
