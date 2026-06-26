#!/usr/bin/env python3
"""
CosyVoice 3 (FunAudioLLM/Fun-CosyVoice3-0.5B-2512) → ONNX → Android.

STATUS: SCAFFOLD. CosyVoice is a flow-matching family (LLM → speech tokens →
conditional flow-matching → mel → HiFiGAN) and also needs a speaker encoder +
speech tokenizer over a reference clip for zero-shot. The official CosyVoice
repo exports the flow/vocoder to ONNX/TRT but keeps the autoregressive LLM in
PyTorch — a full on-device ONNX pipeline (esp. the AR LLM) is an open task.

This script sets up the environment, downloads the model, runs the official
export tooling, then assembles a `model.json` (architecture "cosyvoice") from
whatever ONNX components were produced. Import the ZIP in the app: the
CosyVoice *introspection* engine will load each component and log its ONNX
input/output signatures — share that log so the runtime can be wired.

Run in Google Colab (GPU).
"""

import os
import re
import json
import glob
import zipfile

# ══════════════════════════════════════════════════════════════
# CELL 1: Install dependencies + clone CosyVoice
# ══════════════════════════════════════════════════════════════
# !pip install -q modelscope onnx onnxruntime soundfile hyperpyyaml
# !git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git /content/CosyVoice
# !pip install -q -r /content/CosyVoice/requirements.txt

import sys
sys.path.append("/content/CosyVoice")
sys.path.append("/content/CosyVoice/third_party/Matcha-TTS")

WORK_DIR = "/content/cosyvoice-onnx"
MODEL_DIR = "/content/Fun-CosyVoice3-0.5B-2512"
os.makedirs(WORK_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# CELL 2: Download the model
# ══════════════════════════════════════════════════════════════
from modelscope import snapshot_download
try:
    snapshot_download("FunAudioLLM/Fun-CosyVoice3-0.5B-2512", local_dir=MODEL_DIR)
except Exception as e:
    print("modelscope failed, try HF:", e)
    # !pip install -q huggingface_hub
    from huggingface_hub import snapshot_download as hf_dl
    hf_dl(repo_id="FunAudioLLM/Fun-CosyVoice3-0.5B-2512", local_dir=MODEL_DIR)
print("Model files:", os.listdir(MODEL_DIR))

# ══════════════════════════════════════════════════════════════
# CELL 3: Export to ONNX via the official tooling
# ══════════════════════════════════════════════════════════════
# The repo ships exporters; the flow estimator is the main ONNX target.
# !python /content/CosyVoice/cosyvoice/bin/export_onnx.py --model_dir {MODEL_DIR} || true
#
# Any pre-shipped *.onnx in the model dir (e.g. speech_tokenizer, campplus) are
# reused as-is. Collect everything we found into WORK_DIR.
import shutil
found = glob.glob(f"{MODEL_DIR}/**/*.onnx", recursive=True)
print("ONNX components found:")
for p in found:
    dst = os.path.join(WORK_DIR, os.path.basename(p))
    shutil.copy(p, dst)
    print(f"  {os.path.basename(p)}: {os.path.getsize(p)/1024**2:.0f} MB")

# ══════════════════════════════════════════════════════════════
# CELL 4: Assemble model.json (architecture = cosyvoice)
# ══════════════════════════════════════════════════════════════
def guess_role(name: str) -> str:
    n = name.lower()
    if "llm" in n:                       return "LLM"
    if "flow" in n or "estimator" in n:  return "FLOW"
    if "hift" in n or "vocod" in n or "generator" in n: return "VOCODER"
    if "campplus" in n or "spk" in n or "speaker" in n: return "SPEAKER_ENCODER"
    if "speech_tokenizer" in n or "s3" in n or "tokenizer" in n: return "SPEECH_TOKENIZER"
    return "FLOW"

files = []
for p in sorted(glob.glob(f"{WORK_DIR}/*.onnx")):
    base = os.path.basename(p)
    files.append({
        "filename": base,
        "role": guess_role(base),
        "sizeMb": round(os.path.getsize(p) / 1024**2),
        "required": guess_role(base) in ("LLM", "FLOW", "VOCODER"),
    })

manifest = {
    "schemaVersion": 1,
    "id": "fun-cosyvoice3-0.5b",
    "displayName": "Fun-CosyVoice3 0.5B",
    "family": "cosyvoice",
    "architecture": "cosyvoice",
    "sampleRateHz": 24000,
    "files": files,
    "voices": [
        {"id": "cosy_ru", "locale": "ru-RU", "displayName": "Русский (CosyVoice3)", "speakerId": 0}
    ],
    "notes": "Flow-matching; zero-shot needs reference audio. Runtime WIP — see docs/MODELS.md",
}
with open(f"{WORK_DIR}/model.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
print("Wrote model.json with roles:", [(x["filename"], x["role"]) for x in files])

# ══════════════════════════════════════════════════════════════
# CELL 5: Create ZIP + download (for the introspection engine)
# ══════════════════════════════════════════════════════════════
zip_path = f"{WORK_DIR}/fun-cosyvoice3-android.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in glob.glob(f"{WORK_DIR}/*.onnx"):
        zf.write(p, f"models/{os.path.basename(p)}")
    zf.write(f"{WORK_DIR}/model.json", "models/model.json")
    # If the repo provides a tokenizer vocab, include it as vocab.json.
    for cand in glob.glob(f"{MODEL_DIR}/**/vocab.json", recursive=True)[:1]:
        zf.write(cand, "models/vocab.json")
print(f"Archive: {os.path.getsize(zip_path)/1024**2:.0f} MB")
try:
    from google.colab import files
    files.download(zip_path)
except ImportError:
    print(f"Archive saved to: {zip_path}")
