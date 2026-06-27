#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reconstruct the talker prefill for the x-vector voice path, purely from
text_cond_table.f16 + codec_embed.onnx + a baked speaker x-vector + input_ids.
Validates against the prefill captured from model.generate (xvec_capture.npz).

This is the last contract to port: once it matches, Kotlin can build the prefill
from a tokenized prompt + a baked voice vector, with no PyTorch at runtime.

Recipe (x-vector, non-ICL, streaming), from Qwen3TTSModel.generate:
  role   = text_cond(input_ids[:3])                       # 3
  codec0 = codec_embed([think, think_bos, lang, think_eos])   # 4
  codec1 = codec_embed([codec_pad, codec_bos])                # 2
  codec_input = [codec0, xvector, codec1]                 # 7
  tie    = [tts_pad*5, tts_bos] + codec_input[:6]         # 6
  prefill = [role, tie, text_cond(input_ids[3]) + codec_input[6]]   # 3+6+1 = 10
  trailing_text_hidden = [text_cond(input_ids[4:-5]), tts_eos]
"""
import os
import numpy as np
import onnxruntime as ort

D = os.environ.get("ONNX_DIR", "/content/qwen3-tts-onnx/android")
CAP = os.environ.get("CAP", "/content/xvec_capture.npz")
H = 1024
TTS_BOS, TTS_EOS, TTS_PAD = 151672, 151673, 151671
CODEC_PAD, CODEC_BOS = 2148, 2149
THINK, THINK_BOS, THINK_EOS = 2154, 2156, 2157
LANG_RU = 2069

cap = np.load(CAP)
input_ids = cap["input_ids"]
input_ids = input_ids[0] if input_ids.ndim == 2 else input_ids        # (L,)
xvec = cap["xvector"].reshape(-1).astype(np.float32)                  # (H,)
ref_prefill = cap["inputs_embeds"][0].astype(np.float32)             # (10,H)
ref_trailing = cap["trailing_text_hidden"][0].astype(np.float32)    # (11,H)

tct = np.fromfile(f"{D}/text_cond_table.f16", np.float16).reshape(151936, H).astype(np.float32)
cemb = ort.InferenceSession(f"{D}/codec_embed.onnx", providers=["CPUExecutionProvider"])


def codec_embed(ids):
    return cemb.run(None, {"codes": np.array([ids], np.int64)})[0][0]   # (len,H)


def text_cond(ids):
    return tct[np.asarray(ids, dtype=np.int64)]                        # (len,H)


def build(input_ids, xvec, lang_id=LANG_RU):
    tts_bos = text_cond([TTS_BOS])[0]
    tts_eos = text_cond([TTS_EOS])[0]
    tts_pad = text_cond([TTS_PAD])[0]
    role = text_cond(input_ids[:3])                                   # (3,H)
    codec0 = codec_embed([THINK, THINK_BOS, lang_id, THINK_EOS])       # (4,H)
    codec1 = codec_embed([CODEC_PAD, CODEC_BOS])                       # (2,H)
    codec_input = np.concatenate([codec0, xvec[None], codec1], 0)      # (7,H)
    left = np.stack([tts_pad] * 5 + [tts_bos])                         # (6,H)
    tie = left + codec_input[:6]
    last = text_cond([input_ids[3]])[0] + codec_input[6]
    prefill = np.concatenate([role, tie, last[None]], 0)              # (10,H)
    trailing = np.concatenate([text_cond(input_ids[4:-5]), tts_eos[None]], 0)
    return prefill, trailing


if __name__ == "__main__":
    prefill, trailing = build(input_ids, xvec)
    print(f"prefill  built {prefill.shape}  ref {ref_prefill.shape}  "
          f"max|diff| = {np.abs(prefill - ref_prefill).max():.4e}")
    print(f"trailing built {trailing.shape}  ref {ref_trailing.shape}  "
          f"max|diff| = {np.abs(trailing - ref_trailing).max():.4e}")
    print("(diffs ~1e-3 are fine — text_cond_table is fp16)")
