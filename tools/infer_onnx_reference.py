#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
End-to-end ONNX reference decoder for Qwen3-TTS (path A).

Runs prefill + the autoregressive loop using ONLY the exported ONNX files and
raw tables, then compares the produced 16-codebook frames to ground-truth codes
captured from model.generate. This is the validation gate before porting the
loop to Kotlin: if the codes match, the exported pieces + loop logic are correct.

Inputs (in ONNX_DIR, default /content/qwen3-tts-onnx/android):
  talker_step.onnx       (inputs_embeds[1,1,H], position_ids[3,1,1], cache_position[1],
                          past_k/v_{0..27})  -> logits[1,1,3072], hidden[1,1,H], present_k/v
  subtalker_step.onnx    (inputs_embeds[1,1,H], position_ids[1,1], cache_position[1],
                          past_k/v_{0..4})   -> hidden[1,1,H], present_k/v
  codec_embed.onnx       (codes[1,1]) -> emb[1,1,H]          (talker codec embedding, 3072)
  subtalker_codec_embed.f16  [15,2048,H]    subtalker_heads.f16 [15,2048,H]
  subtalker_heads_bias.f16   [15,2048] (optional)
Ground truth: /content/groundtruth.npz with inputs_embeds, trailing_text_hidden,
              tts_pad_embed, codes.
"""

import os
import numpy as np
import onnxruntime as ort

D = os.environ.get("ONNX_DIR", "/content/qwen3-tts-onnx/android")
GT = os.environ.get("GT", "/content/groundtruth.npz")

H = 1024
KVH, HD = 8, 128
NLT, NLS = 28, 5         # talker / subtalker layers
G = 15                   # residual codebooks
VOCAB = 3072
EOS = 2150
CODEBOOK0 = 2048         # codebook-0 range [0, 2048); 2048..3071 are special
REP_PENALTY = 1.05

gt = np.load(GT)
prefill = gt["inputs_embeds"].astype(np.float32)          # (1, P, H)
tth = gt["trailing_text_hidden"].astype(np.float32)        # (1, Ttext, H)
pad = gt["tts_pad_embed"].astype(np.float32)               # (1, 1, H)
ref_codes = gt["codes"]                                    # (T, 16)
P = prefill.shape[1]

talker = ort.InferenceSession(f"{D}/talker_step.onnx", providers=["CPUExecutionProvider"])
sub = ort.InferenceSession(f"{D}/subtalker_step.onnx", providers=["CPUExecutionProvider"])
cemb = ort.InferenceSession(f"{D}/codec_embed.onnx", providers=["CPUExecutionProvider"])
sce = np.fromfile(f"{D}/subtalker_codec_embed.f16", np.float16).reshape(G, 2048, H).astype(np.float32)
shd = np.fromfile(f"{D}/subtalker_heads.f16", np.float16).reshape(G, 2048, H).astype(np.float32)
_bias_path = f"{D}/subtalker_heads_bias.f16"
shb = (np.fromfile(_bias_path, np.float16).reshape(G, 2048).astype(np.float32)
       if os.path.exists(_bias_path) else np.zeros((G, 2048), np.float32))


def _kv(n):
    return [np.zeros((1, KVH, 0, HD), np.float32) for _ in range(2 * n)]


def codec_embed(code):
    return cemb.run(None, {"codes": np.array([[code]], np.int64)})[0]   # (1,1,H)


def talker_step(emb, pos, kv):
    feed = {"inputs_embeds": emb.astype(np.float32),
            "position_ids": np.full((3, 1, 1), pos, np.int64),
            "cache_position": np.array([pos], np.int64)}
    for i in range(NLT):
        feed[f"past_k_{i}"] = kv[2 * i]; feed[f"past_v_{i}"] = kv[2 * i + 1]
    out = talker.run(None, feed)            # logits, hidden, present...
    return out[0][0, -1], out[1], list(out[2:])   # logits(3072,), hidden(1,1,H), kv


def sub_step(emb, pos, kv):
    feed = {"inputs_embeds": emb.astype(np.float32),
            "position_ids": np.array([[pos]], np.int64),
            "cache_position": np.array([pos], np.int64)}
    for i in range(NLS):
        feed[f"past_k_{i}"] = kv[2 * i]; feed[f"past_v_{i}"] = kv[2 * i + 1]
    out = sub.run(None, feed)               # hidden, present...
    return out[0], list(out[1:])            # hidden(1,1,H), kv


def select_code0(logits, history):
    lg = logits.copy()
    for c in set(history):                              # repetition penalty
        lg[c] = lg[c] / REP_PENALTY if lg[c] > 0 else lg[c] * REP_PENALTY
    for i in range(CODEBOOK0, VOCAB):                   # suppress specials except EOS
        if i != EOS:
            lg[i] = -np.inf
    return int(np.argmax(lg))


def run_subtalker(past_hidden, last_id_hidden):
    """15 residual codes from past_hidden (prev talker hidden) + code0 embedding."""
    kv = _kv(NLS)
    _, kv = sub_step(past_hidden, 0, kv)                 # prefill pos 0
    h, kv = sub_step(last_id_hidden, 1, kv)              # prefill pos 1 -> head[0]
    codes = [int(np.argmax(h[0, 0] @ shd[0].T + shb[0]))]
    for g in range(1, G):
        emb = sce[g - 1, codes[-1]].reshape(1, 1, H)     # codec_embedding[g-1](prev code)
        h, kv = sub_step(emb, 1 + g, kv)
        codes.append(int(np.argmax(h[0, 0] @ shd[g].T + shb[g])))
    return codes                                        # 15 ints (codebooks 1..15)


def main():
    kv = _kv(NLT)
    logits = hidden = None
    for i in range(P):                                  # prefill, one position at a time
        logits, hidden, kv = talker_step(prefill[:, i:i + 1], i, kv)
    code0 = select_code0(logits, [])
    past_hidden = hidden
    pos = P
    step = 0
    frames = []
    history0 = []
    max_frames = ref_codes.shape[0] + 8
    while code0 != EOS and len(frames) < max_frames:
        last_id_hidden = codec_embed(code0)             # talker codec embed (1,1,H)
        residual = run_subtalker(past_hidden, last_id_hidden)
        frames.append([code0] + residual)
        history0.append(code0)
        # next talker input = sum of all 16 code embeds + text condition
        s = last_id_hidden.copy()
        for g in range(G):
            s = s + sce[g, residual[g]].reshape(1, 1, H)
        s = s + (tth[:, step:step + 1] if step < tth.shape[1] else pad)
        logits, hidden, kv = talker_step(s, pos, kv)
        code0 = select_code0(logits, history0)
        past_hidden = hidden
        pos += 1
        step += 1

    out = np.array(frames)
    print(f"produced {out.shape[0]} frames, ref {ref_codes.shape[0]}")
    n = min(out.shape[0], ref_codes.shape[0])
    if n:
        match = (out[:n] == ref_codes[:n]).mean()
        first_bad = next((i for i in range(n) if not np.array_equal(out[i], ref_codes[i])), None)
        print(f"frame-element match over first {n} frames: {match:.3%}")
        print(f"first mismatching frame: {first_bad}")
        if first_bad is not None:
            print("  ref :", ref_codes[first_bad])
            print("  ours:", out[first_bad])

    # close the chain: frames -> code2wav -> audio
    try:
        import soundfile as sf
        c2w = ort.InferenceSession(f"{D}/code2wav.onnx", providers=["CPUExecutionProvider"])
        codes_in = out.T[None].astype(np.int64)            # (1, 16, T)
        audio = c2w.run(None, {"codec_tokens": codes_in})[0].reshape(-1)
        sf.write("/content/onnx_reference.wav", audio, 24000)
        print(f"\ncode2wav -> /content/onnx_reference.wav  ({len(audio)/24000:.2f}s)")
        print("Compare it to /content/ref_audio.wav (should sound identical).")
    except Exception as e:
        print(f"code2wav step skipped: {e}")
    return out


if __name__ == "__main__":
    main()
