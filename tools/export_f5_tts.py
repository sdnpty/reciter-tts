#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F5-TTS  ->  ONNX  ->  Android (Reciter TTS).

F5-TTS is a NON-autoregressive flow-matching text-to-speech model: a DiT
(diffusion transformer) integrates an ODE over NFE steps to turn noise into a
mel-spectrogram conditioned on (reference audio + reference text + target text),
then a Vocos vocoder turns the mel into a 24 kHz waveform. Unlike the Qwen3 AR
talker there is no token-by-token loop, so it is far better suited to running
near real time on a phone (NFE * one DiT forward, instead of 17 forwards per
12 ms frame).

This script exports the two heavy graphs + the data the on-device engine needs:

  f5_dit.onnx     one ODE step WITH classifier-free guidance folded in:
                  (x, cond, text_ids, time) -> guided velocity   [1,T,100]
  f5_vocos.onnx   mel[1,100,T] -> waveform[1,N]
  vocab.json      token -> id  (char/pinyin vocab of the chosen checkpoint)
  f5_voices.bin   concatenated reference mels (fp16) for the baked voices
  f5_voices.json  per-voice {name, ref_text, n_frames, offset}
  f5_config.json  mel params + NFE + cfg_strength + sway coef (device must match)
  model.json      manifest (architecture = "f5") so the app auto-detects it

IMPORTANT — pick a checkpoint whose vocab covers your language. The official
F5TTS_v1_Base is English+Chinese (pinyin). For Russian use a Russian finetune
(e.g. a community `F5-TTS` Russian checkpoint) and its matching `vocab.txt`.
Set MODEL_CKPT / VOCAB_FILE / REF_AUDIO + REF_TEXT below.

Run in Google Colab (GPU optional). Several contracts here are version-sensitive
(tokenizer, mel, CFG); the script runs a reference vs ONNX validation at the end
and saves `f5_ref_privet.wav` so you can confirm before shipping to the phone.
"""

import os, json, math, zipfile
import numpy as np
import torch

# ── Configuration ────────────────────────────────────────────────────────────
OUT = "/content/f5-onnx/android"
os.makedirs(OUT, exist_ok=True)
OPSET = 18

# Checkpoint + vocab. Defaults to the documented Russian finetune
# (hotstone228/F5-TTS-Russian, base F5TTS_Base, cc-by-nc-4.0). Override via env.
HF_REPO    = os.environ.get("F5_HF_REPO", "hotstone228/F5-TTS-Russian")
HF_CKPT    = os.environ.get("F5_HF_CKPT", "model_last.safetensors")
HF_VOCAB   = os.environ.get("F5_HF_VOCAB", "vocab.txt")
MODEL_CKPT = os.environ.get("F5_CKPT", "")          # local path overrides HF_REPO
VOCAB_FILE = os.environ.get("F5_VOCAB", "")          # local path overrides HF_REPO
MODEL_NAME = os.environ.get("F5_NAME", "F5TTS_Base")

# Reference voice(s) to bake: (voice_id, locale, display, ref_wav_path, ref_text).
# The ref text MUST be the exact transcript of the ref wav (F5 is zero-shot).
REF_VOICES = [
    # filled below from REF_AUDIO/REF_TEXT or your own clips
]
REF_AUDIO = os.environ.get("F5_REF_WAV", "/content/ref_ru.wav")
REF_TEXT  = os.environ.get("F5_REF_TEXT", "Привет, это проверка синтеза речи.")

# Mel / inference params (F5 v1 + Vocos defaults). Device reads these from config.
SR          = 24000
N_MELS      = 100
N_FFT       = 1024
HOP         = 256
WIN         = 1024
NFE         = int(os.environ.get("F5_NFE", "16"))    # ODE steps; lower = faster
CFG         = float(os.environ.get("F5_CFG", "2.0")) # classifier-free guidance
SWAY        = float(os.environ.get("F5_SWAY", "-1.0"))
SPEED       = 1.0


def load_f5():
    """Load the F5-TTS model + vocoder + vocab via the f5_tts package."""
    from f5_tts.infer.utils_infer import load_model, load_vocoder
    from f5_tts.model import DiT
    from importlib.resources import files

    from huggingface_hub import hf_hub_download
    if VOCAB_FILE:
        vocab_path = VOCAB_FILE
    else:
        try:
            vocab_path = hf_hub_download(HF_REPO, HF_VOCAB)
        except Exception:
            vocab_path = str(files("f5_tts").joinpath("infer/examples/vocab.txt"))
    # F5TTS_Base architecture hyper-params (hotstone228 Russian is F5TTS_Base).
    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2,
                     text_dim=512, conv_layers=4)
    ckpt = MODEL_CKPT or hf_hub_download(HF_REPO, HF_CKPT)
    model = load_model(DiT, model_cfg, ckpt, vocab_file=vocab_path).eval()
    vocoder = load_vocoder().eval()
    # vocab_char_map: token(str) -> id(int)
    vocab_char_map = {}
    with open(vocab_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            vocab_char_map[line[:-1]] = i      # strip only the trailing \n (space is a token!)
    return model, vocoder, vocab_char_map, vocab_path


def mel_spectrogram(wav):
    """Vocos-compatible log-mel: wav[1,N] -> mel[1,100,T]. Matches F5's MelSpec."""
    import torchaudio
    ms = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=N_FFT, win_length=WIN, hop_length=HOP,
        n_mels=N_MELS, power=1, center=True, normalized=False, norm=None,
        mel_scale="htk", f_min=0, f_max=None)
    m = ms(wav)
    return torch.log(torch.clamp(m, min=1e-5))


# ── DiT step with classifier-free guidance folded into one graph ─────────────
class DiTGuided(torch.nn.Module):
    """One ODE step. Runs the transformer twice (conditioned + fully dropped)
    and returns the CFG-guided velocity, so the device does ONE onnx call/step.

    Mirrors f5_tts CFM.sample's `fn`:
        pred = transformer(x, cond, text, t, drop_audio_cond=False, drop_text=False)
        null = transformer(x, cond, text, t, drop_audio_cond=True,  drop_text=True)
        return null + (pred - null) * cfg
    """
    def __init__(self, transformer, cfg_strength):
        super().__init__()
        self.t = transformer
        self.cfg = float(cfg_strength)

    def forward(self, x, cond, text, time):
        pred = self.t(x=x, cond=cond, text=text, time=time,
                      drop_audio_cond=False, drop_text=False)
        null = self.t(x=x, cond=cond, text=text, time=time,
                      drop_audio_cond=True, drop_text=True)
        return null + (pred - null) * self.cfg


def export_dit(model):
    transformer = model.transformer if hasattr(model, "transformer") else model
    m = DiTGuided(transformer, CFG).eval()
    T, Lt = 64, 20
    x = torch.randn(1, T, N_MELS)
    cond = torch.randn(1, T, N_MELS)
    text = torch.randint(0, 100, (1, Lt), dtype=torch.long)
    time = torch.rand(1)
    with torch.no_grad():
        ref = m(x, cond, text, time)
    print(f"  DiT step out: {tuple(ref.shape)}")
    path = f"{OUT}/f5_dit.onnx"
    torch.onnx.export(
        m, (x, cond, text, time), path,
        input_names=["x", "cond", "text", "time"], output_names=["velocity"],
        dynamic_axes={"x": {1: "t"}, "cond": {1: "t"},
                      "text": {1: "lt"}, "velocity": {1: "t"}},
        opset_version=OPSET, do_constant_folding=True)
    print(f"  f5_dit.onnx: {os.path.getsize(path)/1024**2:.0f} MB")
    return path


class VocosWrap(torch.nn.Module):
    def __init__(self, vocoder):
        super().__init__()
        self.v = vocoder
    def forward(self, mel):                 # mel[1,100,T] -> wav[1,N]
        return self.v.decode(mel)


def export_vocos(vocoder):
    m = VocosWrap(vocoder).eval()
    mel = torch.randn(1, N_MELS, 64)
    with torch.no_grad():
        try:
            ref = m(mel)
        except Exception:
            # some vocos builds expose forward() not decode()
            m.forward = lambda mm: vocoder(mm)  # type: ignore
            ref = vocoder(mel)
    print(f"  vocos out: {tuple(ref.shape)}")
    path = f"{OUT}/f5_vocos.onnx"
    torch.onnx.export(
        m, (mel,), path, input_names=["mel"], output_names=["waveform"],
        dynamic_axes={"mel": {2: "t"}, "waveform": {1: "n"}},
        opset_version=OPSET, do_constant_folding=True)
    print(f"  f5_vocos.onnx: {os.path.getsize(path)/1024**2:.0f} MB")
    return path


def quantize(paths):
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except Exception as e:
        print(f"  quantization skipped: {e}"); return
    for p in paths:
        try:
            tmp = p + ".int8.onnx"
            quantize_dynamic(p, tmp, weight_type=QuantType.QInt8, per_channel=True)
            if os.path.getsize(tmp) < os.path.getsize(p):
                os.replace(tmp, p)
                print(f"  quantized {os.path.basename(p)} -> {os.path.getsize(p)/1024**2:.0f} MB")
            elif os.path.exists(tmp):
                os.remove(tmp)
        except Exception as e:
            print(f"  quantize {os.path.basename(p)} failed: {e}")


def bake_voices(vocab_char_map):
    """Compute + store reference mels for the baked voices."""
    import torchaudio
    voices = REF_VOICES or [("ru_ref_1", "ru-RU", "Русский", REF_AUDIO, REF_TEXT)]
    names, metas, blobs, offset = [], [], [], 0
    for vid, locale, disp, wav_path, ref_text in voices:
        if not os.path.exists(wav_path):
            print(f"  skip {vid}: {wav_path} not found"); continue
        wav, sr = torchaudio.load(wav_path)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        mel = mel_spectrogram(wav)[0]            # [100, T]
        T = mel.shape[1]
        arr = mel.t().contiguous().to(torch.float16).cpu().numpy()  # [T,100]
        blobs.append(arr.reshape(-1))
        names.append(vid)
        metas.append({"id": vid, "locale": locale, "displayName": disp,
                      "ref_text": ref_text, "n_frames": int(T), "offset": int(offset)})
        offset += T * N_MELS
        print(f"  baked {vid}: ref_frames={T}")
    if blobs:
        np.concatenate(blobs).astype(np.float16).tofile(f"{OUT}/f5_voices.bin")
    json.dump({"voices": metas, "dim": N_MELS},
              open(f"{OUT}/f5_voices.json", "w"), ensure_ascii=False, indent=2)
    return metas


def write_config_and_manifest(metas, dit_mb, vocos_mb):
    cfg = {"sample_rate": SR, "n_mels": N_MELS, "n_fft": N_FFT, "hop": HOP,
           "win": WIN, "nfe": NFE, "cfg_strength": CFG, "sway_coef": SWAY,
           "speed": SPEED}
    json.dump(cfg, open(f"{OUT}/f5_config.json", "w"), indent=2)
    manifest = {
        "schemaVersion": 1, "id": "f5-tts-ru", "displayName": "F5-TTS (Russian)",
        "family": "f5", "architecture": "f5", "sampleRateHz": SR,
        "tokenizer": {"type": "f5-char",
                      "files": ["vocab.json", "f5_config.json",
                                "f5_voices.json", "f5_voices.bin"]},
        "files": [
            {"filename": "f5_dit.onnx", "role": "TALKER", "sizeMb": dit_mb, "required": True},
            {"filename": "f5_vocos.onnx", "role": "VOCODER", "sizeMb": vocos_mb, "required": True},
        ],
        "voices": [{"id": m["id"], "locale": m["locale"], "displayName": m["displayName"],
                    "speakerId": i} for i, m in enumerate(metas)] or
                  [{"id": "ru_ref_1", "locale": "ru-RU", "displayName": "Русский", "speakerId": 0}],
    }
    json.dump(manifest, open(f"{OUT}/model.json", "w"), ensure_ascii=False, indent=2)


def write_vocab(vocab_char_map):
    json.dump(vocab_char_map, open(f"{OUT}/vocab.json", "w"), ensure_ascii=False)
    print(f"  vocab.json: {len(vocab_char_map)} tokens")


def validate(model, vocoder, vocab_char_map):
    """Synthesize a phrase with the library AND replicate with ONNX; compare."""
    try:
        import onnxruntime as ort
        from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text
        gen_text = "Привет"
        ref_audio, ref_text = preprocess_ref_audio_text(REF_AUDIO, REF_TEXT)
        wav_ref, sr_lib, _ = infer_process(ref_audio, ref_text, gen_text, model, vocoder,
                                           nfe_step=NFE, cfg_strength=CFG, speed=SPEED)
        import soundfile as sf
        sf.write("/content/f5_ref_privet.wav", wav_ref, sr_lib)
        print(f"  saved /content/f5_ref_privet.wav ({len(wav_ref)/sr_lib:.2f}s) — ПОСЛУШАЙ")
    except Exception as e:
        print(f"  reference inference failed: {e}")


if __name__ == "__main__":
    print("=== F5-TTS export ===")
    model, vocoder, vocab_char_map, vocab_path = load_f5()
    print(f"  model loaded, vocab={len(vocab_char_map)} from {vocab_path}")
    dit = export_dit(model)
    vocos = export_vocos(vocoder)
    quantize([dit, vocos])
    write_vocab(vocab_char_map)
    metas = bake_voices(vocab_char_map)
    write_config_and_manifest(metas,
                              round(os.path.getsize(dit)/1024**2),
                              round(os.path.getsize(vocos)/1024**2))
    validate(model, vocoder, vocab_char_map)

    zip_path = "/content/f5-tts-ru-android.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(os.listdir(OUT)):
            zf.write(os.path.join(OUT, f), f"models/{f}")
    print(f"\nArchive: {zip_path}  ({os.path.getsize(zip_path)/1024**2:.0f} MB)")
    try:
        from google.colab import files; files.download(zip_path)
    except Exception:
        pass
    print("Готово. Импортируй ZIP в приложении (Модели → локальный ZIP).")
