#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen3-TTS autoregressive talker export — PATH A, STAGE 1.

Background: the talker is an autoregressive decoder over the codec vocabulary
(see docs/TALKER_AR_PLAN.md). A correct on-device port needs several ONNX
building blocks. This script exports the two UNAMBIGUOUS ones first; the harder
KV-cache `talker_step` / `subtalker_step` exports follow in stage 2.

Stage 1 exports:
  text_cond.onnx   : text_ids[B,T] -> cond[B,T,H]   (= text_projection(text_embedding(ids)))
                     Use it to build trailing_text_hidden and the bos/eos/pad embeds.
  codec_embed.onnx : code0[B,T] (0..3071) -> emb[B,T,H]  (talker.model.codec_embedding)

Run in Colab after the model is loaded as `model` (Qwen3TTSModel.model).
"""

import os, torch

OUT = "/content/qwen3-tts-onnx/android"
os.makedirs(OUT, exist_ok=True)
OPSET = 18


def _talker(model):
    # `model` here is Qwen3TTSModel.model (the wrapper exposes .talker).
    return model.talker


import json, numpy as np


def export_text_cond_table(model):
    """text cond is a pure lookup: cond[id] = text_projection(text_embedding[id]).
    The combined table is far too big for an ONNX proto (>2 GiB protobuf limit),
    so we pre-compute it and save it as a raw fp16 binary + a small meta json.
    On device Kotlin gathers rows directly (and slices tts_bos/eos/pad by id)."""
    talker = _talker(model)
    with torch.no_grad():
        w = talker.model.text_embedding.weight.float()            # [V, H_in]
        combined = talker.text_projection(w).contiguous()         # [V, H_out]
    V, H = combined.shape
    arr = combined.to(torch.float16).cpu().numpy()
    bin_path = f"{OUT}/text_cond_table.f16"
    arr.tofile(bin_path)
    meta = {
        "rows": int(V), "dim": int(H), "dtype": "float16",
        "file": "text_cond_table.f16",
        "tts_bos_token_id": int(getattr(model.config, "tts_bos_token_id", -1)),
        "tts_eos_token_id": int(getattr(model.config, "tts_eos_token_id", -1)),
        "tts_pad_token_id": int(getattr(model.config, "tts_pad_token_id", -1)),
    }
    with open(f"{OUT}/text_cond_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  text_cond_table.f16: {os.path.getsize(bin_path)/1024**2:.1f} MB, "
          f"shape=({V},{H})  meta={meta}")


class CodecEmbed(torch.nn.Module):
    """code0 (0..3071) -> hidden. The 3072-wide embedding the talker feeds back."""
    def __init__(self, talker):
        super().__init__()
        self.codec_embedding = talker.model.codec_embedding  # Embedding(3072, H)
    def forward(self, codes):
        return self.codec_embedding(codes)


def export_codec_embed(model):
    talker = _talker(model)
    m = CodecEmbed(talker).float().eval()
    dummy = torch.randint(0, talker.model.codec_embedding.num_embeddings, (1, 4), dtype=torch.long)
    path = f"{OUT}/codec_embed.onnx"
    torch.onnx.export(
        m, (dummy,), path,
        input_names=["codes"], output_names=["emb"],
        dynamic_axes={"codes": {0: "b", 1: "t"}, "emb": {0: "b", 1: "t"}},
        opset_version=OPSET, do_constant_folding=True)
    print(f"  codec_embed.onnx: {os.path.getsize(path)/1024**2:.1f} MB "
          f"(num_embeddings={talker.model.codec_embedding.num_embeddings})")


if __name__ == "__main__":
    print("=== Stage 1: text_cond + codec_embed ===")
    export_text_cond_table(model)   # noqa: F821  (provided by the Colab session)
    export_codec_embed(model)       # noqa: F821
    print("Done. Next: stage 2 (talker_step / subtalker_step with KV cache).")
