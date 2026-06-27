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


class TextCond(torch.nn.Module):
    """text ids -> text_projection(text_embedding(ids)). Matches how generate()
    builds trailing_text_hidden and the tts_bos/eos/pad embeds.

    The raw text_embedding (151936 x H) plus the projection blows past the 2 GiB
    protobuf limit during export. Since cond = projection(embedding(id)) is a
    pure lookup, we PRE-COMPUTE the combined table projection(embedding.weight)
    once and export it as a single Embedding — mathematically identical, and a
    single 151936 x H_out table stays under 2 GiB.
    """
    def __init__(self, talker):
        super().__init__()
        with torch.no_grad():
            w = talker.model.text_embedding.weight.float()        # [V, H_in]
            combined = talker.text_projection(w)                  # [V, H_out]
        self.table = torch.nn.Embedding.from_pretrained(combined.contiguous())
    def forward(self, text_ids):
        return self.table(text_ids)


class CodecEmbed(torch.nn.Module):
    """code0 (0..3071) -> hidden. The 3072-wide embedding the talker feeds back."""
    def __init__(self, talker):
        super().__init__()
        self.codec_embedding = talker.model.codec_embedding  # Embedding(3072, H)
    def forward(self, codes):
        return self.codec_embedding(codes)


def export_text_cond(model):
    talker = _talker(model)
    m = TextCond(talker).float().eval()
    dummy = torch.randint(0, talker.model.text_embedding.num_embeddings, (1, 16), dtype=torch.long)
    with torch.no_grad():
        out = m(dummy)
    path = f"{OUT}/text_cond.onnx"
    torch.onnx.export(
        m, (dummy,), path,
        input_names=["text_ids"], output_names=["cond"],
        dynamic_axes={"text_ids": {0: "b", 1: "t"}, "cond": {0: "b", 1: "t"}},
        opset_version=OPSET, do_constant_folding=True)
    print(f"  text_cond.onnx: {os.path.getsize(path)/1024**2:.1f} MB, cond dim = {tuple(out.shape)}")


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
    export_text_cond(model)     # noqa: F821  (provided by the Colab session)
    export_codec_embed(model)   # noqa: F821
    print("Done. Next: stage 2 (talker_step / subtalker_step with KV cache).")
