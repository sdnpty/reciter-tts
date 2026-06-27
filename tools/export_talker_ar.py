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


def _pos_ids(T):
    """3D RoPE position ids (3, 1, T) = arange broadcast. For a non-padded causal
    prefill this matches the model's get_rope_index output, but as an explicit
    input it keeps the exported graph dynamic over sequence length."""
    p = torch.arange(T, dtype=torch.long).view(1, 1, T)
    return p.expand(3, 1, T).contiguous()


class TalkerLogits(torch.nn.Module):
    """inputs_embeds[1,T,H] + position_ids[3,1,T] (+ attention_mask) -> logits[1,T,3072].

    Calls the backbone (talker.model) directly with explicit position_ids, so the
    custom get_rope_index isn't traced (which baked a fixed seq length). The
    Kotlin decoder re-runs this on a growing inputs_embeds and reads logits[:,-1]."""
    def __init__(self, talker):
        super().__init__()
        self.backbone = talker.model
        self.codec_head = talker.codec_head
    def forward(self, inputs_embeds, position_ids, attention_mask):
        out = self.backbone(input_ids=None, inputs_embeds=inputs_embeds,
                            position_ids=position_ids, attention_mask=attention_mask,
                            use_cache=False, return_dict=True)
        return self.codec_head(out.last_hidden_state)


def export_talker_logits(model):
    talker = _talker(model)
    m = TalkerLogits(talker).float().eval()
    T = 8
    emb = torch.randn(1, T, 1024); pos = _pos_ids(T); mask = torch.ones(1, T, dtype=torch.long)
    with torch.no_grad():
        ref = m(emb, pos, mask)
    print(f"  torch logits: {tuple(ref.shape)}")
    path = f"{OUT}/talker_logits.onnx"
    torch.onnx.export(
        m, (emb, pos, mask), path,
        input_names=["inputs_embeds", "position_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={"inputs_embeds": {0: "b", 1: "t"},
                      "position_ids": {1: "b", 2: "t"},
                      "attention_mask": {0: "b", 1: "t"},
                      "logits": {0: "b", 1: "t"}},
        opset_version=OPSET, do_constant_folding=True)
    print(f"  talker_logits.onnx: {os.path.getsize(path)/1024**2:.0f} MB")
    # validate at a DIFFERENT length to prove the graph is dynamic + faithful
    try:
        import onnxruntime as ort, numpy as np
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        T2 = 13
        emb2 = torch.randn(1, T2, 1024); pos2 = _pos_ids(T2); mask2 = torch.ones(1, T2, dtype=torch.long)
        with torch.no_grad():
            tref = m(emb2, pos2, mask2).numpy()
        oout = sess.run(None, {"inputs_embeds": emb2.numpy(),
                               "position_ids": pos2.numpy().astype(np.int64),
                               "attention_mask": mask2.numpy().astype(np.int64)})[0]
        print(f"  VALIDATION (T={T2}) max|onnx-torch| = {np.abs(oout - tref).max():.3e} "
              f"(should be < 1e-3)")
    except Exception as e:
        print(f"  validation FAILED: {e}")


if __name__ == "__main__":
    print("=== Stage 1: text_cond + codec_embed ===")
    export_text_cond_table(model)   # noqa: F821  (provided by the Colab session)
    export_codec_embed(model)       # noqa: F821
    print("\n=== Stage 2a: talker_logits (prefill-mode, reuses RoPE) ===")
    export_talker_logits(model)     # noqa: F821
    print("\nDone. Next: subtalker (code_predictor) export + Kotlin AR loop.")
