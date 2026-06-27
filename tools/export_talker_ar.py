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


def _det_float(*shape):
    """Deterministic stand-in for torch.randn (random ops can trip an active
    vmap randomness-error mode after the onnxscript/dynamo import)."""
    n = 1
    for s in shape:
        n *= s
    return (torch.arange(n, dtype=torch.float32) % 13 - 6).reshape(*shape) * 0.1


def export_codec_embed(model):
    talker = _talker(model)
    m = CodecEmbed(talker).float().eval()
    N = talker.model.codec_embedding.num_embeddings
    dummy = (torch.arange(4, dtype=torch.long) % N).view(1, 4)
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
    emb = _det_float(1, T, 1024); pos = _pos_ids(T); mask = torch.ones(1, T, dtype=torch.long)
    with torch.no_grad():
        ref = m(emb, pos, mask)
    print(f"  torch logits: {tuple(ref.shape)}")
    path = f"{OUT}/talker_logits.onnx"
    # The legacy tracer freezes the seq length (internal shape ops become
    # constants). The TorchDynamo exporter traces length symbolically, so use it
    # with a shared dynamic dim for the sequence axis.
    seq = torch.export.Dim("seq", min=2, max=8192)
    torch.onnx.export(
        m, (emb, pos, mask), path,
        input_names=["inputs_embeds", "position_ids", "attention_mask"],
        output_names=["logits"],
        dynamo=True,
        dynamic_shapes={"inputs_embeds": {1: seq},
                        "position_ids": {2: seq},
                        "attention_mask": {1: seq}},
        opset_version=OPSET)
    print(f"  talker_logits.onnx: {os.path.getsize(path)/1024**2:.0f} MB")
    # validate at a DIFFERENT length to prove the graph is dynamic + faithful
    try:
        import onnxruntime as ort, numpy as np
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        T2 = 13
        emb2 = _det_float(1, T2, 1024) + 0.3; pos2 = _pos_ids(T2); mask2 = torch.ones(1, T2, dtype=torch.long)
        with torch.no_grad():
            tref = m(emb2, pos2, mask2).numpy()
        oout = sess.run(None, {"inputs_embeds": emb2.numpy(),
                               "position_ids": pos2.numpy().astype(np.int64),
                               "attention_mask": mask2.numpy().astype(np.int64)})[0]
        print(f"  VALIDATION (T={T2}) max|onnx-torch| = {np.abs(oout - tref).max():.3e} "
              f"(should be < 1e-3)")
    except Exception as e:
        print(f"  validation FAILED: {e}")


# ── Stage 2b: KV-cache single-step talker export ─────────────────────────────
# The model uses torch.func.vmap internally, which the dynamo exporter cannot
# trace and the legacy tracer bakes seq length for. A single-step export sidesteps
# both: seq is always 1 (constant, nothing to bake), the legacy tracer unrolls
# vmap fine (it produced a graph before), and the only dynamic axis is the KV
# cache length on the past_* tensors. Prefill is done as repeated single steps.

def export_talker_step(model):
    from transformers.cache_utils import DynamicCache
    talker = _talker(model)
    backbone = talker.model
    codec_head = talker.codec_head
    n_layers = len(backbone.layers)
    cfg = backbone.config
    nkv = getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads"))
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    print(f"  layers={n_layers} kv_heads={nkv} head_dim={hd}")

    class Step(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.codec_head = codec_head
            self.n = n_layers
        def forward(self, inputs_embeds, position_ids, cache_position, *pkv):
            past = DynamicCache.from_legacy_cache(
                tuple((pkv[2 * i], pkv[2 * i + 1]) for i in range(self.n)))
            out = self.backbone(input_ids=None, inputs_embeds=inputs_embeds,
                                position_ids=position_ids, past_key_values=past,
                                use_cache=True, cache_position=cache_position,
                                return_dict=True)
            logits = self.codec_head(out.last_hidden_state)
            flat = []
            for k, v in out.past_key_values.to_legacy_cache():
                flat += [k, v]
            return (logits, *flat)

    m = Step().float().eval()

    def make_inputs(L):
        emb = _det_float(1, 1, 1024)
        pos = torch.full((3, 1, 1), L, dtype=torch.long)
        cpos = torch.tensor([L], dtype=torch.long)
        pkv = []
        for _ in range(n_layers):
            pkv += [_det_float(1, nkv, L, hd), _det_float(1, nkv, L, hd)]
        return (emb, pos, cpos, *pkv)

    args = make_inputs(3)
    in_names = ["inputs_embeds", "position_ids", "cache_position"]
    out_names = ["logits"]
    dyn = {"inputs_embeds": {0: "b"}, "logits": {0: "b"}}
    for i in range(n_layers):
        in_names += [f"past_k_{i}", f"past_v_{i}"]
        out_names += [f"present_k_{i}", f"present_v_{i}"]
        dyn[f"past_k_{i}"] = {0: "b", 2: "past"}
        dyn[f"past_v_{i}"] = {0: "b", 2: "past"}
        dyn[f"present_k_{i}"] = {0: "b", 2: "past1"}
        dyn[f"present_v_{i}"] = {0: "b", 2: "past1"}

    # Use the EAGER attention path (plain matmul + softmax). The SDPA paths emit
    # ops ONNX can't lower: flash -> aten::_scaled_dot_product_flash_attention_for_cpu,
    # math -> aten::_safe_softmax. Eager exports cleanly.
    for mod in backbone.modules():
        c = getattr(mod, "config", None)
        if c is not None and hasattr(c, "_attn_implementation"):
            c._attn_implementation = "eager"
    backbone.config._attn_implementation = "eager"

    with torch.no_grad():
        ref = m(*args)
    print(f"  torch step logits: {tuple(ref[0].shape)}  present_k0: {tuple(ref[1].shape)}")
    path = f"{OUT}/talker_step.onnx"
    torch.onnx.export(m, args, path, input_names=in_names, output_names=out_names,
                      dynamic_axes=dyn, opset_version=OPSET, do_constant_folding=True)
    print(f"  talker_step.onnx: {os.path.getsize(path)/1024**2:.0f} MB")

    # validate at a DIFFERENT cache length (proves the cache axis is dynamic)
    try:
        import onnxruntime as ort, numpy as np
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        a2 = make_inputs(7)
        with torch.no_grad():
            tref = m(*a2)[0].numpy()
        feed = {in_names[i]: a2[i].numpy() for i in range(len(in_names))}
        oout = sess.run(None, feed)[0]
        print(f"  VALIDATION (cache_len=7) max|onnx-torch| = {np.abs(oout - tref).max():.3e} "
              f"(should be < 1e-3)")
    except Exception as e:
        print(f"  validation FAILED: {e}")


# ── Stage 2c: subtalker (code_predictor) ─────────────────────────────────────
# The subtalker refines code0 into residual codebooks 1..15. Its forward indexes
# codec_embedding[gs-1] and lm_head[gs] by the generation step, which ONNX can't
# do at runtime. So we export the SHARED backbone once (small_to_mtp_projection +
# 5-layer model, KV cache, seq=1) and dump the 15 embeddings + 15 heads as raw
# fp16 tables; Kotlin gathers/matmuls them per step. The seq=2 prefill (past_hidden
# then last_id_hidden) is replayed as two seq=1 steps.

def export_subtalker(model):
    import numpy as np
    from transformers.cache_utils import DynamicCache
    cp = _talker(model).code_predictor
    backbone = cp.model
    proj = cp.small_to_mtp_projection
    n_layers = len(backbone.layers)
    cfg = backbone.config
    nkv = getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads"))
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    ng = len(cp.lm_head)
    print(f"  cp layers={n_layers} kv_heads={nkv} head_dim={hd} groups={ng}")
    for mod in backbone.modules():
        c = getattr(mod, "config", None)
        if c is not None and hasattr(c, "_attn_implementation"):
            c._attn_implementation = "eager"

    class Step(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.proj = proj; self.backbone = backbone; self.n = n_layers
        def forward(self, inputs_embeds, position_ids, cache_position, *pkv):
            past = DynamicCache.from_legacy_cache(
                tuple((pkv[2 * i], pkv[2 * i + 1]) for i in range(self.n)))
            out = self.backbone(input_ids=None, inputs_embeds=self.proj(inputs_embeds),
                                position_ids=position_ids, past_key_values=past,
                                use_cache=True, cache_position=cache_position, return_dict=True)
            flat = []
            for k, v in out.past_key_values.to_legacy_cache():
                flat += [k, v]
            return (out.last_hidden_state, *flat)

    m = Step().float().eval()

    def make_inputs(L):
        emb = _det_float(1, 1, 1024)
        pos = torch.full((1, 1), L, dtype=torch.long)
        cpos = torch.tensor([L], dtype=torch.long)
        pkv = []
        for _ in range(n_layers):
            pkv += [_det_float(1, nkv, L, hd), _det_float(1, nkv, L, hd)]
        return (emb, pos, cpos, *pkv)

    args = make_inputs(2)
    in_names = ["inputs_embeds", "position_ids", "cache_position"]
    out_names = ["hidden"]
    dyn = {"inputs_embeds": {0: "b"}, "hidden": {0: "b"}}
    for i in range(n_layers):
        in_names += [f"past_k_{i}", f"past_v_{i}"]
        out_names += [f"present_k_{i}", f"present_v_{i}"]
        dyn[f"past_k_{i}"] = {0: "b", 2: "past"}; dyn[f"past_v_{i}"] = {0: "b", 2: "past"}
        dyn[f"present_k_{i}"] = {0: "b", 2: "p1"}; dyn[f"present_v_{i}"] = {0: "b", 2: "p1"}

    with torch.no_grad():
        ref = m(*args)
    print(f"  torch sub hidden: {tuple(ref[0].shape)}")
    path = f"{OUT}/subtalker_step.onnx"
    torch.onnx.export(m, args, path, input_names=in_names, output_names=out_names,
                      dynamic_axes=dyn, opset_version=OPSET, do_constant_folding=True)
    print(f"  subtalker_step.onnx: {os.path.getsize(path)/1024**2:.0f} MB")

    # dump the 15 codec embeddings + 15 heads (raw fp16, gathered in Kotlin)
    ce = torch.stack([backbone.codec_embedding[i].weight.detach() for i in range(ng)])   # [G,2048,1024]
    hw = torch.stack([cp.lm_head[i].weight.detach() for i in range(ng)])                 # [G,2048,1024]
    ce.to(torch.float16).cpu().numpy().tofile(f"{OUT}/subtalker_codec_embed.f16")
    hw.to(torch.float16).cpu().numpy().tofile(f"{OUT}/subtalker_heads.f16")
    print(f"  subtalker_codec_embed.f16: {tuple(ce.shape)}  subtalker_heads.f16: {tuple(hw.shape)}")

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        a2 = make_inputs(5)
        with torch.no_grad():
            tref = m(*a2)[0].numpy()
        feed = {in_names[i]: a2[i].numpy() for i in range(len(in_names))}
        oout = sess.run(None, feed)[0]
        print(f"  VALIDATION (cache_len=5) max|onnx-torch| = {np.abs(oout - tref).max():.3e} (should be < 1e-3)")
    except Exception as e:
        print(f"  validation FAILED: {e}")


# ── Stage 2d: code2wav (codec tokens -> audio) ───────────────────────────────

def _find_code2wav(model):
    """Locate the codec decoder (codes[1,16,T] -> audio) in the loaded model."""
    roots = []
    st = getattr(model, "speech_tokenizer", None)
    if st is not None:
        roots += [st, getattr(st, "model", None)]
    roots.append(model)
    for r in roots:
        if r is None:
            continue
        for attr in ["decoder", "code2wav", "vocoder", "generator"]:
            d = getattr(r, attr, None)
            if d is not None and isinstance(d, torch.nn.Module):
                return d, attr
    return None, None


def export_code2wav(model):
    dec, name = _find_code2wav(model)
    if dec is None:
        print("  code2wav NOT found — run: print([n for n,_ in model.speech_tokenizer.named_children()])")
        return
    c2w = dec.float().eval()
    dummy = (torch.arange(1 * 16 * 32, dtype=torch.long) % 2048).view(1, 16, 32)
    try:
        with torch.no_grad():
            out = c2w(dummy)
        print(f"  code2wav via .{name}: in[1,16,32] -> {tuple(out.shape) if hasattr(out,'shape') else type(out)}")
    except Exception as e:
        print(f"  code2wav forward failed ({e}); share model.speech_tokenizer children")
        return
    path = f"{OUT}/code2wav.onnx"
    torch.onnx.export(c2w, (dummy,), path, input_names=["codec_tokens"], output_names=["audio"],
                      dynamic_axes={"codec_tokens": {0: "b", 2: "t"}, "audio": {0: "b", 2: "n"}},
                      opset_version=OPSET, do_constant_folding=True)
    print(f"  code2wav.onnx: {os.path.getsize(path)/1024**2:.0f} MB")


def write_manifest():
    import json
    files = sorted(os.listdir(OUT))
    info = {f: os.path.getsize(os.path.join(OUT, f)) for f in files}
    with open(f"{OUT}/build_manifest.json", "w") as fp:
        json.dump(info, fp, indent=2)
    print("\n=== Output files ===")
    for f in files:
        print(f"  {f:34s} {info[f]/1024**2:8.1f} MB")


if __name__ == "__main__":
    print("=== Stage 1: text_cond + codec_embed ===")
    export_text_cond_table(model)   # noqa: F821  (provided by the Colab session)
    export_codec_embed(model)       # noqa: F821
    print("\n=== Stage 2b: talker_step (KV-cache, single step) ===")
    export_talker_step(model)       # noqa: F821
    print("\n=== Stage 2c: subtalker (code_predictor) ===")
    export_subtalker(model)         # noqa: F821
    print("\n=== Stage 2d: code2wav ===")
    export_code2wav(model)          # noqa: F821
    write_manifest()
    print("\nDone. Full ONNX model set built. Next: bake voices + Kotlin decoder.")
