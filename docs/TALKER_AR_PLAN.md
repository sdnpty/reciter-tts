# Talker autoregressive decode — plan (path A: ONNX + KV cache)

The single-pass `input_ids(text) → logits` talker export is architecturally
wrong: the talker is a 28-layer autoregressive decoder over the **codec**
vocabulary. Feeding text ids crashes ONNX Runtime (`/embed/Gather`, idx out of
`[0,3071]`). Below is the REAL generation contract, taken from
`Qwen3TTSModel.generate` and `Qwen3TTSTalkerForCausalLM.forward`.

## Modules

- `talker.model.text_embedding` — Embedding(151936) for text ids.
- `talker.text_projection` — projects text embeddings into the talker hidden
  space (used for the text condition AND for tts_bos/eos/pad embeds).
- `talker.model.codec_embedding` — Embedding(3072) for codebook-0 codes.
- `talker.codec_head` — Linear → codebook-0 logits (3072).
- `talker.code_predictor` — the **subtalker** (5 layers) with its own 15
  `codec_embedding[i]` (Embedding(2048) each) and an autoregressive `generate`.

## Per-frame talker step (the hard part)

```
last_id_hidden = codec_embedding(code0)                       # [B,1,H]
# subtalker is itself autoregressive (15 inner steps, own KV cache + sampling):
pred = code_predictor.generate(
        inputs_embeds = cat(past_hidden, last_id_hidden),
        max_new_tokens = 15, do_sample, top_p, top_k, temperature)
codes_1_15 = pred.sequences                                   # [B,15]
codec_hiddens = [last_id_hidden] +
                [code_predictor.codec_embedding[i](codes_1_15[:,i]) for i in 0..14]
inputs_embeds = sum(codec_hiddens)                            # [B,1,H]
# text condition:
inputs_embeds += (trailing_text_hidden[:, step] if step < T_text else tts_pad_embed)
# custom 3D RoPE:
position_ids, rope_deltas = get_rope_index(attention_mask)    # 3D positions
out = model(inputs_embeds, past_key_values, position_ids, cache_position, use_cache)
logits = codec_head(out.last_hidden_state)                    # code0 of NEXT frame
past_hidden = out.last_hidden_state[:, -1:]
```

Stop when `argmax/sample(logits) == codec_eos_token_id`. Generation also uses
`repetition_penalty=1.05` and `suppress_tokens` (top-1024 minus EOS).

## Prefill (once per utterance)

Builds the conditioning sequence from:
`codec_think_*`/`codec_nothink_id` tags, `language_id`, speaker embedding
(`talker.get_input_embeddings()(spk_id)` or x-vector), and
`tts_bos/eos/pad = text_projection(text_embedding([bos,eos,pad]))`.
`trailing_text_hidden = text_projection(text_embedding(text_ids))` is the
per-step text condition consumed by the generate loop above.

## ONNX export set (path A)

1. `text_cond.onnx` — text_ids → `text_projection(text_embedding(ids))`
   (produces trailing_text_hidden, and bos/eos/pad when fed those ids).
2. `codec_embed.onnx` — code0 (0..3071) → embedding (for last_id_hidden).
3. `talker_step.onnx` — `inputs_embeds[B,1,H]` + `past_key_values`(28×{k,v}) +
   `position_ids`(3D) + `cache_position` → `logits[B,1,3072]` + new KV +
   `past_hidden[B,1,H]`. Excludes the subtalker (exported separately).
4. `subtalker_step.onnx` — code_predictor single step with its own KV cache,
   to drive the 15 inner residual codes.
5. `code2wav.onnx` — [1,16,frames] → audio (already correct).

## Hard/risky items (need device iteration)

- **Custom 3D RoPE** (`get_rope_index` + `rope_deltas`) must be reproduced in
  Kotlin exactly, or baked into the ONNX graph with `position_ids` as input.
- **Nested autoregressive subtalker** with its own KV cache + sampling.
- **Prefill construction** (think tags / language id / speaker embed / bos-eos-pad).
- **Sampling** (top_k/top_p/temperature + repetition_penalty + suppress_tokens);
  greedy argmax is a starting point but may degrade quality.
- KV-cache ONNX export with dynamic past_key_values for 28 (talker) + 5
  (subtalker) layers.

## Scope note

This is a large, multi-stage port comparable to the dedicated reference engines
(`qwen3-tts.cpp`, pure-C `qwen3-tts`). It will require several on-device
iterations. Staged delivery: (1) text_cond + codec_embed exports, (2) talker_step
+ subtalker_step with KV cache, (3) Kotlin decode loop + RoPE, (4) sampling +
prefill, (5) quality tuning.
