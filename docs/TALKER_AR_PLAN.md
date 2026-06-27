# Talker autoregressive decode — plan (path A: ONNX + KV cache)

The single-pass `input_ids(text) → logits` talker export is architecturally
wrong: the talker is a 28-layer autoregressive decoder over the **codec**
vocabulary (embedding has 3072 entries, not the text vocab). Feeding text token
ids crashes ONNX Runtime:

```
Gather '/embed/Gather': idx=78910 must be within [-3072, 3071]
```

## Real architecture (from the Qwen3-TTS report + reference C/C++ engines)

```
text → BPE → [text condition]
                 │
   ┌─────────────▼──────────────────────────────┐
   │ Talker (28-layer Qwen3, KV cache)           │  autoregressive, 1 frame/step
   │   step input  = prev audio code embed (3072)│
   │                 + text condition (prefill)  │
   │   step output = codebook-0 logits (3072)    │
   └─────────────┬──────────────────────────────┘
                 │ code 0 per frame
   ┌─────────────▼──────────────┐
   │ Code Predictor (5 layers)  │  15 sequential passes → codebooks 1..15
   └─────────────┬──────────────┘
                 │ 16 codes/frame
   ┌─────────────▼──────────────┐
   │ Code2Wav (causal ConvNet)  │  RVQ dequant + 480× upsample → 24 kHz PCM
   └────────────────────────────┘
```

Stop on the codec EOS token or a max-frames cap.

## Required ONNX exports (replaces the current 4-file set)

1. **text_encoder / embeddings** — produce the text condition (text hidden
   states) once per utterance. May be the main model's text embedding +
   `text_projection`.
2. **talker_step** — single decode step with KV cache:
   - inputs: `inputs_embeds` (or `audio_code` + internal codec embed),
     `past_key_values` (28 layers × {k,v}), `cache_position`,
     and the text-condition tensors (`trailing_text_hidden` / `tts_pad_embed`
     — exact contract TBD from the generation source).
   - outputs: `logits` (codebook-0, width 3072), updated `past_key_values`.
3. **codec_embed** — embeds a generated code (0..3071) back to hidden for the
   next step (the 3072-wide embedding the Gather error referenced).
4. **code_predictor_step** — code 0 (+ text/hidden) → residual codebooks 1..15.
5. **code2wav** — [1, 16, frames] → audio (already correct).

## Kotlin decoder (QwenInferenceEngine)

```
prefill text condition once
init empty KV cache (28 layers)
code = BOS_codec
frames = []
repeat up to MAX_FRAMES:
    h = codec_embed(code)
    logits, kv = talker_step(h, kv, cache_position, text_cond)
    code0 = argmax(logits)            # or sampling
    if code0 == EOS_codec: break
    fine = code_predictor(code0, ...) # 15 codes
    frames += [code0, *fine]
audio = code2wav(stack(frames) as [1,16,T])
```

## Open items (need the model's generation source to finalize)

- Exact tensor contract for the text condition: how `trailing_text_hidden`,
  `tts_pad_embed`, `generation_step` are produced and fed.
- KV-cache I/O naming/shape for the ONNX `talker_step` export.
- Start (BOS) and stop (EOS) codec token ids.
- Whether code_predictor consumes the talker hidden state or the code-0 id.

These come straight from `inspect.getsource(model.generate / talker.forward)`;
the export is written against that contract.
