# Supported & candidate models (Russian-capable)

Reciter is multi-architecture: the active model's `model.json` declares an
`architecture`, and the TTS service routes to the matching engine. New models
are added by shipping a manifest (and, for a new architecture, an engine impl).

| Model | Architecture (`architecture`) | Russian | ONNX / on-device | Status | Notes |
|---|---|---|---|---|---|
| **Qwen3-TTS 0.6B** | `qwen3-codec` | ✅ good | yes (4 ONNX, ~550 MB INT8) | **supported** | AR talker → neural codec → code2wav. `tools/export_onnx_corrected.py` |
| **Qwen2.5-Omni Talker** | `qwen3-codec` | ✅ | yes (same pipeline) | supported* | Same codec contract; export the Talker+Token2Wav, ship a manifest |
| **MMS-TTS (rus)** | `vits` | ✅ ok | yes (1 ONNX, ~40–145 MB) | **experimental** | `facebook/mms-tts-rus`. `tools/export_mms_vits.py`. Char vocab; verify on device |
| **Piper (ru_RU-*)** | `vits` | ✅ good | ONNX-native, but needs eSpeak phonemes | planned | Piper ships ONNX + config; on-device **eSpeak-ng phonemizer** is the open piece |
| **Silero (ru v3/v4)** | `silero` | ✅ excellent | ONNX export exists | planned | Best lightweight Russian; custom symbol set + stress marks; needs a `silero` engine |
| **XTTS v2 (Coqui)** | `xtts` | ✅ good | heavy (GPT + decoder + HiFiGAN) | planned | Zero-shot cloning; flow/decoder engine required; large |
| **F5-TTS** | `f5` | ⚠️ varies | flow-matching, multi-ONNX | planned | DiT flow-matching + Vocos; needs reference audio + a `f5` engine |
| **Fun-CosyVoice3 0.5B** | `cosyvoice` | ✅ improved | LLM + flow-matching + HiFiGAN | **introspection / WIP** | v3 broadened multilingual (Russian decent). `tools/export_cosyvoice3.py`; on-device engine in progress |
| **CosyVoice 2** | `cosyvoice` | ⚠️ weak | same | not recommended | Mostly zh/en/ja/ko/yue; Russian poor — prefer v3 |

### CosyVoice (`cosyvoice`) — current stage
`CosyVoiceInferenceEngine` is an **introspection engine**: it loads every
exported component (LLM / FLOW / VOCODER / SPEAKER_ENCODER / SPEECH_TOKENIZER)
and logs each one's ONNX input/output names + shapes, then reports "runtime not
implemented" (no garbage audio). Import the ZIP from `export_cosyvoice3.py`,
read the component I/O from the in-app log, and use it to implement the runtime
(AR LLM → speech tokens → CFM ODE steps → HiFiGAN; zero-shot needs reference
audio). The official tooling exports flow/vocoder; the AR LLM in ONNX is the
main open piece.

\* drop-in via manifest, no code change.

## How architecture routing works
`QwenTTSEngine.initEngine()` reads `ModelConfig.activeProfile().architecture`:
- `qwen3-codec` → `QwenInferenceEngine` (autoregressive codec).
- `vits` → `VitsInferenceEngine` (single-shot `input_ids` → `waveform`).
- anything else → logged as "not implemented" (no crash); add an engine that
  implements `SpeechSynthesizer` and a branch here to support it.

## Why not "just add a manifest" for CosyVoice / F5 / XTTS?
They are **not** autoregressive-codec models. CosyVoice/F5/XTTS use an LLM or
DiT producing latent/mel features refined by **flow-matching**, then a separate
neural vocoder — a fundamentally different runtime loop (ODE solver, CFG,
reference-audio conditioning). Each needs its own `SpeechSynthesizer`
implementation. The manifest carries their metadata; the engine is the work.

## Adding a new architecture (checklist)
1. Implement `SpeechSynthesizer` (see `VitsInferenceEngine` for the simplest example).
2. Add a branch in `QwenTTSEngine.initEngine()` for the new `architecture` value.
3. Write an export script under `tools/` that emits ONNX + tokenizer assets +
   a `model.json` with the new `architecture` and the model's `voices`.
4. Document it in this table.

## Russian-quality recommendation
For quality+practicality on-device today: **Qwen3-TTS** (multilingual, this
app's primary target) and **Silero ru** (best lightweight Russian — worth a
dedicated engine next). MMS-TTS rus is a quick `vits` win to validate the
multi-engine path. CosyVoice is not advised for Russian.
