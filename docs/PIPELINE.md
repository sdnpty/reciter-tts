# On-device-пайплайн Qwen3-TTS (проверен)

Вся цепочка (текст → префилл → авторегрессионная генерация кодек-кодов →
code2wav → аудио 24 кГц) воспроизведена чисто на экспортированных ONNX + сырых
таблицах и совпадает с PyTorch `model.generate` **бит-в-бит** (коды 100 %,
аудио идентично). Голоса — запечённые x-векторы из русских референс-клипов;
приложение также умеет записывать новые референс-клипы с микрофона (см. §6),
добавляя пользовательские голоса прямо на устройстве.

## 1. Сборка файлов модели (Google Colab, GPU)

```python
# зависимости (чистые; БЕЗ onnxscript — используется legacy-экспорт)
!pip install -q torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
!pip install -q transformers==4.57.3 accelerate sentencepiece protobuf huggingface-hub qwen-tts onnx onnxruntime

import torch
from qwen_tts import Qwen3TTSModel
model_wrapper = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base", device_map="cpu", dtype=torch.float32)
model = model_wrapper.model; model.eval()

# экспорт всех блоков + таблиц (пишет в /content/qwen3-tts-onnx/android)
exec(open(download("tools/export_talker_ar.py")).read())
```

Результат (набор моделей для устройства):

| файл | что это |
|---|---|
| `talker_step.onnx` | 28-слойный talker, 1 шаг, KV-кэш → logits[3072] + hidden + KV |
| `subtalker_step.onnx` | 5-слойный code predictor, 1 шаг, KV-кэш → hidden + KV |
| `codec_embed.onnx` | code0 (0..3071) → эмбеддинг[1024] |
| `code2wav.onnx` | codes[1,16,T] → аудио |
| `text_cond_table.f16` | [151936,1024] = text_projection(text_embedding) |
| `subtalker_codec_embed.f16` | [15,2048,1024] residual-эмбеддинги |
| `subtalker_heads.f16` | [15,2048,1024] веса residual-голов (без bias) |

Три больших ONNX автоматически квантуются в INT8 в конце
`export_talker_ar.py` (`quantize_for_android()`, QInt8, per_channel для
talker) → ~430/80/210 МБ; набор сжимается с ~1,7 ГБ до ~700 МБ.

## 2. Запекание голосов (путь через x-векторы)

Для каждого русского референс-клипа (3–10 с чистой речи; имя файла становится
именем голоса): `generate_speaker_prompt(..., x_vector_only_mode=True)` =
1024-мерный x-вектор. Они сохраняются конкатенацией в `baked_voices.bin` +
`voices.json`. Это делает Colab-ноутбук `tools/qwen3_tts_export_colab.ipynb`
(ячейки 4–5), позволяя загрузить и собственные русские клипы.

## 3. Алгоритм на устройстве (портирован в QwenArEngine.kt)

Константы: H=1024, слоёв talker=28, слоёв subtalker=5, kv_heads=8, head_dim=128,
groups=15, EOS=2150, диапазон codebook0 [0,2048), подавление 2048..3071 кроме EOS,
repetition_penalty=1.05. tts_bos/eos/pad=151672/151673/151671. codec
pad/bos/think/think_bos/think_eos=2148/2149/2154/2156/2157, русский язык=2069.

**Префилл** (собирается из input_ids = role(3)+text+suffix(5) и x-вектора голоса):
```
role   = text_cond(input_ids[:3])                              # 3
codec0 = codec_embed([think, think_bos, lang, think_eos])      # 4
codec1 = codec_embed([codec_pad, codec_bos])                   # 2
codec_input = [codec0, xvector, codec1]                        # 7
tie    = [tts_pad*5, tts_bos] + codec_input[:6]                # 6
prefill = [role, tie, text_cond(input_ids[3]) + codec_input[6]]  # 10
trailing_text = [text_cond(input_ids[4:-5]), tts_eos]
```
10 векторов префилла прогоняются через `talker_step` по одному (позиции 0..9,
3D position_ids = arange, пустой→растущий KV). Логиты последнего шага дают
code0[0], а его hidden — это `past_hidden`.

**AR-цикл** на кадр (пока code0 != EOS):
```
last_id = codec_embed(code0)
# subtalker: подать past_hidden(pos0), last_id(pos1) → head[0] → code_1
#   затем для g in 1..14: emb = subtalker_codec_embed[g-1][code_g]; шаг; head[g] → code_(g+1)
residual = 15 кодов
frame = [code0] + residual
s = last_id + Σ subtalker_codec_embed[g][residual[g]]  (g=0..14)
s += (trailing_text[step] если step < len, иначе tts_pad)
logits, hidden = talker_step(s, pos++, KV)
code0 = select(logits)   # repetition_penalty по истории code0 + подавление спецтокенов
past_hidden = hidden; step++
```
Кадры складываются в `[1,16,T]` → `code2wav.onnx` → PCM.

Эталонные реализации: `tools/infer_onnx_reference.py` (AR-цикл) и
`tools/build_prefill.py` (префилл) — обе сверены с PyTorch.

## 4. Раскладка ассетов на устройстве

`<filesDir>/models/qwen3-ar/` с файлами из шага 1 (INT8) +
`baked_voices.bin` (конкатенированные float32 x-векторы) + `voices.json`
(имена) + `vocab.json`, `merges.txt` для токенизатора.

## 5. inputIdsFor (из ar_config.json, снято с модели)

```
role_tokens   = [151644, 77091, 198]                 # <|im_start|>assistant\n
suffix_tokens = [151645, 198, 151644, 77091, 198]    # <|im_end|>\n<|im_start|>assistant\n
inputIdsFor(text) = role_tokens + Qwen3Tokenizer.encodeForTTS(text) + suffix_tokens
```
Запечённые голоса: русские референс-клипы (по одному x-вектору, имя — по файлу).

## 6. Клонирование голоса на устройстве (микрофон)

Приложение записывает моно-референс 16 кГц (`VoiceRecorder`), сохраняет его
через `CustomVoiceStore` в `filesDir/custom_voices/<id>.wav` и показывает как
чип голоса на вкладке «Синтез». `speaker_encoder.onnx` содержит Kaldi-fbank
фронтенд прямо в графе (волна → x-вектор), поэтому `VoiceCloner` вычисляет
1024-мерный x-вектор из сырого клипа без DSP на устройстве, кэширует его как
`<id>.xvec`, а `QwenArEngine` подмешивает такие голоса в свою карту при
загрузке — записанный голос синтезируется точно так же, как запечённый. Если у
активной модели нет энкодера, клип всё равно сохраняется и клонируется, как
только такая модель станет активной.

`export_talker_ar.py` определяет раскладку фич энкодера, оборачивает его
Kaldi-fbank из torchaudio (80/128 мел, 25/10 мс, окно povey, utterance-CMN) и
сверяет граф «волна → x-вектор» с `generate_speaker_prompt`.
