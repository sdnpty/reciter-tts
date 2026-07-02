# Авторегрессионное декодирование Talker — план (путь A: ONNX + KV-кэш)

Однопроходный экспорт talker `input_ids(text) → logits` архитектурно неверен:
talker — это 28-слойный авторегрессионный декодер над **кодековым** словарём.
Подача текстовых ID роняет ONNX Runtime (`/embed/Gather`, индекс вне
`[0,3071]`). Ниже — НАСТОЯЩИЙ контракт генерации, снятый с
`Qwen3TTSModel.generate` и `Qwen3TTSTalkerForCausalLM.forward`.

## Модули

- `talker.model.text_embedding` — Embedding(151936) для текстовых ID.
- `talker.text_projection` — проецирует текстовые эмбеддинги в скрытое
  пространство talker (используется для текстового условия И для эмбеддингов
  tts_bos/eos/pad).
- `talker.model.codec_embedding` — Embedding(3072) для кодов codebook-0.
- `talker.codec_head` — Linear → логиты codebook-0 (3072).
- `talker.code_predictor` — **subtalker** (5 слоёв) со своими 15
  `codec_embedding[i]` (каждый Embedding(2048)) и авторегрессионным `generate`.

## Пошаговый цикл talker на кадр (самое сложное)

```
last_id_hidden = codec_embedding(code0)                       # [B,1,H]
# subtalker сам авторегрессионный (15 внутренних шагов, свой KV-кэш + сэмплинг):
pred = code_predictor.generate(
        inputs_embeds = cat(past_hidden, last_id_hidden),
        max_new_tokens = 15, do_sample, top_p, top_k, temperature)
codes_1_15 = pred.sequences                                   # [B,15]
codec_hiddens = [last_id_hidden] +
                [code_predictor.codec_embedding[i](codes_1_15[:,i]) for i in 0..14]
inputs_embeds = sum(codec_hiddens)                            # [B,1,H]
# текстовое условие:
inputs_embeds += (trailing_text_hidden[:, step] если step < T_text, иначе tts_pad_embed)
# кастомный 3D RoPE:
position_ids, rope_deltas = get_rope_index(attention_mask)    # 3D-позиции
out = model(inputs_embeds, past_key_values, position_ids, cache_position, use_cache)
logits = codec_head(out.last_hidden_state)                    # code0 СЛЕДУЮЩЕГО кадра
past_hidden = out.last_hidden_state[:, -1:]
```

Останов, когда `argmax/sample(logits) == codec_eos_token_id`. Генерация также
использует `repetition_penalty=1.05` и `suppress_tokens` (топ-1024 минус EOS).

## Префилл (один раз на реплику)

Собирает кондиционирующую последовательность из:
тегов `codec_think_*`/`codec_nothink_id`, `language_id`, эмбеддинга диктора
(`talker.get_input_embeddings()(spk_id)` или x-вектор) и
`tts_bos/eos/pad = text_projection(text_embedding([bos,eos,pad]))`.
`trailing_text_hidden = text_projection(text_embedding(text_ids))` — пошаговое
текстовое условие, потребляемое циклом генерации выше.

## Набор ONNX-экспортов (путь A)

1. `text_cond.onnx` — text_ids → `text_projection(text_embedding(ids))`
   (даёт trailing_text_hidden, а также bos/eos/pad при подаче этих ID).
2. `codec_embed.onnx` — code0 (0..3071) → эмбеддинг (для last_id_hidden).
3. `talker_step.onnx` — `inputs_embeds[B,1,H]` + `past_key_values`(28×{k,v}) +
   `position_ids`(3D) + `cache_position` → `logits[B,1,3072]` + новый KV +
   `past_hidden[B,1,H]`. Без subtalker (экспортируется отдельно).
4. `subtalker_step.onnx` — один шаг code_predictor со своим KV-кэшем, чтобы
   гнать 15 внутренних residual-кодов.
5. `code2wav.onnx` — [1,16,frames] → аудио (уже корректен).

## Сложные/рискованные места (нужны итерации на устройстве)

- **Кастомный 3D RoPE** (`get_rope_index` + `rope_deltas`) надо воспроизвести
  в Kotlin в точности либо запечь в ONNX-граф с `position_ids` на входе.
- **Вложенный авторегрессионный subtalker** со своим KV-кэшем и сэмплингом.
- **Сборка префилла** (think-теги / language id / эмбеддинг диктора / bos-eos-pad).
- **Сэмплинг** (top_k/top_p/temperature + repetition_penalty + suppress_tokens);
  жадный argmax — отправная точка, но может ухудшить качество.
- Экспорт KV-кэша в ONNX с динамическими past_key_values для 28 (talker) + 5
  (subtalker) слоёв.

## Известные константы (из загруженной модели 0.6B)

- скрытая размерность talker `H = 1024`; таблица текстового условия =
  `[151936, 1024]` (fp16, ~297 МБ).
- `tts_bos_token_id = 151672`, `tts_eos_token_id = 151673`, `tts_pad_token_id = 151671`.
- codec_embedding = `Embedding(3072, 1024)`; code predictor = 15 × `Embedding(2048, 1024)`.

## Workflow «сначала валидация» (обязателен для этапа 2)

Этап 2 (talker_step / subtalker_step с KV-кэшем + кастомным 3D RoPE) нельзя
делать вслепую. Перед каждым ONNX-экспортом снимаем эталон PyTorch в Colab:
1. получить рабочий `model.generate(...)`, дающий референс-WAV;
2. хукнуть `talker.forward` и записать реальные формы/типы аргументов
   (раскладка past_key_values, position_ids, trailing_text_hidden,
   cache_position) для первых шагов;
3. выгрузить `logits` первого шага, покадровые коды и итоговое аудио.
Каждый ONNX-блок затем численно сверяется с этими тензорами до отправки на
устройство.

## Точный контракт talker.forward (снят через forward_pre_hook)

Префилл (вызов 0): `inputs_embeds[1,T,1024]`, `attention_mask[1,T]`,
`position_ids[1,T]`, `cache_position[T]`, `trailing_text_hidden[1,1,1024]`,
`tts_pad_embed[1,1,1024]`, `use_cache`. Идёт по ветке префилла
(generation_step=-1, без subtalker) → backbone+codec_head по всей
последовательности.

Шаг генерации (вызовы 1+): `input_ids[1,1]` (текущий code0), `past_key_values`
(DynamicCache), `cache_position[1]`, `position_ids[1,1]`,
`attention_mask[1,t]` (растёт), `past_hidden[1,1,1024]` (пред. hidden talker,
питает subtalker), `trailing_text_hidden[1,1,1024]`, `tts_pad_embed`,
`generation_step`. Внутри: codec_embed(code0) → code_predictor.generate
(15 residual-кодов) → сумма 16 эмбеддингов + текстовое условие →
backbone(шаг, KV) → codec_head → логиты следующего code0; возвращает
past_hidden = hidden[:, -1:].

## Стратегия экспорта (этап 2, меньше риска)

Поскольку ветка префилла (`inputs_embeds` seq>1) переиспользует RoPE самой
модели и пропускает subtalker, этап 2a экспортирует `talker_logits`:
`inputs_embeds[1,T,1024] + attention_mask[1,T] -> logits[1,T,3072]`.
Kotlin-цикл перегоняет его по растущему `inputs_embeds` (O(T²), просто, без
экспорта KV) и читает `logits[:, -1]` для следующего code0. Экспорт KV-кэша —
последующая оптимизация. Subtalker (code_predictor) — аналогично.

## Замечание о масштабе

Это большой многоэтапный порт, сопоставимый со специализированными
референс-движками (`qwen3-tts.cpp`, чистый C `qwen3-tts`). Потребует
нескольких итераций на устройстве. Поэтапная поставка: (1) экспорт text_cond +
codec_embed, (2) talker_step + subtalker_step с KV-кэшем, (3) Kotlin-декодер +
RoPE, (4) сэмплинг + префилл, (5) настройка качества.

## Статус

- [x] таблица text_cond (fp16), codec_embed.onnx
- [x] **talker_step.onnx — один шаг с KV-кэшем, eager attn, сверен: max|onnx-torch|=5.6e-5 при cache_len=7**
- [x] экспорт шага subtalker (code_predictor) + code2wav
- [x] **Сквозной ONNX-референс: AR-цикл совпадает с model.generate на 100 % (29/29 кадров бит-в-бит)**
- [ ] репликация сборки префилла (для запечённых голосов)
- [ ] Kotlin AR-декодер + запекание голосов
