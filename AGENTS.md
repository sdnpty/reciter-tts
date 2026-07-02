# Инструкции для ИИ-агентов

## Основные правила и ограничения

1. **НИКОГДА НЕ УДАЛЯЙ KEYSTORE И ЕГО РЕЗЕРВНУЮ КОПИЮ:**
   - **Не удаляй `app/debug.keystore` и `debug.keystore.base64`.**
   - Keystore критичен для подписи debug-сборок APK — и локально, и в CI/CD на GitHub Actions.
   - Если `app/debug.keystore` отсутствует, он должен автоматически декодироваться из `debug.keystore.base64` в `app/build.gradle.kts`.

2. **ПОДДЕРЖИВАЙ CI/CD-WORKFLOW:**
   - Не удаляй шаги восстановления keystore из `.github/workflows/build.yml`.
   - Workflow должен работать и с существующим `app/debug.keystore`, и с успешным декодированием из `debug.keystore.base64`.

3. **ПОДНИМАЙ ВЕРСИЮ ПРИ КАЖДОМ ИЗМЕНЕНИИ:**
   - **Всегда увеличивай `versionCode` и `versionName` в `app/build.gradle.kts` при каждой правке/фиксе/изменении.**
   - `versionCode` увеличивай на 1 (например, `10018` -> `10019`), в `versionName` увеличивай patch-версию (например, `"1.0.18"` -> `"1.0.19"`).
