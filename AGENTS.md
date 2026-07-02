# Instructions for AI Coding Agents

## Core Rules and Guardrails

1. **NEVER DELETE THE KEYSTORE OR ITS BACKUP:**
   - **Do not delete `app/debug.keystore` or `debug.keystore.base64`.**
   - The keystore is critical for signing the debug APK builds both locally and on GitHub Actions CI/CD workflows.
   - If `app/debug.keystore` is missing, it should be auto-decoded from `debug.keystore.base64` in `app/build.gradle.kts`.

2. **MAINTAIN CI/CD WORKFLOWS:**
   - Do not remove the keystore restoration steps from `.github/workflows/build.yml`.
   - Ensure the workflow handles either an existing `app/debug.keystore` or decodes it successfully from `debug.keystore.base64`.

3. **BUMP VERSION ON EVERY MODIFICATION:**
   - **Always increment `versionCode` and `versionName` in `app/build.gradle.kts` with every edit/fix/change.**
   - Increment the `versionCode` by 1 (e.g., `10018` -> `10019`) and increment the patch version of `versionName` (e.g., `"1.0.18"` -> `"1.0.19"`).
