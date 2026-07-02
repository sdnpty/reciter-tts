import java.util.Base64

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

val keystoreBase64File = rootProject.file("debug.keystore.base64")
val keystoreFile = file("debug.keystore")

logger.lifecycle("Auto-keystore-debug: keystoreBase64File path = ${keystoreBase64File.absolutePath}, exists = ${keystoreBase64File.exists()}")
logger.lifecycle("Auto-keystore-debug: keystoreFile path = ${keystoreFile.absolutePath}, exists = ${keystoreFile.exists()}")

if (keystoreBase64File.exists() && !keystoreFile.exists()) {
    try {
        val base64Content = keystoreBase64File.readText().trim()
        val decoded = Base64.getDecoder().decode(base64Content)
        keystoreFile.writeBytes(decoded)
        logger.lifecycle("Successfully auto-decoded debug.keystore from debug.keystore.base64 to ${keystoreFile.absolutePath}")
    } catch (e: Exception) {
        logger.error("Failed to auto-decode debug.keystore: ${e.message}", e)
    }
}

// Fallback: If still doesn't exist, generate a new one dynamically!
if (!keystoreFile.exists()) {
    try {
        logger.lifecycle("debug.keystore not found. Generating a new one dynamically...")
        val process = ProcessBuilder(
            "keytool", "-genkey", "-v",
            "-keystore", keystoreFile.absolutePath,
            "-storepass", "android",
            "-alias", "androiddebugkey",
            "-keypass", "android",
            "-keyalg", "RSA",
            "-keysize", "2048",
            "-validity", "10000",
            "-dname", "CN=Android Debug,O=Android,C=US"
        ).start()
        val exitCode = process.waitFor()
        if (exitCode == 0) {
            logger.lifecycle("Successfully generated a new debug.keystore at ${keystoreFile.absolutePath}")
        } else {
            logger.error("Failed to generate debug.keystore, keytool exited with $exitCode")
        }
    } catch (e: Exception) {
        logger.error("Failed to generate debug.keystore dynamically: ${e.message}", e)
    }
}

tasks.register("gitAddKeystore") {
    doLast {
        try {
            val process = ProcessBuilder("git", "add", "debug.keystore.base64")
                .directory(rootDir)
                .start()
            val output = process.inputStream.bufferedReader().readText()
            val error = process.errorStream.bufferedReader().readText()
            val exitCode = process.waitFor()
            println("git add output: $output")
            println("git add error: $error")
            println("git add debug.keystore.base64 exited with code $exitCode")
        } catch (e: Exception) {
            println("Failed to run git add: ${e.message}")
        }
    }
}

android {
    namespace = "com.qwen3.tts"
    compileSdk = 34

    // Name built APKs with the version, e.g. reciter-tts-v1.0.1-release.apk
    applicationVariants.all {
        val variant = this
        outputs.all {
            (this as com.android.build.gradle.internal.api.BaseVariantOutputImpl)
                .outputFileName = "reciter-tts-v${variant.versionName}-${variant.name}.apk"
        }
    }

    buildFeatures {
        viewBinding = true
        buildConfig = true
    }

    defaultConfig {
        applicationId = "com.qwen3.tts"
        minSdk = 27
        targetSdk = 34
        // Bump patch (and versionCode) on every commit: 1.0.1 -> 1.0.2 -> ...
        versionCode = 10038
        versionName = "1.0.38"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"


    }

    signingConfigs {
        // Fixed debug key committed to the repo so every CI build is signed with
        // the SAME signature — otherwise updating over a previous build fails with
        // a "version/signature conflict" (each CI run generates a new debug key).
        getByName("debug") {
            storeFile = file("debug.keystore")
            storePassword = "android"
            keyAlias = "androiddebugkey"
            keyPassword = "android"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
        debug {
            isMinifyEnabled = false
            applicationIdSuffix = ".debug"
            // No versionNameSuffix: the APK is already named "...-debug.apk" from
            // the variant name, so "-debug" here produced "...-debug-debug.apk".
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }


    testOptions {
        unitTests.isReturnDefaultValues = true
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
        // Microsoft onnxruntime-android and sherpa-onnx-android each bundle a
        // libonnxruntime.so; keep one to avoid a duplicate-native-lib build error.
        jniLibs {
            pickFirsts += "**/libonnxruntime.so"
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.2")
    // ONNX Runtime Android — последняя стабильная в Maven Central (1.26.0)
    // Проверено: https://central.sonatype.com/artifact/com.microsoft.onnxruntime/onnxruntime-android
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.26.0")

    // sherpa-onnx (k2-fsa): Kokoro / Piper-VITS TTS with built-in espeak-ng g2p.
    // Not on Maven Central — the AAR is downloaded into app/libs/ by CI (see
    // .github/workflows/build.yml). The static-link variant compiles ONNX
    // Runtime into its JNI, so it doesn't clash with microsoft onnxruntime.
    implementation(fileTree(mapOf("dir" to "libs", "include" to listOf("*.aar"))))

    // Если нужна 1.27.0 — скачай AAR вручную и положи в app/libs/
    // Затем закомментируй строку выше и раскомментируй:
    // implementation(files("libs/onnxruntime-android-1.27.0.aar"))

    // Tokenization
    implementation("com.google.code.gson:gson:2.11.0")

    // Testing
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20231013")
}
