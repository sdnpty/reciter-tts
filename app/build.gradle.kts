plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
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
        versionCode = 10019
        versionName = "1.0.19"

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
