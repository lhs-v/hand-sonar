import java.net.URI
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.watchpose.handsonar.mobile"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.watchpose.handsonar" // 워치 앱과 같은 이름·서명이어야 2단계 Data Layer가 이어진다
        minSdk = 30
        targetSdk = 36
        versionCode = 1
        versionName = "0.1.0"
        ndk { abiFilters += "arm64-v8a" } // 폴드8만 쓴다. MediaPipe 네이티브 라이브러리를 한 벌만 넣어 APK를 줄인다
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    androidResources {
        noCompress += "task" // MediaPipe 모델 파일을 압축하지 않고 넣는다
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    implementation(project(":shared"))

    implementation(platform(libs.compose.bom))
    implementation(libs.compose.ui)
    implementation(libs.compose.foundation)
    implementation(libs.compose.material3)
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.camerax.core)
    implementation(libs.camerax.camera2)
    implementation(libs.camerax.lifecycle)
    implementation(libs.mediapipe.tasks.vision)
}

// MediaPipe 손 모델은 공개 저장소에 넣지 않는다(구글 모델 파일을 다시 배포하지 않으려고). 없으면 빌드 전에 공식 주소에서 받아
// SHA-256을 확인한다. 인터넷이 없으면 -PhandModelUrl=file:///경로/hand_landmarker.task 로 가진 파일을 쓴다.
val downloadHandModel = tasks.register("downloadHandModel") {
    description = "MediaPipe 손 모델(hand_landmarker.task)이 없으면 내려받는다"
    val target = layout.projectDirectory.file("src/main/assets/hand_landmarker.task").asFile
    val part = layout.buildDirectory.file("tmp/hand_landmarker.task.part").get().asFile
    val url = providers.gradleProperty("handModelUrl")
        .getOrElse("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
    val expected = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
    onlyIf { !target.exists() }
    doLast {
        part.parentFile.mkdirs()
        URI(url).toURL().openConnection().apply { connectTimeout = 30_000; readTimeout = 60_000 }.getInputStream().use { input -> part.outputStream().use { input.copyTo(it) } }
        val sha = MessageDigest.getInstance("SHA-256").digest(part.readBytes()).joinToString("") { "%02x".format(it) }
        if (sha != expected) throw GradleException("hand_landmarker.task의 SHA-256이 다릅니다: $sha")
        target.parentFile.mkdirs()
        Files.move(part.toPath(), target.toPath(), StandardCopyOption.REPLACE_EXISTING)
    }
}
tasks.named("preBuild") { dependsOn(downloadHandModel) }
