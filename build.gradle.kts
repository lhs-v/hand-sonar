import com.android.build.api.dsl.ApplicationExtension
import java.util.Properties

plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.kotlin.jvm) apply false // KGP 2.4.21을 빌드 클래스패스에 고정(센서 허브와 같음)
    alias(libs.plugins.kotlin.compose) apply false
}

// 두 앱(wear·mobile)의 debug 빌드 서명(전송 설계 §6). 데이터 레이어는 워치 앱과 폰 앱의 applicationId와 서명이 같아야 이어진다.
// local.properties에 handsonar.keystore(키 파일 경로, 상대 경로는 이 폴더 기준)가 있으면 그 키로, 없으면 그 PC의 기본 디버그 키로 서명한다.
// 별칭·암호는 디버그 키 기본값(androiddebugkey / android)이고 handsonar.keyAlias·handsonar.storePassword·handsonar.keyPassword로 바꾼다.
// 경로가 적혀 있는데 파일이 없으면 멈춘다: 다른 키로 몰래 서명하면 데이터 레이어가 안 이어지고, 깔린 앱 위에 덮어 깔기도 거부된다.
// 같은 까닭으로 값이 빈 속성(handsonar.keystore= 등)과, 키 경로 없이 적힌 별칭·암호도 실수로 보고 멈춘다(기본 키로 넘어가지 않는다).
// 키 파일은 저장소에 넣지 않는다(.gitignore가 *.keystore·*.jks를 막는다).
val localProperties = Properties().apply {
    val file = rootProject.file("local.properties")
    if (file.isFile) file.inputStream().use { load(it) }
}
val signingProperties = listOf("handsonar.keystore", "handsonar.keyAlias", "handsonar.storePassword", "handsonar.keyPassword")
val blankSigning = signingProperties.filter { localProperties.containsKey(it) && localProperties.getProperty(it).isBlank() }
if (blankSigning.isNotEmpty()) {
    throw GradleException("local.properties의 ${blankSigning.joinToString(", ")} 값이 비어 있습니다: 값을 적거나 그 줄을 지우세요(비우면 다른 키로 서명됩니다)")
}
val handSonarKeystore: String? = localProperties.getProperty("handsonar.keystore")?.trim()
if (handSonarKeystore == null && signingProperties.drop(1).any { localProperties.containsKey(it) }) {
    throw GradleException("local.properties에 handsonar.keystore 없이 별칭·암호만 있습니다: 키 파일 경로를 적거나 그 줄을 지우세요")
}

subprojects {
    pluginManager.withPlugin("com.android.application") {
        val path = handSonarKeystore ?: return@withPlugin
        val store = rootProject.file(path)
        if (!store.isFile) throw GradleException("local.properties의 handsonar.keystore 파일이 없습니다: $path")
        extensions.configure<ApplicationExtension> {
            val shared = signingConfigs.create("handsonar") {
                storeFile = store
                storePassword = localProperties.getProperty("handsonar.storePassword", "android")
                keyAlias = localProperties.getProperty("handsonar.keyAlias", "androiddebugkey")
                keyPassword = localProperties.getProperty("handsonar.keyPassword", "android")
            }
            buildTypes.matching { it.name == "debug" }.configureEach { signingConfig = shared }
        }
    }
}
