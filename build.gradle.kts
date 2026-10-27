plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.kotlin.jvm) apply false // KGP 2.4.21을 빌드 클래스패스에 고정(센서 허브와 같음)
    alias(libs.plugins.kotlin.compose) apply false
}
