package com.watchpose.handsonar.mobile.ui

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.watchpose.handsonar.mobile.HandSonarPhoneApp
import com.watchpose.handsonar.mobile.session.Screen
import com.watchpose.handsonar.mobile.session.SessionController

private val PERMISSIONS = arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)

/**
 * 권한을 받고 장치를 연 뒤 지금 화면을 그린다. [activity]는 카메라를 이을 수명 주인이자 화면 정보(가장 짧은 변, 회전)다.
 * 워치에서 받은 녹음 상태(전송 설계 §3)는 앱 객체의 받기 상태를 그대로 준비·기록 화면에 넘긴다.
 */
@Composable
fun App(controller: SessionController, activity: ComponentActivity) {
    val ui by controller.ui.collectAsStateWithLifecycle()
    val inbox by (activity.application as HandSonarPhoneApp).inbox.status.collectAsStateWithLifecycle()
    fun open() = controller.openDevices(
        activity,
        activity.resources.configuration.smallestScreenWidthDp,
        ContextCompat.getDisplayOrDefault(activity).rotation,
    )
    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) {
        if (PERMISSIONS.all { p -> granted(activity, p) }) open() else controller.onPermissionsDenied()
    }
    LaunchedEffect(Unit) {
        if (PERMISSIONS.all { granted(activity, it) }) open() else launcher.launch(PERMISSIONS)
    }
    Surface(Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
        // targetSdk 36·Android 17은 화면 가장자리까지 그리는 방식(edge-to-edge)이라 시스템 막대가 내용을 덮는다(2026-10-27 실기기:
        // 상태 막대가 미리보기를 덮음). 내용은 시스템 막대·카메라 구멍을 피해 안쪽에 그리고, 배경(Surface)만 가장자리까지 깐다.
        Box(Modifier.fillMaxSize().safeDrawingPadding()) {
            when (ui.screen) {
                Screen.READY -> ReadyScreen(ui, controller, inbox, onRequestPermission = { launcher.launch(PERMISSIONS) })
                Screen.COLLECT -> CollectScreen(ui, controller)
                Screen.SUMMARY -> SummaryScreen(ui, controller)
                Screen.RECORDS -> RecordsScreen(ui, controller, inbox)
            }
        }
    }
}

private fun granted(context: Context, permission: String): Boolean =
    ContextCompat.checkSelfPermission(context, permission) == PackageManager.PERMISSION_GRANTED
