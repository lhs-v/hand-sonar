package com.watchpose.handsonar.mobile.transfer

import android.net.Uri
import com.google.android.gms.tasks.Task
import com.google.android.gms.tasks.Tasks
import com.google.android.gms.wearable.ChannelClient
import com.google.android.gms.wearable.Wearable
import com.google.android.gms.wearable.WearableListenerService
import com.watchpose.handsonar.mobile.HandSonarPhoneApp
import com.watchpose.handsonar.shared.TransferProtocol

/**
 * 워치 녹음 받기(전송 설계 §2). 앱이 꺼져 있어도 데이터 레이어가 이 서비스를 깨워 부른다(매니페스트의 REQUEST_RECEIVED·CHANNEL_EVENT,
 * 경로로 이 앱의 것만). 콜백은 이 서비스의 작업 스레드에서 차례로 불린다(주 스레드가 아님). Task는 기다리지 않는다.
 * - 목록 요청([TransferProtocol.OFFER_PATH]): 아직 없거나 크기가 다른 파일 이름을 답한다. 빈 답이 그 녹음을 "다 받음"으로 확인하는 것이다.
 * - 채널(파일 하나): 같은 크기가 이미 있으면 받지 않고 HAVE로 닫는다. 아니면 `.part`로 받는다(receiveFile — 쓰기는 Play 서비스가 한다).
 *   입력이 닫히면 크기·WAV 머리말을 확인해 이름을 바꾸고 결과 코드로 채널을 닫는다. 워치는 그 코드를 보고 다음 파일로 간다.
 */
class WatchReceiverService : WearableListenerService() {
    private val inbox: WatchInbox get() = (application as HandSonarPhoneApp).inbox
    private val channels: ChannelClient by lazy { Wearable.getChannelClient(this) }

    override fun onRequest(nodeId: String, path: String, request: ByteArray): Task<ByteArray>? {
        if (path != TransferProtocol.OFFER_PATH) return null
        val offer = TransferProtocol.decodeOffer(request)
            ?: return Tasks.forException<ByteArray>(IllegalArgumentException("녹음 목록을 못 읽음"))
        return Tasks.forResult(TransferProtocol.encodeNeed(inbox.need(offer)))
    }

    override fun onChannelOpened(channel: ChannelClient.Channel) {
        val ref = TransferProtocol.parseChannelPath(channel.path)
        if (ref == null) {
            channels.close(channel, TransferProtocol.CODE_BAD_PATH)
            return
        }
        val part = inbox.begin(ref, channel)
        if (part == null) {
            channels.close(channel, TransferProtocol.CODE_HAVE)
            return
        }
        channels.receiveFile(channel, Uri.fromFile(part), false).addOnFailureListener {
            inbox.abandon(channel)
            channels.close(channel, TransferProtocol.CODE_IO)
        }
    }

    override fun onInputClosed(channel: ChannelClient.Channel, closeReason: Int, appSpecificErrorCode: Int) {
        val ref = TransferProtocol.parseChannelPath(channel.path) ?: return
        val code = inbox.finish(ref, channel) ?: return
        channels.close(channel, code)
    }

    override fun onChannelClosed(channel: ChannelClient.Channel, closeReason: Int, appSpecificErrorCode: Int) {
        inbox.abandon(channel)
    }
}
