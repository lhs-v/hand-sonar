package com.watchpose.handsonar.shared

/**
 * "워치 다시 참" 표시의 규칙(전송 설계 §8, 계획 6 검토 고침 M1·M10). [on]이면 다음 세션 meta.json `rewear`가 참이다. [K]는 세션을 가리는 열쇠
 * (폰은 세션 폴더)다. PC(`dataset.wear_episodes`)는 rewear가 참인 세션마다 새 착용이 시작해 다음 표시까지 간다고 보고, 휴지통·끝나지 않은 세션의
 * 표시도 경계로 본다 — 그래서 한 번 다시 찬 것에 표시가 두 세션에 남지 않게 한다.
 * - 사람이 [toggle]로 켜고 끈다(다시 찬 일을 사람이 정했으니 아래 "방금 세션" 추적도 지운다).
 * - 세션을 닫을 때([closed]): 표시로 적은 세션이 보관되면 끈다(다시 찬 뒤 첫 보관 세션이 생겼다). 취소한 세션(휴지통)이면 켜 둔다.
 * - 다음 세션을 시작하기 전에 그 세션을 휴지통으로 보내면(되돌리기·기록 화면 휴지통, [trashed]) 다시 켠다: 첫 보관 세션은 다음 세션이 된다.
 * - 그렇게 다시 켠 뒤 그 세션을 되살리면([restored]) 다시 끈다: 그 세션이 다시 첫 보관 세션이다.
 * - 다음 세션을 시작하면([started]) 앞 세션은 이 규칙에서 빠진다: 그 뒤 세션은 같은 착용이라, 앞 세션을 버리고 표시를 다시 켜면 경계가 둘이 된다.
 * 한 스레드(폰은 주 스레드)에서만 부른다. 켬·끔은 부른 쪽이 바로(commit) 남긴다.
 */
class RewearMark<K : Any>(on: Boolean = false) {
    var on: Boolean = on
        private set

    /** 표시로 적고 보관한 방금 세션(다음 세션을 시작하기 전까지). */
    private var saved: K? = null

    /** 휴지통으로 보내 표시를 다시 켠 방금 세션(다음 세션을 시작하기 전까지). */
    private var reopened: K? = null

    fun toggle() {
        on = !on
        saved = null
        reopened = null
    }

    /** 세션을 시작한다: 그 세션 meta.json `rewear`에 적을 값을 돌려준다. */
    fun started(): Boolean {
        saved = null
        reopened = null
        return on
    }

    /** 세션을 닫았다: [marked]는 그 세션 meta의 rewear, [kept]는 보관(아니면 취소로 휴지통). */
    fun closed(session: K, marked: Boolean, kept: Boolean) {
        if (marked && kept) {
            on = false
            saved = session
        }
    }

    /** 세션을 휴지통으로 보냈다(되돌리기, 기록 화면). */
    fun trashed(session: K) {
        if (session != saved) return
        saved = null
        reopened = session
        on = true
    }

    /** 휴지통의 세션을 되살렸다. */
    fun restored(session: K) {
        if (session != reopened) return
        reopened = null
        saved = session
        on = false
    }
}
