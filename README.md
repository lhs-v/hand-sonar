# hand-sonar

갤럭시 워치의 스피커와 마이크로 18–21 kHz 초음파 소나를 녹음하고, 폰 정면 카메라로 손 관절 정답을 함께 모으는 안드로이드 앱 두 개입니다. 손 관절 추적 논문 WatchHand([arXiv 2602.21610](https://arxiv.org/abs/2602.21610))를 따라 해 보는 개인 연구용이며, 논문 저자와는 관계가 없습니다.

## 구성

| 폴더 | 내용 |
|---|---|
| `wear/` | 워치 앱. 18–21 kHz 처프(600 샘플, 48 kHz)를 계속 내면서 마이크를 녹음합니다. 1분 조각 WAV와 기록(`rec.json`)을 남기고, 녹음이 멈추면 블루투스로 폰에 넘깁니다. |
| `mobile/` | 폰 앱. 정면 카메라와 MediaPipe Hand Landmarker로 손 관절 21점을 프레임마다 적고(`landmarks.csv`), 같은 프레임을 영상(`video.mp4`)으로 남깁니다. 안내 동작(손가락 굽히기, ASL 숫자, 손목, 자유 동작)을 시간표대로 띄우고(`prompts.csv`), 세션 시작과 끝에 표지음을 냅니다. 워치가 넘긴 녹음을 받아 두고, 기록 화면에서 zip으로 내보냅니다. |
| `shared/` | 두 앱이 같이 쓰는 순수 Kotlin 코드입니다(처프·표지음 규격, 검출기, 파일 형식, 세션 진행). JVM 시험이 있습니다. |
| `tools/` | PC용 파이썬 도구입니다. 받기(`pull.py`), 녹음 점검(`check_recording.py`), 표지음으로 시간 맞춤(`align.py`), 에코 프로파일(`profiles.py`), 카메라와 맞춤 검증(`verify_sync.py`), 카메라 프레임↔프로파일 줄 대응표(`pairs.py`), 공개 데이터와 나란히 보기(`compare.py`). |

## 두 기기 시간 맞추기

시간 맞추기에는 두 앱의 연결을 쓰지 않습니다. 폰이 세션 시작과 끝에 표지음(21.5→23 kHz, 40 ms)을 내고, PC에서 워치 녹음 안의 표지음을 찾아 두 기기의 시계를 맞춥니다.

## 워치 녹음을 폰으로 넘기기

워치를 adb로 붙일 수 없는 PC에서도 데이터를 모을 수 있게, 워치가 녹음을 블루투스(Wear OS 데이터 레이어)로 짝지은 폰에 넘깁니다. 두 앱이 같은 키로 서명돼 있어야 합니다(아래 "서명").

- 워치는 녹음이 멈추면(정지 이유와 상관없이) 그 녹음을 폰으로 보냅니다. 워치 앱 화면이 보일 때도 아직 못 보낸 녹음을 다시 보냅니다. 녹음 중에는 보내지 않습니다(녹음을 시작하면 보내던 것을 끊습니다). 녹음하지 않을 때 워치 화면에 보내는 상태가 한 줄 나옵니다.
- 파일마다 채널 하나로 보냅니다. 폰은 `.part`로 받아 크기와 WAV 머리말을 확인한 뒤 원래 이름으로 바꿉니다. 끊기면 그 파일을 처음부터 다시 보냅니다(1분씩 쉬며 5번까지, 그 뒤에는 다음 녹음이 멈출 때나 워치 앱을 열 때). 폰이 녹음의 모든 파일을 확인하면 워치는 그 녹음을 "보냄"으로 적습니다. 워치의 녹음은 지우지 않습니다.
- 폰은 받은 녹음을 `Android/data/com.watchpose.handsonar/files/watch-recordings/<녹음>/`에 워치와 같은 모양(`part-NNN.wav`, `rec.json`)으로 둡니다. 준비 화면과 기록 화면에 받은 녹음 수, 마지막으로 받은 시각, 받는 중 진행률이 나옵니다.
- 10분 녹음은 약 58 MB입니다. 블루투스만으로는 몇 분 넘게 걸릴 수 있습니다.

## 내보내기 (USB 파일 전송만 되는 PC)

폰 기록 화면의 "내보내기"는 보관한 세션과 받은 워치 녹음을 `Download/HandSonar/hand-sonar-<날짜-시각>.zip` 하나로 저장합니다. zip 안은 `phone/sessions/<세션>/…`, `watch/recordings/<녹음>/…`라서 PC의 `hand-sonar-data`에 그대로 풀면 `pull.py`로 받은 것과 같습니다. "새 것만"은 지난번에 내보낸 뒤 파일 수·크기·시각이 그대로인 폴더를 뺍니다. 받는 중인 `.part`는 넣지 않습니다.

## 빌드

- JDK 17 이상과 Android SDK 36이 필요합니다. `local.properties`에 `sdk.dir`를 적습니다(Android Studio가 만들어 줍니다).
- `./gradlew :wear:assembleDebug :mobile:assembleDebug`
- 폰 앱이 쓰는 MediaPipe 손 모델(`hand_landmarker.task`, 7.8 MB)은 저장소에 없습니다. 첫 빌드 때 구글 공식 주소에서 받아 SHA-256을 확인합니다. 인터넷이 없으면 `-PhandModelUrl=file:///경로/hand_landmarker.task`로 가진 파일을 씁니다.
- 시험: `./gradlew :shared:test`, `python -I -m unittest discover -s tools/tests -t tools`

### 서명

워치 앱과 폰 앱은 applicationId가 같고(`com.watchpose.handsonar`) **같은 키로 서명돼야** 데이터 레이어로 이어집니다. 두 앱의 debug 빌드는 `local.properties`에 `handsonar.keystore`가 있으면 그 키로, 없으면 그 PC의 기본 디버그 키(`~/.android/debug.keystore`)로 서명합니다.

```properties
# 키 파일 경로(상대 경로는 저장소 기준). 윈도 경로는 / 로 쓰거나 \\ 로 씁니다.
handsonar.keystore=D:/keys/hand-sonar-debug.keystore
# 적지 않으면 디버그 키 기본값(androiddebugkey / android)
handsonar.keyAlias=androiddebugkey
handsonar.storePassword=android
handsonar.keyPassword=android
```

- 다른 PC에서 빌드해 지금 깔린 앱을 이어 쓰려면, 그 앱을 서명한 키 파일(처음 빌드한 PC의 `~/.android/debug.keystore`)을 옮겨 그 경로를 적습니다. 경로가 적혀 있는데 파일이 없으면 빌드가 멈춥니다.
- 키 파일은 저장소에 넣지 않습니다(`.gitignore`가 `*.keystore`·`*.jks`를 막습니다).
- **다른 서명으로 덮어 깔면 설치가 거부됩니다.** 그렇다고 앱을 지우고 다시 깔면 **워치에 쌓인 녹음도 함께 지워집니다**(앱 외부 저장소는 앱과 함께 지워집니다. 폰 앱을 지우면 폰의 세션과 받은 워치 녹음도 지워집니다). 지우기 전에 녹음을 폰이나 PC로 옮겨 두세요.

## PC 처리

파이썬 3.13과 numpy·scipy·matplotlib을 씁니다. 기기에서 받은 파일은 `python -I`로만 읽습니다. 데이터는 저장소 옆 `../hand-sonar-data`에 받고(다른 곳이면 `--data`), 세션마다 결과를 `../hand-sonar-data/processed/<세션>/`에 씁니다. 세션 이름을 주지 않으면 보관된(`kept`) 끝난 세션을 모두 처리합니다.

1. `python -I tools/pull.py` — 폰의 세션과 폰이 받은 워치 녹음, 워치 녹음을 adb로 받습니다. 폰을 먼저 받고(USB라 빠름), 워치에서는 폰에 없거나 크기가 다른 녹음만 받습니다. 워치가 없어도 됩니다. 기기는 모델(워치 SM-L320, 폰 SM-F971N)로 찾고, `--watch`·`--phone`으로 serial을 줄 수 있습니다. adb를 쓸 수 없는 PC는 폰의 "내보내기" zip을 `hand-sonar-data`에 풉니다.
2. `python -I tools/align.py` — 폰 표지음으로 워치 샘플 번호와 폰 시각을 맞춥니다(`align.json`).
3. `python -I tools/profiles.py` — 공개 데이터(WatchHand)와 같은 처리로 에코 프로파일을 만듭니다(`profiles.npy`, `diff_profiles.npy`, `rows.npz`).
4. `python -I tools/verify_sync.py` — 카메라 관절 속도와 손 거리 소리 세기로 남은 지연을 잽니다(`sync.json`, `sync.png`). 지연이 1프레임(33 ms) 밖이거나 상관 봉우리가 문턱(0.1) 아래면 맞춤을 확인하지 못한 세션으로 적고 끝 코드 1로 끝납니다.
5. `python -I tools/pairs.py` — 카메라 프레임마다 프로파일 줄을 짝짓습니다(`pairs.csv`). 지연이 1프레임 안이고 상관이 문턱 이상인 세션만, 그리고 `sync.json`이 지금의 `align.json`·`rows.npz`·`landmarks.csv`에서 잰 것일 때만 짝짓습니다. 아니면 `pairs.csv`를 쓰지 않고(옛 것은 `pairs.stale.csv`로 바꿈) 끝 코드 1로 끝납니다.
6. `python -I tools/compare.py <세션>` — 공개 데이터 왼손 표본과 나란히 그린 그림입니다(`compare.png`, 표본 위치는 `--public`).

## 설치와 사용

1. 워치(무선 디버깅)와 폰에 APK를 설치합니다. 두 APK는 같은 키로 서명한 것이어야 합니다(빌드 › 서명).
   - `adb -s <워치> install -r wear/build/outputs/apk/debug/wear-debug.apk`
   - `adb -s <폰> install -r mobile/build/outputs/apk/debug/mobile-debug.apk`
2. 워치 앱에서 녹음을 시작합니다. 미디어 볼륨이 0이면 시작하지 않습니다.
3. 폰을 가로로 세우고(폴드는 펼쳐서) 정면 카메라가 손을 보게 둡니다. 표지음은 앱이 낼 때만 미디어 볼륨을 잠깐 올렸다 되돌립니다. 이어폰·블루투스 오디오는 빼 두는 편이 안전합니다. 준비 화면에서 추적할 손(워치를 찬 손)을 고르고 시작합니다.
4. 화면 안내대로 손을 움직입니다. 끝나면 요약이 나옵니다.
5. 데이터는 각 기기의 `Android/data/com.watchpose.handsonar/files/` 아래에 남습니다(워치 `recordings/`, 폰 `sessions/`와 워치에서 받은 `watch-recordings/`). `pull.py`나 폰의 "내보내기"로 PC에 가져옵니다.

확인한 기기는 갤럭시 워치8과 갤럭시 Z 폴드8뿐입니다. 다른 워치는 스피커가 18–21 kHz를 내고 마이크가 48 kHz로 녹음해야 합니다. 폰 앱은 arm64 기기만 지원합니다.

## 데이터

수집한 녹음과 영상(얼굴이 찍힘)은 이 저장소에 넣지 않습니다.
