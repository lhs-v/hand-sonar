# hand-sonar

갤럭시 워치의 스피커와 마이크로 18–21 kHz 초음파 소나를 녹음하고, 폰 정면 카메라로 손 관절 정답을 함께 모으는 안드로이드 앱 두 개입니다. 손 관절 추적 논문 WatchHand([arXiv 2602.21610](https://arxiv.org/abs/2602.21610))를 따라 해 보는 개인 연구용이며, 논문 저자와는 관계가 없습니다.

## 구성

| 폴더 | 내용 |
|---|---|
| `wear/` | 워치 앱. 18–21 kHz 처프(600 샘플, 48 kHz)를 계속 내면서 마이크를 녹음합니다. 1분 조각 WAV와 기록(`rec.json`)을 남깁니다. |
| `mobile/` | 폰 앱. 정면 카메라와 MediaPipe Hand Landmarker로 손 관절 21점을 프레임마다 적고(`landmarks.csv`), 같은 프레임을 영상(`video.mp4`)으로 남깁니다. 안내 동작(손가락 굽히기, ASL 숫자, 손목, 자유 동작)을 시간표대로 띄우고(`prompts.csv`), 세션 시작과 끝에 표지음을 냅니다. |
| `shared/` | 두 앱이 같이 쓰는 순수 Kotlin 코드입니다(처프·표지음 규격, 검출기, 파일 형식, 세션 진행). JVM 시험이 있습니다. |
| `tools/` | PC용 파이썬 도구입니다(워치 녹음 점검 `check_recording.py`). |

## 두 기기 시간 맞추기

두 앱은 서로 연결하지 않습니다. 폰이 세션 시작과 끝에 표지음(21.5→23 kHz, 40 ms)을 내고, PC에서 워치 녹음 안의 표지음을 찾아 두 기기의 시계를 맞춥니다.

## 빌드

- JDK 17 이상과 Android SDK 36이 필요합니다. `local.properties`에 `sdk.dir`를 적습니다(Android Studio가 만들어 줍니다).
- `./gradlew :wear:assembleDebug :mobile:assembleDebug`
- 폰 앱이 쓰는 MediaPipe 손 모델(`hand_landmarker.task`, 7.8 MB)은 저장소에 없습니다. 첫 빌드 때 구글 공식 주소에서 받아 SHA-256을 확인합니다. 인터넷이 없으면 `-PhandModelUrl=file:///경로/hand_landmarker.task`로 가진 파일을 씁니다.
- 시험: `./gradlew :shared:test`, `python -m unittest discover -s tools/tests -t tools`

## 설치와 사용

1. 워치(무선 디버깅)와 폰에 APK를 설치합니다.
   - `adb -s <워치> install -r wear/build/outputs/apk/debug/wear-debug.apk`
   - `adb -s <폰> install -r mobile/build/outputs/apk/debug/mobile-debug.apk`
2. 워치 앱에서 녹음을 시작합니다. 미디어 볼륨이 0이면 시작하지 않습니다.
3. 폰을 가로로 세우고(폴드는 펼쳐서) 정면 카메라가 손을 보게 둡니다. 표지음이 나가도록 폰 미디어 볼륨을 올립니다. 준비 화면에서 추적할 손(워치를 찬 손)을 고르고 시작합니다.
4. 화면 안내대로 손을 움직입니다. 끝나면 요약이 나옵니다.
5. 데이터는 각 기기의 `Android/data/com.watchpose.handsonar/files/` 아래에 남습니다(워치 `recordings/`, 폰 `sessions/`). `adb pull`로 PC에 가져옵니다.

확인한 기기는 갤럭시 워치8과 갤럭시 Z 폴드8뿐입니다. 다른 워치는 스피커가 18–21 kHz를 내고 마이크가 48 kHz로 녹음해야 합니다. 폰 앱은 arm64 기기만 지원합니다.

## 데이터

수집한 녹음과 영상(얼굴이 찍힘)은 이 저장소에 넣지 않습니다.
