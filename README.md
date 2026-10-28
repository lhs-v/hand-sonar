# hand-sonar

갤럭시 워치의 스피커와 마이크로 18–21 kHz 초음파 소나를 녹음하고, 폰 정면 카메라로 손 관절 정답을 함께 모으는 안드로이드 앱 두 개입니다. 손 관절 추적 논문 WatchHand([arXiv 2602.21610](https://arxiv.org/abs/2602.21610))를 따라 해 보는 개인 연구용이며, 논문 저자와는 관계가 없습니다.

## 구성

| 폴더 | 내용 |
|---|---|
| `wear/` | 워치 앱. 18–21 kHz 처프(600 샘플, 48 kHz)를 계속 내면서 마이크를 녹음합니다. 1분 조각 WAV와 기록(`rec.json`)을 남깁니다. |
| `mobile/` | 폰 앱. 정면 카메라와 MediaPipe Hand Landmarker로 손 관절 21점을 프레임마다 적고(`landmarks.csv`), 같은 프레임을 영상(`video.mp4`)으로 남깁니다. 안내 동작(손가락 굽히기, ASL 숫자, 손목, 자유 동작)을 시간표대로 띄우고(`prompts.csv`), 세션 시작과 끝에 표지음을 냅니다. |
| `shared/` | 두 앱이 같이 쓰는 순수 Kotlin 코드입니다(처프·표지음 규격, 검출기, 파일 형식, 세션 진행). JVM 시험이 있습니다. |
| `tools/` | PC용 파이썬 도구입니다. 받기(`pull.py`), 녹음 점검(`check_recording.py`), 표지음으로 시간 맞춤(`align.py`), 에코 프로파일(`profiles.py`), 카메라와 맞춤 검증(`verify_sync.py`), 카메라 프레임↔프로파일 줄 대응표(`pairs.py`), 공개 데이터와 나란히 보기(`compare.py`). |

## 두 기기 시간 맞추기

두 앱은 서로 연결하지 않습니다. 폰이 세션 시작과 끝에 표지음(21.5→23 kHz, 40 ms)을 내고, PC에서 워치 녹음 안의 표지음을 찾아 두 기기의 시계를 맞춥니다.

## 빌드

- JDK 17 이상과 Android SDK 36이 필요합니다. `local.properties`에 `sdk.dir`를 적습니다(Android Studio가 만들어 줍니다).
- `./gradlew :wear:assembleDebug :mobile:assembleDebug`
- 폰 앱이 쓰는 MediaPipe 손 모델(`hand_landmarker.task`, 7.8 MB)은 저장소에 없습니다. 첫 빌드 때 구글 공식 주소에서 받아 SHA-256을 확인합니다. 인터넷이 없으면 `-PhandModelUrl=file:///경로/hand_landmarker.task`로 가진 파일을 씁니다.
- 시험: `./gradlew :shared:test`, `python -I -m unittest discover -s tools/tests -t tools`

## PC 처리

파이썬 3.13과 numpy·scipy·matplotlib을 씁니다. 기기에서 받은 파일은 `python -I`로만 읽습니다. 데이터는 저장소 옆 `../hand-sonar-data`에 받고(다른 곳이면 `--data`), 세션마다 결과를 `../hand-sonar-data/processed/<세션>/`에 씁니다. 세션 이름을 주지 않으면 보관된(`kept`) 끝난 세션을 모두 처리합니다.

1. `python -I tools/pull.py` — 두 기기의 녹음·세션 폴더를 adb로 받습니다. 기기는 모델(워치 SM-L320, 폰 SM-F971N)로 찾고, `--watch`·`--phone`으로 serial을 줄 수 있습니다.
2. `python -I tools/align.py` — 폰 표지음으로 워치 샘플 번호와 폰 시각을 맞춥니다(`align.json`).
3. `python -I tools/profiles.py` — 공개 데이터(WatchHand)와 같은 처리로 에코 프로파일을 만듭니다(`profiles.npy`, `diff_profiles.npy`, `rows.npz`).
4. `python -I tools/verify_sync.py` — 카메라 관절 속도와 손 거리 소리 세기로 남은 지연을 잽니다(`sync.json`, `sync.png`). 지연이 1프레임(33 ms) 밖이거나 상관 봉우리가 문턱(0.1) 아래면 맞춤을 확인하지 못한 세션으로 적고 끝 코드 1로 끝납니다.
5. `python -I tools/pairs.py` — 카메라 프레임마다 프로파일 줄을 짝짓습니다(`pairs.csv`). 지연이 1프레임 안이고 상관이 문턱 이상인 세션만, 그리고 `sync.json`이 지금의 `align.json`·`rows.npz`·`landmarks.csv`에서 잰 것일 때만 짝짓습니다. 아니면 `pairs.csv`를 쓰지 않고(옛 것은 `pairs.stale.csv`로 바꿈) 끝 코드 1로 끝납니다.
6. `python -I tools/compare.py <세션>` — 공개 데이터 왼손 표본과 나란히 그린 그림입니다(`compare.png`, 표본 위치는 `--public`).

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
