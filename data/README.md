# 수집 데이터 묶음

저장소 주인 한 사람이 왼손에 갤럭시 워치8(SM-L320)을 차고, 폰(갤럭시 Z 폴드8) 정면 카메라 앞에서 모은 세션 10개(2026-10-27–28)입니다. 처리 신호(에코 프로파일)와 손 관절, 그리고 그것을 맞추는 데 필요한 기록만 담았습니다. `tools/export_data.py`가 만들고, `tools/unpack_data.py`가 학습 도구가 읽는 폴더로 되돌립니다. 연구 현황과 결과는 [docs/research-status.md](../docs/research-status.md)에 있습니다.

## 넣지 않은 것과 까닭

| 뺀 것 | 까닭 |
|---|---|
| 영상(`video.mp4`)·이미지 | 얼굴과 방이 찍힌다. |
| 원본 소리: 워치 `part-NNN.wav`, 폰 `phone_audio.wav` | 18–21 kHz 처프만이 아니라 방 안의 들리는 소리(말소리 포함)가 녹음돼 있다. 대신 18–21 kHz 대역 통과 뒤 계산한 에코 프로파일을 넣었다(말소리가 남지 않는다). 공개 WatchHand 데이터도 같은 이유로 프로파일만 공개했다. |
| 센서 기록(`sensors-NNN.csv`, `phone_sensors-NNN.csv`) | 지금 처리·학습에 쓰지 않는다. 필요하면 따로 묶는다. |
| 학습 결과(모델·지표) | 다시 만들 수 있다(아래 "쓰는 법"). |

원본 소리가 없으므로 표지음 맞춤(`align.py`)과 프로파일 계산(`profiles.py`)은 다시 할 수 없습니다. 그 결과(`align.json`, `echo.npz`)를 그대로 씁니다. 지연 확인(`verify_sync.py`), 짝짓기(`pairs.py`), 신호 잣대(`signal_check.py`), 학습(`train.py`)은 되돌린 폴더에서 다시 돌릴 수 있습니다.

## 구조

```
data/
  sessions.csv                 세션마다 한 줄(아래 표)
  sessions/<세션 yyyyMMdd-HHmmss>/
    meta.json                  폰 세션 기록(카메라, 손, 시작·끝 시각, 안내 시간표 설정, 통계, 세션 정보, 다시 참)
    prompts.csv                안내 동작 시간표(index, segment, classId, name, startNs, endNs, flag)
    landmarks.csv.gz           카메라 프레임마다 손 관절(아래)
    rec.json                   짝지은 워치 녹음의 기록(처프 창·진폭, 녹음 음원, 녹음 볼륨, 녹음·재생 시각)
    align.json                 표지음으로 두 기기 시계를 맞춘 결과(method "two" = 시작·끝 표지음 둘)
    sync.json                  카메라–소리 남은 지연(lagMs)과 상관 봉우리(peak), 쓸 수 있는가(usable)
    pairs.csv.gz               카메라 프레임 → 프로파일 줄 대응(맞춤 확인을 통과한 세션만)
    signal.json                신호 잣대(signal_check.py)
    echo.npz                   에코 프로파일(아래)
  recordings/<녹음>/            세션 없는 워치 시험 녹음(처프 창·음원 판단의 근거)
    rec.json, echo.npz         앞 최대 60 s의 프로파일(profiles.compute와 같은 처리)
```

## 파일 형식과 규약

**echo.npz**
- `raw` (230, R) float32: 정렬된 에코 프로파일의 칸 250–479(`bin0` = 250).
  - 워치 녹음 600샘플(12.5 ms)마다 한 줄이고, 줄은 처프 한 번이다.
  - 만드는 법(`tools/profiles.py`): 18–21 kHz 5차 Butterworth 대역 통과(앞뒤로) → 송신 처프와 원형 상호상관 → 직접 경로(스피커 → 마이크)를 칸 300에 맞춤 → 시간 방향 3줄 중앙값.
  - 값은 부호 있는 상관이다(포락선이 아니다).
  - 칸 하나는 표본 하나(1/48 kHz)이고, 반사까지의 거리로 3.57 mm다(왕복이라 343 m/s ÷ 48 kHz ÷ 2). 칸 301–360이 손목에서 약 0–21 cm(손 자리)다.
- 차분 프로파일은 넣지 않았다. `|raw[:, t+1]| − |raw[:, t]|`로 다시 만든다(profiles.py와 같은 식, 열 t는 줄 t와 t+1 사이).
- `rows_ns` (R,): 줄 가운데 시각을 폰 시계(elapsedRealtime ns)로 옮긴 값(표지음 맞춤).
- `rows_frame` (R,): 그 줄이 시작하는 워치 녹음 표본 번호.
- `rows_direct` (R,): 정렬 전 직접 경로 칸.
- `rows_ok` (R,): 직접 경로가 있고, 따라간 칸이 튀거나 옮겨 간 둘레가 아닌 줄. 학습 창은 97줄이 모두 ok일 때만 쓴다.
- `rows_drift` (R,): 포락선 최대 칸 − 따라간 칸.
- `recording`: 워치 녹음 이름.

**landmarks.csv(.gz)**: 130열.
- 열 구성: `frameNs, hands, handedness, score`, 화면 좌표 `x0,y0,z0 … x20,y20,z20`, 세계 좌표 `wx0,wy0,wz0 … wx20,wy20,wz20`.
- MediaPipe Hand Landmarker(Tasks 0.10.35, float16 모델, GPU)를 폰 정면 카메라에 돌린 결과다.
  - **입력은 좌우를 뒤집은 거울 그림**이다. 실제 손 모양은 x → 1 − x, wx → −wx로 되돌린다.
  - `handedness`는 앱이 바로잡은 실제 손 이름이다(이 데이터는 모두 Left = 실제 왼손).
  - 추적하는 손이 없는 프레임은 handedness부터 빈 칸이다.
- `frameNs`는 카메라 센서 시각(폰 elapsedRealtime, REALTIME 기준)이다. 카메라는 실제로 초당 약 20프레임이다(간격 33/67 ms).
- 세계 좌표는 미터이고 원점은 손 중심이다. 학습 목표는 `tools/joints.py`가 이 좌표에서 만든다: 거울 되돌림, 손목 원점, 손바닥 틀, 손목–새끼 MCP = 1인 단위 손, 점 1–20.

**pairs.csv(.gz)**: 프레임마다 한 줄.
- 열 구성: `frame, frameNs, soundNs(= frameNs − lagMs), row(가장 가까운 프로파일 줄), dtMs, hand(쓸 손), direct(그 줄이 ok), prompt(prompts.csv 행, −1 = 없음), classId, flag, edge(동작 경계 0.5 s), video(영상 프레임 번호)`.

**sessions.csv**: 세션, 날짜, 워치 녹음, 볼륨, 처프 창, 녹음 음원, 손, 다시 참, 자세, 팔 받침, 줄 조임, 프레임, 길이(s), 지연(ms), 맞춤 상관, 학습에 쓸 수 있는가, 대응표 있음, 신호 잣대 넷(`corr0` 속도–세기 상관, `contrast` 움직임 대비, `peakWidthBins` 직접 경로 반높이 폭, `peakJitter90Bins` 줄 사이 흔들림 90%).

## 세션

모두 워치 처프 창 hann, 녹음 음원 VOICE_RECOGNITION, 진폭 0.5, 왼손, 앉은 자세입니다. 팔 받침은 기록하지 않았습니다. 세션마다 워치를 거의 매번 풀었다 다시 찼지만 "다시 참" 표시는 켜지 않았습니다.

| 세션 | 워치 볼륨 | 맞춤 상관 · 지연 | 학습에 쓸 수 있음 | 속도–세기 상관 | 움직임 대비 | 직접 경로 폭(칸) |
|---|---|---|---|---|---|---|
| 20261027-235412 | 11 | 0.178 · +7.8 ms | 예 | 0.23 | 3.5 | 42 |
| 20261028-193708 | 15 | 0.077 · +11.9 ms | 아니오(문턱 0.1 아래, 대응표 없음) | 0.12 | 2.5 | 37 |
| 20261028-195736 | 13 | 0.142 · +19.1 ms | 예 | 0.17 | 3.7 | 36 |
| 20261028-200730 | 13 | 0.174 · +18.0 ms | 예 | 0.22 | 3.3 | 80 |
| 20261028-201138 | 13 | 0.148 · +25.8 ms | 예 | 0.22 | 3.1 | 77 |
| 20261028-214248 | 13 | 0.157 · +27.5 ms | 예 | 0.24 | 2.8 | 82 |
| 20261028-214654 | 13 | 0.188 · +11.4 ms | 예 | 0.25 | 3.1 | 85 |
| 20261028-215102 | 13 | 0.151 · +16.5 ms | 예 | 0.19 | 3.1 | 42 |
| 20261028-215513 | 13 | 0.277 · +16.7 ms | 예 | 0.28 | 2.9 | 37 |
| 20261028-215926 | 13 | 0.305 · +3.6 ms | 예 | 0.36 | 3.2 | 34 |

같은 잣대로 잰 공개 WatchHand(참가자 9 왼손 10세션)는 속도–세기 상관 0.39–0.70, 움직임 대비 4.6–10.9, 직접 경로 폭 21칸입니다.

`recordings/`의 시험 녹음(손 정답 없음):

| 녹음 | 처프 창 | 볼륨 | 녹음 음원 | 무엇의 근거인가 |
|---|---|---|---|---|
| 20261027-202740 | 창 없음 | 15 | VOICE_RECOGNITION | 창 없음이면 직접 경로 폭 20칸(공개 데이터와 같음) |
| 20261027-203239 | 창 없음 | 11 | VOICE_RECOGNITION | 같음(21칸) |
| 20261027-203600 | hann | 11 | VOICE_RECOGNITION | hann이면 넓음 |
| 20261027-205800 | hann | 11 | VOICE_RECOGNITION | 같음 |
| 20261028-233631 | hann | 13 | UNPROCESSED | 같은 자리 1분씩: 두 음원의 세기·대역·직접 경로가 거의 같음 |
| 20261028-233738 | hann | 13 | VOICE_RECOGNITION | 위와 짝 |

## 쓰는 법

```bash
python -I tools/unpack_data.py ../hand-sonar-unpacked
python -I tools/signal_check.py --data ../hand-sonar-unpacked
python -I tools/train.py --data ../hand-sonar-unpacked --mode blocks --run blocks-9
```

`unpack_data.py`는 세션마다 `phone/sessions/<세션>/`과 `processed/<세션>/`을 만듭니다. `profiles.npy`(600칸, 묶음에 없는 칸은 0)와 `diff_profiles.npy`를 다시 만들고, `sync.json`의 입력 지문을 새 파일로 다시 잽니다. 되돌린 폴더에서 `dataset.build`가 만드는 학습 자료는 원래 처리 폴더와 프레임·정답이 같고, 입력 창은 float32 저장 정밀도 안(최대 차이 1.4e-4)에서 같습니다(2026-10-29 확인).

도구 없이 읽기:

```python
import csv, gzip
import numpy as np
s = "data/sessions/20261028-215926"
e = np.load(f"{s}/echo.npz")
raw = e["raw"]                                   # (230, R), 칸 250–479
diff = np.abs(raw[:, 1:]) - np.abs(raw[:, :-1])  # 차분 프로파일
with gzip.open(f"{s}/landmarks.csv.gz", "rt", encoding="utf-8") as f:
    marks = list(csv.DictReader(f))              # 프레임마다 관절(wx0…wz20 = 세계 좌표 m, 거울 그림 기준)
with gzip.open(f"{s}/pairs.csv.gz", "rt", encoding="utf-8") as f:
    pairs = list(csv.DictReader(f))              # 프레임 → 프로파일 줄(row)
# 프레임 n의 모델 입력: 원본 줄 row−95…row, 차분 열 row−96…row−1, 칸 300–359 = raw[50:110]
```

## 출처

저장소 주인이 자기 손으로 모은 연구용 데이터입니다. 얼굴·영상·원본 소리는 없습니다. 공개 WatchHand 데이터(Cornell eCommons, CC BY 4.0, doi:10.7298/qf1v-j805)는 이 묶음에 들어 있지 않습니다. 받는 법과 우리 형식으로 옮기는 법은 [docs/public-watchhand-data.md](../docs/public-watchhand-data.md)와 `tools/experiments/public_to_ours.py`에 있습니다.
