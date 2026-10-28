# 공개 WatchHand 데이터: 영상과 에코 프로파일의 시간 맞추기, 검증, 관절 정답 만들기

공개 데이터셋 *Data from: WatchHand: Enabling Continuous Hand Pose Tracking On Off-the-Shelf Smartwatches*(Cornell eCommons, CC BY 4.0)로 손 관절 모델을 학습하려는 사람을 위한 안내다. 이 프로젝트의 맥락을 모른다고 가정하고 쓴다. 이 데이터에는 관절 파일이 없고, 영상(정답)과 소리를 맞추는 README의 식은 글자 그대로 쓰면 틀린다. 이 문서는 영상 프레임을 에코 프로파일의 줄에 대응시키는 법, 그 대응을 확인하는 법과 한계, 영상에서 관절 정답을 만드는 법을 적는다. 2026-10-28에 썼다.

표기: **[확인]** 공개 데이터에서 직접 계산해 본 것, **[추정]** 정황에서 읽은 것, **[미확인]** 확인하지 못한 것. 확인한 범위는 Study 1의 참가자 9–12(`Study1_sub09-12.zip`, 8폴더 × 10세션 = 80세션)의 설정 파일과, 그 가운데 일부 세션 표본이다. 영상은 한 개만 디코딩해 봤다.

## 요약

1. 관절 파일은 데이터에 없다. mp4에 MediaPipe를 돌려 직접 만든다(§8).
2. 세션 `i`(0부터 센다)의 짝은 `audio00(i+1)` ↔ `ground_truth.videos[i]` ↔ 두 `syncing_poses[i]`다. 파일 이름 정렬이 아니라 `config.json`에서 이름으로 찾는다(§3).
3. `ground_truth.syncing_poses[i]`는 **영상 프레임 번호**, `audio.syncing_poses[i]`는 **에코 프로파일 줄 번호**(1줄 = 12.5 ms, 초당 80줄)다. README는 이 둘을 timestamp와 sample index라고 적었는데, 그렇게 읽으면 틀린다(§4).
4. 프레임 `v`의 줄: `echo_row(v) = audio_sync + round((frame_time[v] − frame_time[gt_sync]) × 80)`. `frame_time`은 앞에서부터 디코딩한 프레임 수만큼만 쓴다. 줄이 더 많으면 남는 줄은 끝에 있다(§5).
5. 카메라–소리 상호상관은 2.02 s 간격의 빗살(봉우리 여러 개) 모양이라, ±1 s 안의 봉우리 0.3은 맞다는 증거가 아니다. ±6 s 창에서 가운데 빗살을 보되, 이 기준은 짝이 맞는 세션에서 아직 확인하지 못했다(§6).

## 1. 목적과 현재 상태

**목적.** 영상 프레임(30 fps) 하나마다 그 시각의 에코 프로파일 창을 짝지어 학습 쌍(입력 = 소리, 정답 = 손 관절)을 만든다. 그러려면 (a) 프레임이 프로파일의 몇 번째 줄인지 알아야 하고(§5), (b) 그 대응이 맞는지 확인해야 하고(§6), (c) 영상에서 관절을 뽑아 정규화해야 한다(§8). 논문(arXiv 2602.21610, CHI 2026)은 동기를 "timestamp alignment"라고만 적었고, 동기에 쓴 동작이나 시계 맞추는 방법은 어디에도 없다.

**확인한 것.**

- **[확인]** `imu_config.json`의 `imu.syncing_poses`는 80세션 모두에서 `round(audio.syncing_poses × 0.0125 × IMU 표본율)`이다(Samsung 100 Hz는 ×1.25, Xiaomi 50 Hz는 ×0.625, 최대 차이 0). 저자들이 `audio.syncing_poses`를 12.5 ms 줄 번호로 다뤘다는 뜻이다(§4).
- **[확인]** README 식을 글자 그대로 쓰면 줄 번호가 약 1.4×10¹¹이 되어 프로파일과 겹치는 프레임이 없다. `audio_sync`만 sample로 읽으면 카메라가 2.6–5.5 s 어긋난다(§7).
- **[확인]** 디코딩해 본 영상 하나(참가자 9 왼손의 `videos[1]`)는 `frame_time.txt`가 5,076줄, mp4가 4,869프레임이고, 남는 207줄(6.9 s)은 끝에 있다(§2, §6.1).
- **[확인]** 짝이 맞는 세션 하나(참가자 9 오른손 세션 0)에서 지시 시간표를 §5 식으로 에코 시간에 옮기면, 마지막 손목 블록이 시작한 직후 소리 에너지가 뛴다. README의 sample 읽기로 옮기면 같은 블록이 3.6 s 앞에 놓여 맞지 않는다. 카메라 없이 본 0.5 s 눈금의 거친 확인이다(§6.1).
- **[확인]** 지시가 2.02 s마다 바뀌어 카메라–소리 상호상관은 2.02 s 간격 빗살이다. 세션 짝이 틀린 쌍에서도 ±1 s 봉우리가 0.34(+164 ms)로 나왔다(§6.2).
- **[확인]** 공개 영상(참가자 9 왼손)에 MediaPipe를 돌리면 4,869프레임 모두에서 손이 잡히고, 영상은 거울상으로 보인다(§8).

**확인하지 못한 것.**

- **[미확인] 짝이 맞는 세션에서의 카메라–소리 상호상관.** 우리 표본에서 카메라와 소리를 모두 가진 쌍은 짝이 틀린 것(참가자 9 왼손 `videos[1]`의 영상 + `audio001`, 이하 **우리 쌍**)뿐이었다. 데이터를 모두 가진 사람은 아무 세션에서나 §6.2를 돌려 볼 수 있다. 가장 먼저 해 볼 일이다.
- **[미확인] ±6 s 창에서 "가운데 빗살이 ±2 s 이웃보다 뚜렷이 높다"는 기준.** 짝이 틀린 쌍에서는 빗살이 0.30–0.34로 비슷해 이 기준이 걸러 낸다는 것만 봤다. 맞는 쌍에서 가운데가 얼마나 높아지는지, 문턱을 얼마로 잡을지는 모른다.
- **[미확인]** 남는 `frame_time` 줄이 끝에 있다는 결론은 영상 하나에서 나왔다. 다른 영상도 그런지, 줄이 모자란 영상이 있는지 모른다.
- **[미확인]** `syncing_poses`가 0부터 센 번호인지. 1부터라면 프레임은 ±1(33 ms), 줄은 ±1(12.5 ms) 어긋난다.
- **[미확인]** `anchor_length`의 단위(mm로 추정). 오른손 폴더의 영상이 거울상인지. 참가자 9–12 밖과 Study 2–4의 설정 키와 값.

## 2. 데이터 구조

용어부터 정한다.

- **chirp**: 워치 스피커가 반복해서 내는 18–21 kHz FMCW 신호(600샘플, 48 kHz).
- **echo profile**: 마이크로 받은 소리를 송신 chirp와 상관시킨 결과. 값이 클수록 그 거리에 반사체가 있다.
- **row(줄)**: 시간 방향 한 칸 = chirp 1개 = 600샘플 = 12.5 ms. 초당 80줄.
- **bin(칸)**: 거리 방향 한 칸 = 3.57 mm. 가까운 60 bins = 21.4 cm.
- **direct path**: 스피커에서 마이크로 곧장 오는 소리. bin 300에 고정되고, bin 301부터 멀어진다.

zip은 `Study-1_sub9,10,11,12/<참가자>_<기종>_<손>_video/` 8개 폴더를 담는다(`sub9_samsung_left_video`처럼. 9·10·11은 samsung, 12는 xiaomi). **폴더 하나 = 한 사람의 한 손 = 세션 10개**다. 압축을 풀면 `__MACOSX/`·`.DS_Store`·`._*` 파일이 같이 나온다. 맥이 만든 찌꺼기이니 지우거나 무시한다(`*audio001*` 같은 glob에 걸려 npy로 읽다가 실패한다). 받은 파일은 새 폴더에 풀고 `np.load(..., allow_pickle=False)`로 연다.

폴더 안 파일(세션마다 하나씩, `NNN`은 001–010):

| 파일 | 내용 |
|---|---|
| `audioNNN_fmcw_16bit_profiles.npy` | echo profile. float64 `(600, R)`, R ≈ 13,600(약 170 s). 축 0 = bin, 축 1 = row. **[확인]** 부호 있는 상관값 그대로다(포락선이 아니다). direct path(약 2×10¹⁰)가 bin 300. 시간 방향 3줄 중앙값이 이미 걸려 있고 direct path는 bin 300에 맞춰져 있다(논문의 median filter와 peak correction에 해당하는 것으로 보이나 **[추정]**, 논문은 커널·창 크기를 적지 않았다). 처음 12줄은 direct path가 아직 없다(0 또는 작은 값). |
| `audioNNN_fmcw_16bit_diff_profiles.npy` | diff profile. float64 `(600, R−1)`. **[확인]** `diff[:, j] = \|p[:, j+1]\| − \|p[:, j]\|`(`p`는 위 profile). 그래서 열 `j`는 줄 `j`와 `j+1` 사이, 즉 **줄 좌표 `j + 0.5`**에 놓는다. 어긋나는 열이 파일마다 1–7개 있다(무시해도 된다). 약 34%가 정확히 0인 것은 3줄 중앙값 때문이다. |
| `audioNNN_*.png` | 위 npy의 그림. 쓰지 않는다. |
| `record_YYYYMMDD_HHMMSS_ffffff.mp4` | 손 영상. 1280×720, 약 50 MB. 컨테이너는 **고정 30 fps**다(모든 PTS가 1/30 s 간격. `ffprobe`로 확인). 그래서 PTS로는 촬영 시각을 알 수 없다. |
| `record_…_frame_time.txt` | 프레임마다 UNIX 시각(초) 한 줄. 실제 촬영은 약 29.97 fps다(우리가 본 두 파일 모두). 영상 하나에서 간격은 18–48 ms로 흔들렸고 첫 간격만 99.7 ms였으며, 중복도 중간 빈틈도 없었다. **줄이 디코딩한 프레임보다 많을 수 있고, 남는 줄은 끝에 있다.** |
| `record_…_records.txt` | 지시 간격. 줄마다 `class,시작 UNIX 시각,끝 UNIX 시각,라벨`. 지시끼리 틈 없이 이어지고 하나가 2.0–2.05 s(시작 간격 평균 2.02 s)다. 우리가 본 두 세션은 74줄. class는 라벨의 고정 번호다: 0–4 = 라벨 `1`–`5`(손가락 하나 굽히기), 5–14 = `Zero`–`Nine`(ASL 숫자), 15–17 = `wrist_up`/`wrist_down`/`wrist_left`. |
| `config.json` | 아래 표. |
| `imu_config.json`, `00NN_MMDD_HHMMSS.csv` | IMU(가속도·자이로·지자기) 설정과 폴더당 큰 csv 하나(Samsung 약 22 MB). **관절이 아니다.** 열 이름은 확인하지 않았다. |

`config.json`의 키(`i`는 세션 번호):

| 키 | 값 |
|---|---|
| `audio.files[i]` | 세션 `i`의 echo profile 파일 이름 |
| `audio.syncing_poses[i]` | 동기 위치. echo profile **줄 번호**(§4) |
| `ground_truth.videos[i]`, `ground_truth.files[i]` | 세션 `i`의 mp4, records.txt 이름 |
| `ground_truth.syncing_poses[i]` | 동기 위치. 영상 **프레임 번호**(§4) |
| `ground_truth.anchor_length` | 사람·손마다 하나(참가자 9–12는 73–84, 같은 사람의 두 손이 0–3 차이). 세션별 값은 없다. 손목–새끼 MCP의 실측 길이(mm)로 읽힌다 **[추정]**(§8). |
| `sessions[i]` | 구간 6개 `{start(UNIX 시각), duration(s)}`. 길이는 대략 20, 20, 40, 40, 16, 8 s. **[확인]** `sessions[i][0].start`는 `records`의 첫 지시 시작과 같다(차이 < 10⁻⁶ s, 두 세션). |
| `audio.config` | 48 kHz, `frame_length` 600, 18–21 kHz 대역 통과, 16비트, 송신 파일 이름 `fmcw18000_b3000_l600_s48k.wav`(파일 자체는 데이터에 없다) |
| `tasks` | `["classification", "hand_landmarks"]`(imu_config는 `["hand_landmarks"]`) |

`imu_config.json`에는 `imu.syncing_poses[i]`(IMU 표본 번호), `imu.syncing_offsets[i]`(초), `imu.config.sampling_rate`(Samsung 100, Xiaomi 50), `imu.files`(csv 이름), `config.json`의 `ground_truth`·`sessions` 사본이 있다. **어느 파일에도 관절 좌표는 없다.**

## 3. 세션 짝짓기

README의 규칙은 "session 0 = audio001"이다. 세션 번호 `i`(0부터)에 대해 다음이 모두 같은 세션이다.

```
audio.files[i]  (= audio00(i+1)_fmcw_16bit_profiles.npy, diff는 _diff_profiles)
ground_truth.videos[i]   ground_truth.files[i]   (그리고 그 mp4의 _frame_time.txt)
audio.syncing_poses[i]   ground_truth.syncing_poses[i]   sessions[i]
```

`i`는 mp4 이름으로 `ground_truth.videos.index(이름)`을 써서 구한다. 폴더의 파일 목록을 정렬해서 위치로 짝지으면, 일부 세션만 받았을 때 한 칸씩 밀리기 쉽다.

예: 참가자 9 왼손 폴더의 `record_20250320_202105_929737.mp4`는 `videos[1]`이다. 짝은 `audio002_fmcw_16bit_profiles.npy`이고, 쓸 값은 `audio.syncing_poses[1]`(257)과 `ground_truth.syncing_poses[1]`(34)다. `audio001`의 짝은 `videos[0]`(`record_20250320_201714_769383.mp4`)이다. 우리는 처음에 이 영상에 우리 표본에 있던 `audio001`과 `audio.syncing_poses[0]`(272)을 붙여 상관을 계산했고(이것이 **우리 쌍**이다), 짝이 틀렸다는 것은 config를 다시 읽고서야 알았다(§6.2).

값싼 확인 두 가지(§6.1의 5번): `sessions[i][0].start`가 `records[i]`의 첫 시작과 같고, 그 첫 지시는 `frame_time[0]`보다 약 10.0 s 뒤다(두 영상에서 10.01 s). 이것은 **영상 쪽** 번호가 맞는지만 가려 준다. audio 쪽 짝은 위 규칙을 믿는 수밖에 없고, 확인은 §6.2의 상호상관 몫이다.

## 4. 동기 숫자의 뜻: README와 다른 점

README는 이렇게 적었다(요지).

- `audio.syncing_poses`: "세션별 오디오 동기 sample index".
- `ground_truth.syncing_poses`: "세션별 정답 동기 timestamp".
- 식: `echo_idx = round(((t − gt_sync) × sampling_rate + audio_sync) / frame_length)`. `t`는 `frame_time.txt`의 시각, `sampling_rate` = 48000, `frame_length` = 600. 예제 코드는 `config['ground_truth']['syncing_poses'][i]` 값을 그대로 `gt_sync`에 넣는다.

실제는 이렇다.

| | README | 실제 |
|---|---|---|
| `ground_truth.syncing_poses` | timestamp | **영상 프레임 번호.** 값이 24–68(80세션)이라 30 fps에서 0.8–2.3 s다. UNIX 시각(약 1.74×10⁹)에서 이 값을 빼면 식이 성립하지 않는다. |
| `audio.syncing_poses` | sample index | **echo profile 줄 번호.** 값이 211–437이라 12.5 ms 줄로 2.6–5.5 s다. 48 kHz 표본 번호라면 4–9 ms에 지나지 않는다. |
| 식 | `((t − gt_sync)·sr + audio_sync) / fl` | `(t − frame_time[gt_sync])·sr/fl + audio_sync`. 같은 프레임의 시각을 빼고, `audio_sync`는 나눈 **뒤**(줄 단위에서) 더한다. |

근거:

1. **[확인]** 80세션 모두에서 `imu.syncing_poses == round(audio.syncing_poses × 0.0125 × IMU 표본율)`이다(최대 차이 0). 저자들이 IMU 번호를 만들 때 `audio.syncing_poses`를 12.5 ms 줄 번호로 보고 시간으로 바꿨다는 뜻이다.
2. **[확인]** 위 표의 값 크기, 그리고 README 식을 글자 그대로 쓰면 줄 번호가 1.4×10¹¹이 되는 것(§7).
3. **[확인]** 짝이 맞는 세션 하나에서 위 읽기가 지시 시간표와 0.5 s 안에서 맞는다(§6.1의 4번).
4. **[추정]** 이름이 `syncing_poses`이므로 양쪽 스트림에서 같은 동작이나 자세를 표시했을 것이다. 무엇인지는 논문과 README 어디에도 없다. 본 영상의 첫 1.5 s(프레임 0–44)에는 손이 둘 잡히고, 그 영상의 `gt_sync`(34)가 그 안에 든다. 다른 손으로 동기 동작을 한 흔적일 수 있으나 결정적 증거는 아니다.

## 5. 대응 공식과 코드

프레임 `v`(디코딩한 mp4의 0부터 센 번호)가 놓이는 echo profile의 줄 좌표는 이렇다. `i`는 세션 번호이고 `frame_time`은 그 영상의 줄 목록이다.

```
echo_row(v) = audio_sync[i] + (frame_time[v] − frame_time[gt_sync[i]]) × 80      (실수. 분석에 쓴다)
echo_row(v) = audio_sync[i] + round((frame_time[v] − frame_time[gt_sync[i]]) × 80)  (정수 줄 번호. 창을 자를 때 쓴다)
역: frame_time = frame_time[gt_sync[i]] + (줄 − audio_sync[i]) / 80   → 가장 가까운 프레임
```

- `v`는 mp4를 **순서대로 한 장도 건너뛰거나 늘리지 않고** 디코딩한 번호다. `frame_time[v]`는 앞에서부터 디코딩한 프레임 수만큼 쓴다. fps를 바꾸는 처리(예: ffmpeg `-r 25`, `fps` 필터)는 프레임을 복제·삭제해서 대응을 깬다.
- 80은 `1 / 0.0125`다. 두 기기 시계의 미세한 어긋남은 무시했다(170 s에서 50 ppm이면 9 ms).
- 상관 분석에는 실수 줄 좌표를 쓴다. 정수로 반올림해도 우리 쌍에서 ±1 s 봉우리가 0.337에서 0.332로 바뀌는 정도였다.
- 차분 열 `j`의 줄 좌표는 `j + 0.5`다(§2). "줄 `t`로 들어오는 변화"는 열 `t − 1`이다.
- 프레임의 줄 좌표가 프로파일 안(`0 ≤ echo_row ≤ R`)에 들어오는지 확인한다(§6.1의 2번).

```python
import json
import os

import numpy as np

ROW_S = 0.0125        # echo profile 한 줄 = 600 샘플 ÷ 48 kHz = 12.5 ms
ROWS_PER_S = 80.0


def read_floats(path):
    with open(path, encoding="utf-8") as f:
        return np.array([float(x) for x in f if x.strip()])


def read_records(path):
    with open(path, encoding="utf-8") as f:
        rows = [ln.strip().split(",") for ln in f if ln.strip()]
    return [(float(r[1]), float(r[2]), r[3]) for r in rows]        # (시작 UNIX 시각, 끝 UNIX 시각, 라벨)


def load_session(folder, video_name):
    """folder: 참가자·손 폴더, video_name: 'record_….mp4'. 세션 번호 i는 이름으로 찾는다(파일 정렬 순서를 믿지 않는다)."""
    with open(os.path.join(folder, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    i = cfg["ground_truth"]["videos"].index(video_name)              # 0부터 센 세션 번호
    prof = cfg["audio"]["files"][i]                                  # audio00(i+1)_fmcw_16bit_profiles.npy
    stem = os.path.join(folder, video_name[:-len(".mp4")])
    return dict(
        i=i,
        profile=os.path.join(folder, prof),                                              # (600, R)
        diff=os.path.join(folder, prof.replace("_profiles", "_diff_profiles")),          # (600, R-1)
        frame_time=read_floats(stem + "_frame_time.txt"),
        records=read_records(stem + "_records.txt"),
        gt_sync=cfg["ground_truth"]["syncing_poses"][i],             # 영상 프레임 번호
        audio_sync=cfg["audio"]["syncing_poses"][i],                 # echo profile 줄 번호
        session_start=cfg["sessions"][i][0]["start"],
        anchor_length=cfg["ground_truth"]["anchor_length"],          # 사람·손마다 하나
    )


def echo_rows(frame_time, n_frames, gt_sync, audio_sync):
    """디코딩한 프레임 v = 0..n_frames-1 → echo profile 줄 좌표(실수). frame_time은 앞에서 n_frames줄만 쓴다."""
    t = np.asarray(frame_time[:n_frames], dtype=np.float64)
    return audio_sync + (t - frame_time[gt_sync]) * ROWS_PER_S


def echo_row_index(rows):
    return np.rint(rows).astype(np.int64)          # 창을 자를 때 쓰는 정수 줄 번호


def nearest_frame(rows, row):
    return int(np.argmin(np.abs(rows - row)))      # 역방향: echo 줄 좌표 → 가장 가까운 프레임 번호
```

**학습 창**(우리 선택): 프레임 `v`의 정수 줄 `r = echo_row_index(rows)[v]`에서 끝나는 96줄(1.2 s)과 bins 301–360(60 bins)을 자른다. profile 창은 줄 `r−95 … r`, diff 창은 열 `r−96 … r−1`이다. 논문은 입력을 2채널(원본·차분) × 60 bins × 96줄로, 프레임마다(stride 1/30 s) 창 하나를 만든다고만 적었다. 창이 프레임에서 끝나는지 가운데인지, 60 bins가 정확히 어디인지(300–359인지 301–360인지), 창별 정규화 방법은 밝히지 않았다 **[미확인]**.

## 6. 검증

### 6.1 카메라 없이 하는 값싼 확인

`n`은 mp4에서 실제로 디코딩한 프레임 수다(컨테이너 헤더 값이 아니라 순서대로 읽어 센 값. 우리 영상은 둘이 같았다). 아래는 모두 `load_session`의 결과로 할 수 있다.

1. **남는 줄.** `len(frame_time) − n`을 영상마다 찍는다. 0이면 문제없다. 양수면 앞 `n`줄을 쓰되 3번으로 확인한다. 음수면(프레임이 줄보다 많으면) 대응할 수 없으니 그 영상은 쓰지 않는다.
2. **범위.** `rows.min() ≥ 0`, `rows.max() ≤ R`(`R = diff.shape[1]`)이어야 한다. README 식은 프레임이 하나도 프로파일 안에 들어오지 않는다. 끝쪽 줄을 잘못 쓰는 실수(§7)는 우리 영상에서 46/4,869프레임이 끝 밖으로 나가 걸리지만, 프로파일 끝에 남는 줄(우리 영상은 6.9 s)만큼의 여유가 있는 세션에서는 걸리지 않는다. 범위는 필요조건일 뿐이다.
3. **`frame_time` 읽기.** 남는 줄이 있을 때, 첫 지시 시작을 기준으로 카메라 속도의 중앙값을 앞(−6…−0.5 s)과 뒤(+0.7…+3 s)에서 비교한다(`still_vs_moving`. 두 손이 잡힌 프레임은 뺀다). 앞 `n`줄, 마지막 `n`줄, 가운데 `n`줄 읽기를 모두 넣어 본다. 우리 영상에서 **[확인]**(mm/s):

   | 읽기 | 첫 지시 앞 | 첫 지시 뒤 |
   |---|---|---|
   | 앞 `n`줄 | 9.4 | **28.0** (3배) |
   | 마지막 `n`줄 | 7.9 | 10.4 |
   | 가운데 `n`줄 | 9.3 | 8.5 |

   앞 `n`줄 읽기에서만 손이 첫 지시 전에 가만히 있다가 지시를 따라 움직인다. 이 읽기에서 카메라 움직임이 지시 시작을 0.73 s 뒤따르는 위상(2.02 s로 나눈 나머지)도 세션 내내 일정하다. 카메라 속도와 지시 변화 열차의 상관 값(0.51)은 마지막 `n`줄 읽기도 0.51이라 값만으로는 못 가른다. 남는 줄을 영상 전체에 고르게 늘여 맞추면(stretch) 상관이 0.05로 무너지므로 줄이 중간에서 빠진 것은 아니다. 영상은 첫 지시보다 10.0 s 앞에서 시작해 마지막 지시가 끝나고 3.1 s 뒤에 끝난다.
4. **지시 블록 경계(거친 눈금).** `records` 끝의 손목 블록(라벨이 `wrist_`로 시작하는 마지막 지시들. 우리가 본 두 세션은 12개, 24 s)의 시작과 끝을 §5 식으로 echo 시간으로 옮기고(`block_check`), 그 둘레 diff 에너지(bins 301–360의 \|값\| 합을 0.5 s로 평균해 전체 중앙값으로 나눔)를 본다. 짝이 맞는 세션(참가자 9 오른손 세션 0)에서 **[확인]**: 블록은 137.1–161.3 s에 놓이고, 에너지는 134.5–137.0 s에 0.7 안팎으로 조용하다가 블록이 시작한 직후의 0.5 s 칸(137.5 s)에서 3.9로 뛴다. `audio_sync`를 sample로 읽으면 같은 블록이 133.5–157.7 s에 놓여, 시작 직후 3 s가 조용하고 끝난 뒤에도 움직임이 이어져 맞지 않는다. 카메라가 필요 없고 2.02 s 빗살 모호함이 없지만 눈금이 0.5 s라 33 ms 단위의 어긋남은 못 본다. 짝이 틀린 audio를 걸러 내는지는 시험하지 않았다. 시간표에는 세션 중간의 구간 경계도 있으니 같은 방식으로 더 볼 수 있다 **[미확인]**.
5. **짝.** `session_start == records[0][0]`이고 `records[0][0] − frame_time[0] ≈ 10.0` (§3).

```python
def still_vs_moving(t, world, nhands, first_prompt):
    """t: 후보 읽기의 프레임 UNIX 시각(예: frame_time[:n], frame_time[-n:]). 첫 지시 시작(first_prompt) 앞 -6…-0.5 s와
    뒤 +0.7…+3 s의 카메라 속도 중앙값(mm/s). 두 손이 잡힌 프레임은 뺀다."""
    t = np.asarray(t, dtype=np.float64)
    sp = np.linalg.norm(np.diff(world, axis=0), axis=2).mean(axis=1) / np.diff(t) * 1e3
    mid = (t[1:] + t[:-1]) / 2 - first_prompt
    good = (nhands[1:] <= 1) & (nhands[:-1] <= 1) & np.isfinite(sp)
    return tuple(float(np.median(sp[good & (mid >= a) & (mid < b)])) for a, b in ((-6.0, -0.5), (0.7, 3.0)))


def block_check(energy, records, frame_time, gt_sync, audio_sync, last=None):
    """records 끝의 손목 블록(라벨이 wrist로 시작하는 연속한 마지막 지시들. last로 개수를 줄 수도 있다)의 시작·끝을
    echo 시간(s)으로 옮기고, 0.5 s 평균 diff 에너지 ÷ 전체 중앙값을 돌려준다."""
    if last is None:
        last = 0
        while last < len(records) and records[-1 - last][2].startswith("wrist"):
            last += 1
    to_s = lambda t: (audio_sync + (t - frame_time[gt_sync]) * ROWS_PER_S) * ROW_S
    nb = len(energy) // 40                                  # 40줄 = 0.5 s
    e = energy[:nb * 40].reshape(nb, 40).mean(axis=1)
    return to_s(records[-last][0]), to_s(records[-1][1]), e / np.median(e)
```

### 6.2 카메라–소리 상호상관과 빗살

우리 방법은 `tools/verify_sync.py`의 `estimate_lag`와 같은 처리다. 아래 코드는 우리 쌍에서 같은 봉우리와 높이를 냈다(지연은 포물선 보간을 뺀 12.5 ms 격자 값이다).

- **카메라**: 이웃한 두 프레임 사이에 MediaPipe world landmarks 21점이 움직인 거리의 평균 ÷ 시간(m/s). 시각은 두 프레임의 가운데이고, 간격이 0.1 s를 넘으면 뺀다. world 좌표는 손 중심 기준이라 손 전체의 이동보다 손가락·손목의 움직임을 본다.
- **소리**: diff profile의 bins 301–360 \|값\| 합. 열 `j`는 줄 좌표 `j + 0.5`.
- 카메라를 소리의 줄 좌표에 선형 보간으로 놓고, 둘 다 로그(`v + 중앙값의 1%`)를 취한 뒤 1 s(80줄) 이동 평균을 빼고 표준화한다. 로그는 큰 튐이, 이동 평균 빼기는 구간마다 다른 느린 세기가 상관을 끌지 않게 한다.
- `corr(L) = mean z_sound[t] · z_cam[t + L]`. **`L > 0`이면 카메라가 소리보다 늦다.** 봉우리를 찾았으면 카메라 줄 좌표에서 `L / 0.0125`를 빼서 보정한다.

```python
from scipy.ndimage import uniform_filter1d

NEAR = slice(301, 361)      # direct path(bin 300) 바로 뒤 60 bins = 0–21.4 cm
DETREND_ROWS = 80           # 1 s 이동 평균


def sonar_energy(diff):
    """diff (600, R-1) → (R-1,). 열 j는 줄 좌표 j + 0.5."""
    return np.abs(diff[NEAR]).sum(axis=0)


def camera_speed(rows, world):
    """프레임 사이 21점 평균 이동 거리 ÷ 시간(m/s). 시각(줄 좌표)은 두 프레임의 가운데. 손이 없는(NaN) 프레임은 빠진다."""
    dt = np.diff(rows) * ROW_S
    step = np.linalg.norm(np.diff(world, axis=0), axis=2).mean(axis=1)
    ok = np.isfinite(step) & (dt > 0) & (dt <= 0.1)
    return ((rows[1:] + rows[:-1]) / 2)[ok], step[ok] / dt[ok]


def _z(v, ok, width):
    v = np.log(v + 0.01 * np.median(v[ok]))
    cnt = uniform_filter1d(ok.astype(float), width, mode="nearest")
    v = v - uniform_filter1d(np.where(ok, v, 0.0), width, mode="nearest") / np.maximum(cnt, 1e-9)
    out = np.zeros(len(v))
    s = v[ok].std()
    out[ok] = (v[ok] - v[ok].mean()) / (s if s > 0 else 1.0)
    return out


def lag_curve(energy, cam_rows, speed, max_lag_s=6.0, part=None):
    """corr(L) = mean z_sound[t] · z_cam[t+L]. L > 0이면 카메라가 소리보다 늦다. part=(lo, hi)면 소리 열 lo..hi-1만 쓴다.
    반환 (lags_s, corr)."""
    lo, hi = part if part else (0, len(energy))
    energy = energy[lo:hi]
    grid = np.arange(lo, hi) + 0.5                       # 소리 열의 줄 좌표
    j = np.searchsorted(cam_rows, grid)
    jj = np.clip(j, 1, len(cam_rows) - 1)
    cam = np.interp(grid, cam_rows, speed)
    ok = (j > 0) & (j < len(cam_rows)) & ((cam_rows[jj] - cam_rows[jj - 1]) * ROW_S <= 0.1) & (energy >= 0)
    ze = _z(np.where(ok, energy, 1.0), ok, DETREND_ROWS)
    zc = _z(np.where(ok, cam, 1.0), ok, DETREND_ROWS)
    k = int(round(max_lag_s / ROW_S))
    lags = np.arange(-k, k + 1)
    corr = np.full(len(lags), np.nan)
    n = len(ze)
    for q, lag in enumerate(lags):
        a, b = (slice(0, n - lag), slice(lag, n)) if lag >= 0 else (slice(-lag, n), slice(0, n + lag))
        m = ok[a] & ok[b]
        if m.sum() > 10:
            corr[q] = np.mean(ze[a][m] * zc[b][m])
    return lags * ROW_S, corr


def report(energy, cam_rows, speed, tol_s=0.1):
    """±6 s 곡선에서 전체·앞 절반·뒤 절반마다: 최고점(argmax), 가운데 빗살(|lag| <= tol_s)의 최고점(center),
    이웃 빗살(1–3 s 밖 양쪽)의 최고점(side), 가운데가 이웃보다 높은 정도(margin = center - max(side))."""
    cols = np.arange(len(energy)) + 0.5
    cover = np.flatnonzero((cols >= cam_rows[0]) & (cols <= cam_rows[-1]))      # 카메라가 덮는 소리 열
    h = len(cover) // 2
    out = {}
    for name, part in (("full", None), ("half1", (cover[0], cover[h])), ("half2", (cover[h], cover[-1] + 1))):
        lags, corr = lag_curve(energy, cam_rows, speed, 6.0, part)
        top = int(np.nanargmax(corr))
        c = np.flatnonzero(np.abs(lags) <= tol_s)
        c = c[np.nanargmax(corr[c])]
        side = [float(np.nanmax(corr[(lags > lo) & (lags <= hi)])) for lo, hi in ((-3.0, -1.0), (1.0, 3.0))]
        out[name] = dict(argmax_ms=round(float(lags[top]) * 1e3, 1), argmax=round(float(corr[top]), 3),
                         center_ms=round(float(lags[c]) * 1e3, 1), center=round(float(corr[c]), 3),
                         side=[round(s, 3) for s in side], margin=round(float(corr[c]) - max(side), 3))
    return out
```

**빗살.** 지시가 2.02 s마다 바뀌고 손도 그때마다 움직이므로, 카메라와 소리가 모두 2.02 s 주기의 덩어리를 가진다. 그래서 상관은 지연 축에서 2.02 s 간격의 빗살(comb tooth)이 되고, 빗살마다 봉우리가 거의 같다. 우리 쌍(짝이 틀린 `videos[1]` + `audio001`, §3)에 위 코드를 돌리고 대응 식만 바꾼 **[확인]** 결과다. 식 A2와 C는 §7에서 설명한다. 표의 지연은 `estimate_lag`의 포물선 보간 값이다.

| 대응 | ±5 s 창 최고 | ±1 s 창 최고 |
|---|---|---|
| 권장식(§5) | −1,840 ms (0.339) | **+164 ms (0.337)** |
| A2: `audio_sync`를 sample로 읽음 | −3,229 ms (0.330) | +799 ms (0.322) |
| C: 프레임 번호 ÷ 30 fps | +4,147 ms (0.343) | +115 ms (0.339) |

- 권장식 곡선의 빗살은 −3.86, −1.84, +0.16, +2.19, +4.19 s에 있고 높이가 0.30–0.34다. ±5 s 최고점은 창을 바꾸면 다른 빗살로 건너뛴다: 전체 −1.84 s, 앞 절반 +4.17 s, 뒤 절반 +0.17 s.
- **세션 짝이 틀렸는데도 ±1 s 봉우리가 0.34다.** 이 데이터에서는 봉우리 0.3도, ±1 s 창의 최고 지연이 0 근처인 것도 대응이 맞다는 증거가 못 된다. ±1 s 창은 어떤 대응에서든 빗살 하나를 잡는다. 몇 초 틀린 대응의 어긋남이 ±1 s 안의 엉뚱한 지연으로 접히는 것이 §7의 증상들이다.
- 같은 쌍에서 ±6 s 창의 `report`는 가운데 빗살(0.304)을 이웃(0.339, 0.330)보다 낮게(`margin` −0.035), 앞·뒤 절반의 최고점을 서로 다른 빗살(+4,163 ms 대 −5,850 ms)로 돌려준다. 맞지 않는 대응이 걸러진다는 뜻이지, 맞는 대응이 통과한다는 뜻은 아니다.
- 가운데에 떨어진다는 것만으로는 부족하다. 같은 영상·프로파일에 `audio_sync`만 `audio002`의 값 257로 바꿔 넣어도(프로파일은 여전히 `audio001`이라 짝은 틀리다) 뒤 절반의 최고점이 −25 ms(0.405)에 떨어지지만 이웃 빗살(0.404)과의 차이가 0.001이다. 전체에서는 −2,025 ms가 최고다.

**판정(제안. 짝이 맞는 세션에서는 아직 확인하지 못했다 [미확인]).**

1. **±6 s 창**으로 본다. ±1 s 창의 최고값만 보지 않는다.
2. 가운데 빗살(`|lag| ≲ 100 ms`)이 전체 최고점이고 ±2 s 이웃(`side`)보다 **뚜렷이** 높아야 한다. 틀린 쌍에서 빗살 높이의 퍼짐이 0.04였으니, `margin`이 그보다 한참 커야 한다. 문턱은 정하지 못했다.
3. **앞 절반과 뒤 절반**이 같은 빗살(`argmax_ms`가 서로 한 프레임 안팎)을 최고점으로 골라야 한다. ±1 s 창에서는 틀린 쌍도 앞·뒤가 +158, +168 ms로 맞아 보이므로 ±6 s 창으로 본다.
4. 통과한 세션들에서 `center_ms`가 일정하게 0이 아니면(예: +100 ms) 그 값이 체계적 지연이다. 카메라 줄 좌표에서 빼서 보정한다.
5. 통과하지 못한 세션을 가장 높은 빗살로 옮겨 맞추지 않는다. 2.02 s 틀린 대응이 가장 높은 빗살로 보일 수 있다. 이웃과 비슷하면 그 세션은 "정렬 미확인"으로 두고 따로 센다.
6. 학습 전에 여러 세션에서 `report`를 돌려 `margin`의 분포를 보고 2번의 문턱을 정한다. 맞는 대응이라면 세션들의 `margin`이 대체로 양수여야 한다. 그렇지 않으면 이 방법으로는 정렬을 확인할 수 없는 것이고, 그때는 §6.1의 4번처럼 카메라 없는 확인에 기댄다.

참고: direct path가 bin 300에서 벗어난 줄(우리 표본 파일에서 234/13,599줄)의 diff 에너지는 중앙값의 약 15배로 튀지만, 그 열 254개를 빼도 곡선이 달라지지 않아서(같은 봉우리와 높이) 위 코드는 빼지 않는다.

사용 순서 예(실행하지 않는 예시):

```python
# 사용 예 (실행하지 않는 예시)
sess = load_session(folder, "record_20250320_202105_929737.mp4")          # i = 1 → audio002
world, score, label, nhands = extract_world(os.path.join(folder, "record_20250320_202105_929737.mp4"), "hand_landmarker.task")
n = len(world)                                                           # 디코딩한 프레임 수
print("남는 frame_time 줄:", len(sess["frame_time"]) - n)
rows = echo_rows(sess["frame_time"], n, sess["gt_sync"], sess["audio_sync"])
diff = np.asarray(np.load(sess["diff"], mmap_mode="r", allow_pickle=False), dtype=np.float64)
assert rows.min() >= 0 and rows.max() <= diff.shape[1]
energy = sonar_energy(diff)
w = np.where(keep_mask(world, nhands)[:, None, None], world, np.nan)     # §8의 keep_mask
cam_rows, speed = camera_speed(rows, w)
print(report(energy, cam_rows, speed))
start, end, e = block_check(energy, sess["records"], sess["frame_time"], sess["gt_sync"], sess["audio_sync"])
```

## 7. 함정과 증상

아래 숫자는 우리 쌍(짝이 틀린 `videos[1]` + `audio001`)에 대응 식만 바꿔 돌린 **[확인]** 결과다(§6.2의 표). 모든 식이 같은 `gt_sync`, `audio_sync`, `frame_time`을 쓴다.

| 실수 | 증상 | 근거 |
|---|---|---|
| **README 식을 글자 그대로** (`gt_sync`를 UNIX 시각에서 빼고 `audio_sync`를 sample로 더함) | 프레임 줄 번호가 약 1.4×10¹¹. 프로파일 안에 드는 프레임이 0개. | 첫 프레임 줄 1.39×10¹¹ |
| **`audio_sync`를 sample 번호로 읽음** (`((t − frame_time[gt_sync])·48000 + audio_sync) / 600`) | 카메라가 `audio_sync × 12.5 ms`만큼 일찍 놓인다(80세션에서 2.6–5.5 s). ±5 s 창에서는 참 어긋남이 보이지만 **±1 s 창에서는 이웃 빗살로 접혀** 세션마다 −1…+1 s에 흩어진 지연(예측 −962…+1,000 ms, 중앙 +550 ms)과 늘 0.33–0.34인 봉우리가 나온다. | 우리 쌍: 어긋남 −3.23 s(±5 s 창), 접힌 지연 예측 +800 ms, 실측 +799 ms(0.322). 80세션 값은 이 쌍의 빗살 모양을 다른 `audio_sync`에 옮긴 거친 예측이다. 다른 파이프라인에서 본 −600…+922 ms 흩어짐과 모양이 같지만, 그쪽이 어떤 식을 썼는지는 확인하지 못했다. |
| **`frame_time`의 끝쪽 줄을 씀** (앞 207줄을 버리고 마지막 `n`줄을 씀) | 영상이 6.9 s 늦게 놓인다. 첫 지시 뒤 7.6 s 동안 손이 가만히 있는 것으로 보이고, ±1 s 창에서는 접힌 엉뚱한 지연이 나온다. 우리 영상에서는 끝 46프레임이 프로파일 밖으로 나가 범위 확인에 걸리지만, 프로파일 끝에 남는 줄만큼의 여유가 있으면 걸리지 않는다. | §6.1의 3번 표와 2번 |
| **세션 짝이 틀림** (영상 `i` + `audio` `j ≠ i`) | **0 근처로 보인다**: ±1 s 창 +164 ms, 봉우리 0.34. 모든 세션이 같은 시간표(지시 2.02 s)를 따르기 때문에 빗살이 비슷한 것으로 보인다 **[추정]**. 눈에 띄는 이상이 없다. | §6.2. 예방: `ground_truth.videos.index(이름)` (§3) |
| **프레임 번호 ÷ 30 fps** | 실제 촬영은 29.97 fps에 간격이 흔들려서, 동기 프레임을 기준으로 30 fps 시계가 영상 처음에서 +0.08 s, 끝에서 −0.08 s 어긋난다(처음과 끝이 0.16 s, ≈5프레임 다르다. 첫 간격 99.7 ms가 66 ms를 더한다). 우리 쌍에서는 짝이 틀려 이 어긋남이 상관에 드러나지 않았다(§6.2 표의 C). | `frame_time` 산술 |
| **±1 s 창의 최고값만 봄** | 빗살 하나가 늘 잡히므로 봉우리 0.3이 늘 나온다. | §6.2 |

## 8. 관절 정답 만들기

공개 데이터에는 관절이 없다. 영상에서 만든다. 논문(§3.6.1)의 정답 파이프라인은 코드가 공개되지 않았고 README에도 없다. 아래는 논문 서술을 따르면서 우리 쪽 실험에서 막은 곳을 더한 것이다. 논문과 같다고 확인된 것은 아니다.

**모델.** MediaPipe Hand Landmarker의 `hand_landmarker.task`(float16/1, 7,819,105 B, SHA-256 `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`). 주소는 `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`다. VIDEO 모드, 손 2개까지, 감지·존재·추적 신뢰도 0.5, 프레임마다 world landmarks(m, 원점은 손의 대략적 기하 중심). 논문은 판, 신뢰도, world와 image 중 무엇을 썼는지 밝히지 않았다. 우리는 Python MediaPipe 1.2.0으로 돌렸다. 다른 판은 프레임 단위로 조금 다를 수 있다.

```python
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision


def extract_world(mp4, model_path, max_frames=None):
    """mp4를 순서대로 한 장씩 읽어(건너뛰거나 늘리지 않는다) Hand Landmarker(VIDEO 모드)에 넣는다.
    반환: world (N, 21, 3) m — 손이 없으면 NaN, score (N,) handedness 점수, label (N,) 0 없음 / 1 Left / 2 Right, nhands (N,).
    프레임마다 handedness 점수가 가장 높은 손 하나만 남긴다."""
    opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(mp4)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    world, score, label, nhands = [], [], [], []
    with vision.HandLandmarker.create_from_options(opts) as lm:
        while max_frames is None or len(world) < max_frames:
            ok, bgr = cap.read()
            if not ok:
                break
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            r = lm.detect_for_video(img, int(round(len(world) * 1000.0 / fps)))     # 단조 증가하는 ms
            k = len(r.hand_world_landmarks)
            nhands.append(k)
            if k == 0:
                world.append(np.full((21, 3), np.nan)); score.append(np.nan); label.append(0)
            else:
                b = max(range(k), key=lambda q: r.handedness[q][0].score)
                world.append([[p.x, p.y, p.z] for p in r.hand_world_landmarks[b]])
                score.append(r.handedness[b][0].score)
                label.append(1 if r.handedness[b][0].category_name == "Left" else 2)
    cap.release()
    return np.array(world, dtype=np.float64), np.array(score), np.array(label), np.array(nhands)
```

**공개 영상 하나(참가자 9 왼손의 `videos[1]`)에서 본 것 [확인].**

- 4,869/4,869프레임에서 손이 잡혔다(100%).
- **거울상이다.** `wx → −wx`로 되돌린 뒤 손바닥 틀(아래)에서 엄지 관절 2·3·4가 손바닥 쪽에 있는 프레임이 거른 4,794프레임 모두(100%)이고, 되돌리지 않으면 하나도 없다(0%). 거르지 않은 4,869프레임 전체에서는 99.8%와 0.2%다. 왼손 모양이면 이 비율이 1에 가깝다. 왼손 폴더인데 MediaPipe는 4,858프레임을 `Right`, 11프레임을 `Left`로 불렀다. 라벨 이름은 믿지 말고 이 엄지 검사(`palm_unit`의 `thumb_palmar`)로 모양의 손잡이를 직접 본다. **오른손 폴더 영상은 보지 못했다 [미확인].** 폴더마다 두 설정(`mirrored` 참/거짓)의 `thumb_palmar`를 찍어서 정한다.
- **두 손이 잡힌 프레임이 73개**다: 앞 45프레임(0–44, 첫 1.5 s)과 끝 28프레임(4,800번 이후). 점수가 높은 쪽을 고르면 두 손 사이를 오가는 것으로 보인다 **[추정]**. 손목→중지 MCP 벡터가 프레임 사이 50 mm 넘게 튀는 곳이 7번이고 모두 이 프레임들이다. `nhands > 1` 프레임은 목표에서 뺀다. `gt_sync`는 이 앞 구간 안에 있지만 대응 계산에는 `frame_time`만 쓰므로 상관없다.
- **우리 폰 파이프라인의 길이 거르기를 그대로 쓰면 안 된다.** 손목–새끼 MCP 길이(world, 중앙값 89 mm)는 자세에 따라 자연스럽게 흔들린다. 중앙값 ±3σ(σ = 1.4826 × MAD = 1.26 mm)로 거르면 13.1%가 빠지고 손목 동작 프레임이 29.7%(`wrist_up` 40.5%)로 동작에 치우쳐 빠진다. 5σ는 6.4%. 중앙값의 ±10%로 거르면 0.9%(손목 5.2%), ±12%는 0.12%다. 단위 손(아래)은 프레임마다 자기 길이로 나누므로 길이 오차는 목표에 해가 없고, 이 거르기는 추적 실패만 걸러 내면 된다. 아래 `keep_mask`는 ±12%와 한 손만 쓰고 4,794/4,869프레임을 남긴다(1.5% 제외). 거르는 문턱을 바꿀 때는 라벨별 제외율을 찍어서 동작에 치우치지 않는지 본다. 보간하지 않는다.

**전처리 순서.**

1. 거르기(`keep_mask`).
2. 거울 되돌리기: world `wx → −wx`(영상이 거울상일 때. 위 확인). 두 손을 한 모델에 섞으려면 모든 폴더에서 `thumb_palmar ≈ 1`(왼손 모양)이 되게 `mirrored`를 골라 한쪽으로 접는다(우리 도구가 이렇게 한다). 손별로 따로 학습하면 실제 손 모양(왼손 폴더 ≈ 1, 오른손 폴더 ≈ 0)을 그대로 둔다. 논문은 어느 쪽인지 적지 않았다 **[미확인]**.
3. 손목 원점: 21점에서 0번(WRIST)을 뺀다. 목표는 1–20번 20점이다(엄지 1–4, 검지 5–8, 중지 9–12, 약지 13–16, 소지 17–20. MCP는 손가락 뿌리 관절로 5 = 검지, 9 = 중지, 17 = 새끼). 0번은 늘 0이라 뺀다.
4. 손바닥 틀: `x = unit(p5 − p0)`, `z = unit((p5 − p0) × (p17 − p0))`, `y = z × x`로 만든 정규직교 틀로 회전한다(프레임마다).
5. 손 크기: 논문 계보(EchoWrist, Ring-a-Pose, WatchHand)는 프레임마다 `|p17 − p0|`(손목–새끼 MCP)가 사용자별 실측 길이가 되게 크기를 고친다. 그 길이가 `anchor_length`(mm로 추정)일 것이다. 이는 `|p17 − p0|`를 1로 맞춘 단위 손에 `anchor_length`를 곱한 것과 같다. 우리는 단위 손으로 학습하고 mm로 보고할 때 곱한다(사람·손마다 상수라 목표의 크기만 다르다).
6. 시간 평활은 첫 판에 하지 않는다. 손목 회전은 목표에서 뺀다(거울을 되돌린 카메라 좌표의 `p9 − p0`는 따로 저장해 둔다. `tools/joints.py`의 `targets`가 돌려준다).

```python
def keep_mask(world, nhands, rel_tol=0.12):
    """추적 실패만 뺀다: 손이 없음(NaN), 손이 둘 잡힘, 손목–새끼 MCP 길이가 세션 중앙값에서 rel_tol 넘게 벗어남."""
    L = np.linalg.norm(world[:, 17] - world[:, 0], axis=1)
    med = np.nanmedian(L)
    return np.isfinite(L) & (nhands <= 1) & (np.abs(L - med) <= rel_tol * med)


def palm_unit(world, mirrored=True):
    """world (M, 21, 3) m → (목표 (M, 20, 3) 단위 손, 손목–새끼 MCP 길이 (M,) m, 엄지가 손바닥 쪽인 프레임 비율).
    mirrored=True면 wx → −wx로 거울을 되돌린다. 단위 손에 anchor_length를 곱하면 mm 좌표다(단위는 추정)."""
    p = np.asarray(world, dtype=np.float64) * (np.array([-1.0, 1.0, 1.0]) if mirrored else 1.0)
    p = p - p[:, :1]                                                     # 손목(0번) 원점
    a, b = p[:, 5], p[:, 17]                                             # 검지 MCP, 새끼 MCP
    ex = a / np.linalg.norm(a, axis=1, keepdims=True)
    n = np.cross(a, b)
    ez = n / np.linalg.norm(n, axis=1, keepdims=True)
    ey = np.cross(ez, ex)
    q = np.einsum("mac,mjc->mja", np.stack([ex, ey, ez], axis=1), p)     # 손바닥 틀 좌표
    length = np.linalg.norm(q[:, 17], axis=1)
    thumb_palmar = float((q[:, [2, 3, 4], 2].mean(axis=1) < 0).mean())   # 왼손 모양이면 1에 가깝고 오른손 모양이면 0에 가깝다
    return (q / length[:, None, None])[:, 1:], length, thumb_palmar
```

`tools/joints.py`에 같은 처리가 있다(`to_left`·`palm_frame`·`unit_hand`·`targets`·`thumb_palmar`, 위 `palm_unit`과 같은 결과를 낸다). `keep_frames`의 ±3σ는 우리 폰 세션에 맞춘 값이니 공개 영상에는 위 `keep_mask`를 쓴다. 같은 파일에 평가 계기(`frame_errors`, `mean_pose`, `collapse`, `bone_shortfall`)가 있다. 평균 자세를 상수로 예측했을 때의 오차를 맨 윗줄에 두고 모델 오차를 그 비율로 읽는 것이 우리 평가 방식이다.

**손실과 입력(논문 서술).** 손실은 20관절 3D 좌표(60값)의 MSE + 0.1 × velocity 손실이다. velocity 식은 밝히지 않았다. 예측과 정답의 프레임 간 차이를 맞추는 형태가 안전하다고 본다(예측의 움직임 자체를 벌하면 예측이 평균으로 쏠릴 수 있다) **[추정]**. 입력은 §5의 2 × 60 × 96 창이다. 논문은 지표로 MPJPE·MPJAE·MWAE를 쓰고 각도 식은 밝히지 않았다.

**`anchor_length`.** 참가자 9–12에서 73–84이고 사람·손마다 하나다. 손목–새끼 MCP의 실측 길이(mm)로 읽는 것이 가장 그럴듯하다 **[추정]**(논문에 "anchor"라는 말이 없고 단위를 적은 문서도 없다).

**회전 서술의 모호함.** 논문은 손바닥 평면을 기준 자세(baseline posture)의 평면에 맞춘다고 하면서 "손목 회전은 정규화하지 않는다"고도 적었다. 매 프레임 평면을 기준에 완전히 맞추면 손 전체의 방향이 지워지므로 두 서술이 함께 서기 어렵다(기준 자세에서 한 번 구한 회전을 세션 전체에 걸었거나, 회전의 일부 성분만 맞췄을 수 있다). 위 4번(프레임마다 손바닥 틀로 돌려 손목 회전을 목표에서 뺀 것)이 논문과 같은지는 확인하지 못했다 **[미확인]**.

## 9. 라이선스와 인용

- 데이터: Kim 외, *Data from: WatchHand: Enabling Continuous Hand Pose Tracking On Off-the-Shelf Smartwatches*, Cornell eCommons, 2026, doi:[10.7298/qf1v-j805](https://doi.org/10.7298/qf1v-j805). 페이지: https://ecommons.cornell.edu/handle/1813/121443. 라이선스 **CC BY 4.0**(출처 표시, 변경했으면 변경 표시). 전체 약 175 GB(Study 1 93.58 GB, Study 2 22.65 GB, Study 3 28.82 GB, Study 4 29.60 GB).
- 논문: WatchHand, arXiv 2602.21610 (CHI 2026).
- 이 저장소는 저자와 관계없는 개인 연구다. **이 저장소에는 공개 데이터도, 그것에서 만든 파생물(관절 npz, 프로파일 npy, 그림)도 넣지 않는다.** `.gitignore`가 `*.mp4`, `*.npy`, `*.npz`, `*.png`, `*.wav`를 막지만 csv·txt·json은 막지 않으니, 받은 데이터는 저장소 밖 폴더에 둔다. 커밋하지 않는다.

이 문서가 다루지 않는 것: 모델 구조와 학습 설정, 데이터 분할(논문은 교차 세션·세션 내·교차 사용자), 창별 정규화, IMU 사용.
