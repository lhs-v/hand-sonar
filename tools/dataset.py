"""학습 자료: 카메라 프레임 하나에 에코 창 하나와 손 관절 목표 하나(손 소나 계획 4, 학습 설계 §2·§4).

    python -I tools/dataset.py [--data 데이터 폴더] [--out dataset.npz] [--save-x] [세션 이름 …]   (보통은 train.py가 부른다)

- 쓰는 세션(설계 §2): hs_io.chosen이 고른 세션 중 align.json이 "표지음 둘"(method two)이고, sync.json이 usable이며 그 입력 지문
  (align.json·rows.npz·landmarks.csv)이 지금 파일과 같고, pairs.csv가 있는 것. "표지음 하나" 세션은 쓰지 않는다.
- 창: pairs.csv의 짝 줄 r에서 끝나는 96줄(1.2 s, 인과적 정렬), 칸 300–359(60칸, 직접 경로부터 약 21.4 cm). 채널 0은 원본(부호 있는
  상관), 채널 1은 차분이다. 차분 열 j는 원본 줄 j+1의 변화라(pairs.py) 원본 열 r−95…r에 맞는 차분 열은 r−96…r−1이다.
  창마다 채널별로 평균을 빼고 표준편차로 나눈다. 모양 (2, 96, 60).
- 대응표 확인: pairs.csv의 soundNs가 지금 sync.json 지연으로 만든 값(soundNs − frameNs = −lagMs·1e6, ±1 ns)이 아니면 그 세션을 건너뛴다
  (verify_sync.py만 다시 돌고 pairs.py는 다시 돌지 않아 옛 지연의 row가 남은 경우. 입력 지문은 같아 eligible로는 못 가린다).
- 거르기: pairs.csv hand·direct가 1이고, 창이 쓰는 원본 줄 r−96…r(차분 첫 열이 원본 줄 r−96도 쓰므로 97줄)이 모두 rows.npz ok이고,
  정답 거르기(joints.keep_frames: 점수·손목 튐·길이 ±3σ)를 지난 프레임만 쓴다. 영상 조건(pairs.csv video 칸)으로는 거르지 않고 알리기만
  한다: 정답은 영상 없이 landmarks.csv frameNs로 맞추고(수집 설계 §8.1), video 칸은 세션 단위로 모두 차거나 모두 비어 영상 인코더가
  한 번 어긋난 세션을 통째로 버리게 되기 때문이다. 길이 거르기가 동작
  종류별로 뺀 비율을 보고 줄에 적고, 한 종류가 치우쳐 빠지면(전체 비율의 2배 넘게) 그 사실도 적는다. 거울을 되돌린 손에서 엄지가
  손바닥 쪽인 프레임(joints.thumb_palmar)이 90% 아래인 세션은 손 방향 처리가 틀렸거나 반쯤 거울인 것이라 쓰지 않는다.
- 동작 라벨(평가용, 학습 목표 아님): pairs.csv prompt·classId·flag·edge. 라벨로 쓸 줄 = prompt ≥ 0, edge 0, 일시정지 아님,
  flag ≠ wrong(pairs.py와 같은 규칙). 자유 동작(−1)은 라벨이 있는 줄로 친다. 기준 자세(−3, 세션 처음 평행 자세)는 동작이 아니라 라벨에서 뺀다
  (프레임은 학습에 그대로 쓴다).
- 손목 정답(wrist.session_wrist, 2026-10-30): 프레임마다 워치 축으로 적은 손 방향 u와 카메라 깊이 축 d. 워치·폰 센서 기록이 없는 세션
  (이 기능 전·공개 데이터)은 wrist_ok가 모두 거짓이고 u·d는 0이다. 학습은 train.py --wrist일 때만 쓴다.
- 나누기(설계 §4). 셋 다 (학습, 검증, 시험) 프레임 번호를 세션·시각 순서로 낸다.
  - session_split(세션 단위): 착용 묶음 통째로 나눈다(계획 6 검토 고침 M1). 착용(wear_episodes): 시각(meta start wallMs) 순으로 meta.json
    rewear가 참인 세션에서 새 착용이 시작해 다음 그런 세션 전까지 간다. 경계에는 자료에 들지 않은 세션(휴지통·끝나지 않음·처리 안 됨)의
    표시도 쓴다(rewear_marks — 다시 찬 첫 세션을 버렸어도 그 뒤는 새 착용). 시험 = 자료의 착용이 둘 이상이면 시각 순 마지막 착용의 세션 전부
    (같은 착용의 뒤 세션이 학습에 들지 않게). 착용이 하나뿐이면(표시가 없거나 맨 앞 세션에만) 시각 순 마지막 max(1, round(0.15 × 세션 수))세션
    (같은 착용 안이라 낙관적 — 알림). 검증 = 시험이 아닌 세션 중 가장 늦은 하나. 시험이 아닌 세션이 하나뿐이면(세션 둘) 검증 세션을 따로 둘 수
    없어 학습 세션의 끝 15%(앞 2 s 비움)를 검증으로 쓴다. 학습 = 나머지. 시험·검증 세션을 직접 고를 수도 있다(시험과 같은 착용이 학습·검증에
    남으면 알림). 세션이 하나면 나누지 못한다(ValueError). rewear는 폰 준비 화면의 "워치 다시 참"이 적는다(계획 6, 다시 찬 뒤 처음 보관된
    세션에 참). 그 전 세션은 늘 false다.
  - block_split(세션 안, 실험용): 세션 시간표를 10 s 블록으로 자르고 블록 번호 mod 20(+offset)으로 학습 14 · 검증 3 · 시험 3자리를 고정
    무늬로 흩어 놓는다(세 자료가 세션 전체에 걸침). 역할이 바뀌는 모든 경계 뒤 2 s를 비운다(시험 → 학습 경계도). 같은 착용 안의
    낙관적 상한이다: 원본 채널이 세션 안 시각을 담고 있어 학습 블록 사이에 낀 시험 블록은 표류를 보간할 수 있다.
  - time_split(세션 안): 세션마다 프레임 시각 순서로 앞 70% 학습, 다음 15% 검증, 나머지 시험. 검증·시험은 앞쪽 경계 뒤 2 s 안의
    프레임을 버린다(창 1.2 s가 앞쪽 정답 시각과 겹치지 않게). 안내 시간표가 고정 순서라 시험은 늘 자유 동작 꼬리다.
dataset.npz(allow_pickle 없이 읽힘): y (N, 60) float32(단위 손 손바닥 틀, 점 1–20 × xyz),
frame_ns (N,) int64, session (N,) int16, row (N,) int64, prompt (N,) int32, class_id (N,) int16(라벨 없음 −99), flag (N,) 문자열,
edge·labelled (N,) bool, v9 (N, 3) float32(m, 거울을 되돌린 카메라 좌표의 p9 − p0), wrist·wrist_depth (N, 3) float32와 wrist_ok (N,) bool
(손목 정답, 위), wrist_method (S,) 문자열("calibration" | "ulnar" | ""), sessions·hands (S,) 문자열,
length_mm (S,) float64(세션의 L), rewear (S,) bool(meta.json rewear), wear (S,) int64(착용 번호, wear_episodes — 옛 dataset.npz에는 없고 그때는
rewear로 만든다), wall_ms (S,) float64(meta start wallMs), label_ids (K,) int16,
label_names (K,) 문자열, input_files (3,) 문자열(verify_sync.INPUTS)과 inputs (S, 3) 문자열(그 세션 sync.json의 입력 SHA-256 지문).
창 x (N, 2, 96, 60) float32는 기본으로 쓰지 않는다(프레임당 46 KB, 세션당 약 170 MB). 창은 지문이 같은 processed 폴더의
profiles.npy·diff_profiles.npy에서 row로 다시 만든다(windows). save(…, with_x=True)(명령의 --save-x)일 때만 x도 쓴다.
"""
import argparse
import csv
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # python -I는 3.11부터 -P를 겸해 스크립트 폴더를 경로에 넣지 않는다
import hs_io  # noqa: E402  사용자 설치 numpy 경로도 여기서 붙는다
import joints  # noqa: E402
import numpy as np  # noqa: E402
import pairs  # noqa: E402
import verify_sync  # noqa: E402
import wrist  # noqa: E402

WINDOW = 96  # 창 줄 수(1.2 s)
BINS = slice(300, 360)  # 칸 300–359(설계 §8: 첫 판은 칸 300을 넣는다)
SPLIT = (0.70, 0.15, 0.15)  # 학습·검증·시험
GAP_SEC = 2.0  # 나눈 경계마다 비우는 시간
MIN_PALMAR = 0.9  # 거울을 되돌린 손에서 엄지가 손바닥 쪽인 프레임 비율 하한. 20261027-235412는 99.66%(되돌리기 전 0.34%)라 여유가 크고, 0.5면 프레임 40%가 뒤집힌 세션도 지난다
KINDS = (("finger", "손가락 굽히기"), ("asl", "ASL 숫자"), ("wrist", "손목"), ("free", "자유 동작"))  # 동작 종류: classId 0–4 · 5–14 · 15–17 · −1
KIND_KEYS = tuple(k for k, _ in KINDS) + ("none",)  # 마지막 none = 라벨 없음(동작 시작 0.5 s·쉬는 때·동작 밖)
SKEW_RATIO = 2.0  # 길이 거르기가 한 동작 종류에서 전체 비율의 이 배 넘게 빼면 치우쳤다고 알린다(20261027-235412: 손목 10%, 전체 2%)
SKEW_MIN_FRAMES = 5  # 몇 프레임뿐인 우연은 알리지 않는다
LAG_TOL_NS = 1  # pairs.csv soundNs가 frameNs − sync.json lagMs(pairs.py와 같은 float64 셈)에서 이만큼 넘게 다르면 옛 지연으로 만든 대응표다(실데이터 0.5 ns 안)
BLOCK_SEC = 10.0  # 블록 나누기: 세션 시간표를 이 길이의 블록으로 자른다
BLOCK_PERIOD = 20  # 블록 번호 mod 20(+offset)으로 역할을 정한다: 학습 14 · 검증 3 · 시험 3자리 = 70/15/15
BLOCK_TEST = (0, 7, 14)  # 시험 자리: 20블록 안에 고르게 흩어짐(7·7·6칸 간격). 이 세션(22블록)에서는 자리 0이 두 번(블록 0·20) 나와 시험 4블록
BLOCK_VAL = (3, 10, 17)  # 검증 자리: 시험 사이사이(3·4·3·4·3·3칸 간격으로 시험·검증이 번갈아 놓임)
ROLES = ("train", "val", "test")  # block_assign의 역할 번호 0·1·2


def kind_of(class_id, labelled):
    """프레임마다 동작 종류 번호 (N,): 0 손가락 굽히기(classId 0–4), 1 ASL 숫자(5–14), 2 손목(15–17), 3 자유 동작(−1), 4 라벨 없음
    (라벨이 아니거나 그 밖의 음수 번호 — 기준 자세 −3 등)."""
    cid = np.asarray(class_id)
    none = ~np.asarray(labelled, dtype=bool) | ((cid < 0) & (cid != -1))
    return np.where(none, 4, np.where(cid == -1, 3, np.where(cid >= 15, 2, np.where(cid >= 5, 1, 0))))


def filter_lines(class_id, base, good, length):
    """길이 거르기가 동작 종류별로 뺀 비율을 적은 보고 줄. base = 점수·손목 조건을 지난 프레임, good = 길이까지 지난 프레임.
    종류는 프레임을 담은 동작 줄로 센다(시작 0.5 s 경계도 그 동작). 한 종류가 치우쳐 빠지면 그 사실을 한 줄 더 적는다."""
    if not base.any():
        return []
    med, sigma = joints.length_limits(length, base)
    cid = np.asarray(class_id)
    kind = kind_of(cid, (cid != pairs.NO_CLASS) & (cid != pairs.PAUSE_CLASS))
    drop = base & ~good
    total = float(drop.sum() / base.sum())
    cells, skewed = [], []
    for k, (_, name) in enumerate(KINDS + (("none", "라벨 없음·쉬는 때"),)):
        n, d = int((base & (kind == k)).sum()), int((drop & (kind == k)).sum())
        if n:
            cells.append(f"{name} {d / n:.1%}({d}/{n})")
            if d >= SKEW_MIN_FRAMES and d / n >= SKEW_RATIO * total:
                skewed.append(f"{name} {d / n:.1%}(전체의 {d / n / total:.1f}배)")
    lines = [f"  손목–새끼 MCP 길이 거르기(중앙값 {med * 1e3:.1f} mm, σ = {joints.MAD_SCALE:g} MAD = {sigma * 1e3:.2f} mm, ±{joints.MAD_LIMIT:g}σ 밖):"
             f" {int(drop.sum())}프레임({total:.1%}), 동작 종류별 " + " · ".join(cells)]
    if skewed:
        lines.append("  치우쳐 빠짐: " + ", ".join(skewed) + " — 그 동작에서 MediaPipe 손 크기(손목–새끼 MCP)가 실제로 흔들린다."
                     " 단위 손이라 크기 오차는 목표에 해가 없지만 그 동작의 프레임이 줄어든다")
    return lines


def read_pairs(path):
    """pairs.csv를 열마다 배열로. 빈 칸은 row·prompt·video −1, classId pairs.NO_CLASS."""
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    def ints(key, empty):
        return np.array([int(r[key]) if r[key] != "" else empty for r in rows], dtype=np.int64)

    return {"frameNs": ints("frameNs", 0), "soundNs": ints("soundNs", 0), "row": ints("row", -1), "hand": ints("hand", 0),
            "direct": ints("direct", 0), "prompt": ints("prompt", -1), "classId": ints("classId", pairs.NO_CLASS), "edge": ints("edge", 0),
            "video": ints("video", -1), "flag": np.array([r["flag"] for r in rows], dtype=str)}


def lag_mismatch(p, lag_ms):
    """pairs.csv(read_pairs)의 soundNs가 지연 lag_ms로 만든 값과 다른 프레임 수. pairs.py는 soundNs = round(frameNs − lagMs·1e6)를 float64로
    셈하므로 같은 셈과 견준다(폰 가동 시간이 길어 frameNs가 2^52 ns를 넘어도 ±1 ns 안에 든다)."""
    expected = np.round(p["frameNs"].astype(np.float64) - lag_ms * 1e6)
    return int((np.abs(p["soundNs"] - expected) > LAG_TOL_NS).sum())


def eligible(data, name):
    """학습에 쓸 수 있는 세션인가: (됨, 안 되는 이유)."""
    out = os.path.join(data, "processed", name)
    for f in ("align.json", "sync.json", "pairs.csv", "rows.npz", "profiles.npy", "diff_profiles.npy"):
        if not os.path.exists(os.path.join(out, f)):
            return False, f"{f} 없음 — PC 도구(align·profiles·verify_sync·pairs)를 먼저"
    method = hs_io.read_json(os.path.join(out, "align.json")).get("method")
    if method != "two":
        return False, f'맞춤이 "표지음 둘"이 아님(method {method})'
    sync = hs_io.read_json(os.path.join(out, "sync.json"))
    good, why = verify_sync.usable(sync)
    if not good:
        return False, why
    now = verify_sync.inputs(data, name)
    changed = [f for f in verify_sync.INPUTS if (sync.get("inputs") or {}).get(f) != now[f]]
    if changed:
        return False, f"sync.json을 잰 뒤 바뀐 입력: {', '.join(changed)} — verify_sync.py·pairs.py를 다시"
    return True, ""


def window_ok(ok, rows):
    """창마다 원본 줄 r−96…r(97줄)이 모두 ok인가 (N,). 앞 줄이 모자라거나(r < 96) 범위 밖이면 False."""
    ok = np.asarray(ok, dtype=bool)
    rows = np.asarray(rows, dtype=np.int64)
    bad = np.concatenate([[0], np.cumsum(~ok)])  # bad[k] = 줄 0…k−1 중 ok가 아닌 줄 수
    r = np.clip(rows, WINDOW, len(ok) - 1)
    return (rows >= WINDOW) & (rows < len(ok)) & (bad[r + 1] - bad[r - WINDOW] == 0)


def window_columns(rows):
    """(원본 열 (N, 96), 차분 열 (N, 96)). 창의 k번째 줄(0이 가장 이름)은 원본 열 r−95+k이고, 같은 시각의 차분 열은 r−96+k다."""
    raw_cols = np.asarray(rows, dtype=np.int64)[:, None] + np.arange(1 - WINDOW, 1)
    return raw_cols, raw_cols - 1


def standardize(a):
    """마지막 두 축(줄 × 칸)마다 평균을 빼고 표준편차로 나눈다(표준편차가 0이면 1로 나눔)."""
    mean = a.mean(axis=(-2, -1), keepdims=True)
    std = a.std(axis=(-2, -1), keepdims=True)
    return (a - mean) / np.where(std > 0, std, 1.0)


def windows(raw, diff, rows, block=512, bins=BINS, standardized=True):
    """모델 입력 (N, 2, 96, 칸 수) float32. raw (600, R)·diff (600, R−1)는 profiles.npy·diff_profiles.npy(mmap이어도 된다).
    bins를 넓히고 standardized=False면 표준화 전 값이다(train --shift: 학습 때 60칸을 잘라 표준화한다)."""
    raw60 = np.asarray(raw[bins], dtype=np.float64).T  # (R, 칸 수)
    diff60 = np.asarray(diff[bins], dtype=np.float64).T  # (R − 1, 칸 수)
    raw_cols, diff_cols = window_columns(rows)
    out = np.empty((len(raw_cols), 2, WINDOW, raw60.shape[1]), dtype=np.float32)
    for a in range(0, len(raw_cols), block):
        prep = standardize if standardized else (lambda v: v)
        out[a: a + block, 0] = prep(raw60[raw_cols[a: a + block]])
        out[a: a + block, 1] = prep(diff60[diff_cols[a: a + block]])
    return out


def session_data(data, name, bins=BINS, standardized=True):
    """세션 하나: (학습 자료 dict 또는 None, 보고 줄). bins·standardized는 창 x를 만드는 방식(windows)."""
    out = os.path.join(data, "processed", name)
    folder = hs_io.session_folder(data, name)
    p = read_pairs(os.path.join(out, "pairs.csv"))
    sync = hs_io.read_json(os.path.join(out, "sync.json"))
    lag_ms = hs_io.num(sync, "lagMs")
    stale = lag_mismatch(p, lag_ms)
    if stale:
        return None, [f"{name}: 건너뜀(pairs.csv soundNs가 지금 sync.json 지연 {lag_ms:+.3f} ms로 만든 값과 {stale}프레임에서 다름 — 옛 지연으로 만든"
                      " 대응표: pairs.py를 다시)"]
    ok = np.load(os.path.join(out, "rows.npz"), allow_pickle=False)["ok"]
    raw = np.load(os.path.join(out, "profiles.npy"), mmap_mode="r", allow_pickle=False)
    diff = np.load(os.path.join(out, "diff_profiles.npy"), mmap_mode="r", allow_pickle=False)
    if raw.shape != (600, len(ok)) or diff.shape != (600, len(ok) - 1):
        return None, [f"{name}: 건너뜀(profiles.npy {raw.shape}·diff_profiles.npy {diff.shape}와 rows.npz {len(ok)}줄이 맞지 않음 — profiles.py를 다시)"]
    marks = hs_io.read_landmarks(folder)
    if not np.array_equal(marks.frame_ns, p["frameNs"]):
        return None, [f"{name}: 건너뜀(pairs.csv와 landmarks.csv의 프레임이 다름 — pairs.py를 다시)"]
    phone = hs_io.read_session(folder)
    hand = phone.hand
    good, length = joints.keep_frames(marks.world, marks.score, marks.valid)
    base, _ = joints.scored(marks.world, marks.score, marks.valid)
    paired = (p["hand"] == 1) & (p["direct"] == 1)
    windowed = window_ok(ok, p["row"])
    use = paired & windowed & good
    scored = np.nan_to_num(marks.score, nan=-1.0) >= joints.MIN_SCORE
    lines = [f"{name}({hand}): 프레임 {len(use)}개 중 {int(use.sum())}개 씀",
             f"  뺀 것(겹침 있음): 짝 줄 못 씀(손·직접 경로) {int((~paired).sum())}, 창 안 ok 아닌 줄·앞 줄 모자람 {int((~windowed).sum())},"
             f" 정답 거르기 {int((~good).sum())}(손 없음·점수 {joints.MIN_SCORE} 아래 {int((~scored).sum())},"
             f" 손이 둘인 줄 손목 튐 {int((scored & ~marks.valid).sum())},"
             f" 손목–새끼 MCP ±{joints.MAD_LIMIT:g}σ 밖 {int((base & ~good).sum())})"] + filter_lines(p["classId"], base, good, length)
    if (p["video"] < 0).any():  # 거르지 않고 알리기만 한다(머리말)
        lines.append(f"  영상 n = 줄 n 아님(pairs.csv video 칸 빔 {int((p['video'] < 0).sum())}줄 — meta video·프레임 수를 볼 것):"
                     " 정답은 영상 없이 frameNs로 맞추므로 그대로 쓴다")
    if not use.any():
        return None, lines
    y, v9, palmar = joints.targets(marks.world[use], hand)
    if palmar < MIN_PALMAR:
        return None, lines + [f"  건너뜀: 거울을 되돌린 손에서 엄지가 손바닥 쪽인 프레임 {palmar:.1%}(문턱 {MIN_PALMAR:.0%}. 실제 손 모양이 아니거나"
                              " 반쯤 거울 — 거울·손 처리를 볼 것)"]
    labelled = ((p["prompt"] >= 0) & (p["edge"] == 0) & (p["classId"] != pairs.PAUSE_CLASS) & (p["classId"] != pairs.NEUTRAL_CLASS)
                & (p["flag"] != "wrong"))
    try:
        w, wrist_lines = wrist.session_wrist(data, name)
    except Exception as e:  # noqa: BLE001  손목 정답이 넘어져도 손가락 자료는 쓴다
        w, wrist_lines = None, [f"{name}: 손목 정답 계산 오류({type(e).__name__}: {e})"]
    lines += ["  " + line for line in wrist_lines]
    if w is None:
        w = {"u": np.zeros((len(use), 3), np.float32), "depth": np.zeros((len(use), 3), np.float32), "ok": np.zeros(len(use), bool), "method": ""}
    length_mm = float(np.median(length[good])) * 1000.0
    lines.append(f"  L(손목–새끼 MCP 중앙값) {length_mm:.1f} mm, 엄지가 손바닥 쪽인 프레임 {palmar:.1%}, 라벨 붙은 프레임 {int((use & labelled).sum())}개")
    part = {"x": windows(raw, diff, p["row"][use], bins=bins, standardized=standardized), "y": y.reshape(-1, 60).astype(np.float32), "frame_ns": p["frameNs"][use],
            "row": p["row"][use], "prompt": p["prompt"][use].astype(np.int32), "class_id": p["classId"][use].astype(np.int16),
            "flag": p["flag"][use], "edge": p["edge"][use] == 1, "labelled": labelled[use], "v9": v9.astype(np.float32),
            "wrist": w["u"][use], "wrist_depth": w["depth"][use], "wrist_ok": w["ok"][use], "wrist_method": w["method"],
            "name": name, "hand": hand, "length_mm": length_mm, "rewear": bool(phone.meta.get("rewear")),
            "wall_ms": hs_io.num(phone.meta.get("start"), "wallMs", np.nan),
            "inputs": [str((sync.get("inputs") or {}).get(f, "")) for f in verify_sync.INPUTS],
            "label_names": {q["classId"]: q["name"] for q in hs_io.read_prompts(folder)}}
    return part, lines


def merge(parts):
    """세션 dict 여럿을 dataset.npz 모양 하나로."""
    keys = ("x", "y", "frame_ns", "row", "prompt", "class_id", "flag", "edge", "labelled", "v9", "wrist", "wrist_depth", "wrist_ok")
    ds = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    ds["session"] = np.concatenate([np.full(len(p["y"]), i, dtype=np.int16) for i, p in enumerate(parts)])
    ds["sessions"] = np.array([p["name"] for p in parts])
    ds["hands"] = np.array([p["hand"] for p in parts])
    ds["wrist_method"] = np.array([p["wrist_method"] for p in parts])
    ds["length_mm"] = np.array([p["length_mm"] for p in parts], dtype=np.float64)
    ds["rewear"] = np.array([p["rewear"] for p in parts], dtype=bool)
    ds["wall_ms"] = np.array([p["wall_ms"] for p in parts], dtype=np.float64)
    ds["input_files"] = np.array(verify_sync.INPUTS)
    ds["inputs"] = np.array([p["inputs"] for p in parts])
    names = {}
    for p in parts:
        names.update(p["label_names"])
    ids = sorted(names)
    ds["label_ids"] = np.array(ids, dtype=np.int16)
    ds["label_names"] = np.array([names[i] for i in ids])
    return ds


def build(data, names=(), bins=BINS, standardized=True):
    """(학습 자료 dict 또는 None, 보고 줄). 세션 이름이 없으면 쓸 수 있는 세션 모두. bins·standardized는 창 x를 만드는 방식(windows)."""
    parts, lines = [], []
    for name, why in hs_io.chosen(data, names):
        if not why:
            _, why = eligible(data, name)
        if why:
            lines.append(f"{name}: 건너뜀({why})")
            continue
        try:
            part, more = session_data(data, name, bins, standardized)
        except Exception as e:  # noqa: BLE001  한 세션이 넘어져도 다른 세션은 쓴다
            part, more = None, [f"{name}: 처리 중 오류로 건너뜀({type(e).__name__}: {e})"]
        lines += more
        if part is not None:
            parts.append(part)
    if not parts:
        return None, lines
    ds = merge(parts)
    ds["wear"] = wear_episodes(ds["wall_ms"], ds["rewear"], rewear_marks(data))
    return ds, lines


def rewear_marks(data):
    """phone/sessions의 모든 세션(휴지통·끝나지 않은 세션 포함) 중 meta.json rewear가 참인 것의 시작 wallMs들. 읽지 못하는 meta와 시각이 없는
    세션은 건너뛴다(착용 경계로 쓸 수 없다)."""
    marks = []
    for name in hs_io.session_names(data):
        try:
            meta = hs_io.read_json(os.path.join(hs_io.session_folder(data, name), "meta.json"))
        except (OSError, ValueError):
            continue
        wall = hs_io.num(meta.get("start"), "wallMs")
        if meta.get("rewear") and wall is not None:
            marks.append(wall)
    return marks


def wear_episodes(wall_ms, rewear, marks_ms=()):
    """세션마다 착용 번호(머리말 session_split, 계획 6 검토 고침 M1): 시각 순으로 rewear가 참인 세션에서 새 착용이 시작해 다음 그런 세션 전까지
    간다. marks_ms는 자료에 들지 않은 세션까지 포함한 rewear 참 세션의 시작 wallMs들이다(rewear_marks). 첫 표시 전 세션은 0. (S,) int64."""
    wall = np.asarray(wall_ms, dtype=np.float64)
    marked = np.asarray(rewear, dtype=bool)
    starts = np.array(sorted({float(m) for m in marks_ms} | {float(w) for w in wall[marked]}), dtype=np.float64)
    return np.searchsorted(starts, wall, side="right").astype(np.int64)


def time_split(t_ns, session, fractions=SPLIT, gap_sec=GAP_SEC):
    """(학습, 검증, 시험) 프레임 번호, 각각 세션·시각 순서. 세션마다 시각 순서로 앞 fractions[0]은 학습, 다음 fractions[1]은 검증,
    나머지는 시험이고, 검증·시험은 앞쪽 경계(그쪽 첫 프레임 시각) 뒤 gap_sec 안의 프레임을 버린다."""
    t_ns = np.asarray(t_ns, dtype=np.int64)
    session = np.asarray(session)
    out = ([], [], [])
    for s in np.unique(session):
        idx = np.flatnonzero(session == s)
        idx = idx[np.argsort(t_ns[idx], kind="stable")]
        t = t_ns[idx]
        k1 = min(int(round(fractions[0] * len(idx))), len(idx) - 1)
        k2 = min(int(round((fractions[0] + fractions[1]) * len(idx))), len(idx) - 1)
        gap = gap_sec * 1e9
        out[0].append(idx[:k1])
        out[1].append(idx[(t >= t[k1] + gap) & (t < t[k2])])
        out[2].append(idx[t >= t[k2] + gap])
    return tuple(np.concatenate(o).astype(np.int64) for o in out)


def block_roles(count, offset=0):
    """블록 번호 0…count−1의 역할 (count,) int8: 0 학습, 1 검증, 2 시험. 자리 = (블록 번호 + offset) mod BLOCK_PERIOD가 BLOCK_TEST·BLOCK_VAL에
    들면 시험·검증, 아니면 학습이다(고정 무늬라 씨앗값이 없다. offset은 무늬를 블록 단위로 돌린다)."""
    slot = (np.arange(count) + int(offset)) % BLOCK_PERIOD
    return np.where(np.isin(slot, BLOCK_TEST), 2, np.where(np.isin(slot, BLOCK_VAL), 1, 0)).astype(np.int8)


def block_assign(t_ns, session, block_sec=BLOCK_SEC, offset=0, gap_sec=GAP_SEC):
    """블록 나누기의 프레임별 배정: (블록 번호 (N,) int64, 역할 (N,) int8, 쓸 프레임 (N,) bool). 세션마다 그 세션의 첫 프레임 시각부터 block_sec 길이로 자르고,
    역할이 앞 블록과 달라진 블록은 시작 뒤 gap_sec 안의 프레임을 쓰지 않는다(time_split이 경계 뒤 2 s를 비우는 것과 같은 규칙. 창 1.2 s가
    다른 쪽 자료에 걸치지 않게). 세션의 첫 블록은 앞이 없어 비우지 않는다."""
    t_ns = np.asarray(t_ns, dtype=np.int64)
    session = np.asarray(session)
    block = np.zeros(len(t_ns), dtype=np.int64)
    role = np.zeros(len(t_ns), dtype=np.int8)
    keep = np.ones(len(t_ns), dtype=bool)
    for s in np.unique(session):
        idx = np.flatnonzero(session == s)
        sec = (t_ns[idx] - t_ns[idx].min()) / 1e9
        b = np.floor(sec / block_sec).astype(np.int64)
        roles = block_roles(int(b.max()) + 1, offset)
        changed = np.r_[False, roles[1:] != roles[:-1]][b]  # 앞 블록과 역할이 다른 블록의 프레임
        block[idx], role[idx] = b, roles[b]
        keep[idx] = ~(changed & (sec - b * block_sec < gap_sec))
    return block, role, keep


def block_split(t_ns, session, block_sec=BLOCK_SEC, offset=0, gap_sec=GAP_SEC):
    """(학습, 검증, 시험) 프레임 번호, 각각 세션·시각 순서(time_split과 같은 모양). 10 s 블록을 고정 무늬로 학습·검증·시험에 흩어 놓아
    세 자료가 세션 전체에 걸친다. 역할이 바뀌는 경계 뒤 gap_sec의 프레임은 버린다(block_assign)."""
    block, role, keep = block_assign(t_ns, session, block_sec, offset, gap_sec)
    order = np.lexsort((np.asarray(t_ns), np.asarray(session)))
    return tuple(order[keep[order] & (role[order] == r)].astype(np.int64) for r in range(3))


def session_split(t_ns, session, wall_ms, rewear, test=None, val=None, fractions=SPLIT, gap_sec=GAP_SEC, wear=None):
    """세션 단위 나누기(머리말): ((학습, 검증, 시험) 프레임 번호, 알릴 줄, (학습, 검증, 시험) 세션 번호 목록(각각 시각 순서)).
    wall_ms·rewear (S,)는 세션마다 시작 시각과 다시 찬 표시, wear (S,)는 착용 번호(dataset.npz wear — 없으면 rewear로 이 자료 안에서만 만든다:
    휴지통 세션의 표시는 모른다). test·val(세션 번호 목록)을 주면 그대로 쓰고, 주지 않은 쪽은 규칙을 따른다. 검증 세션이 빈 목록이면 학습
    세션마다 끝 fractions[1]을 검증으로 쓴다(앞 gap_sec 비움). 나눌 수 없으면 ValueError(이유)."""
    t_ns = np.asarray(t_ns, dtype=np.int64)
    session = np.asarray(session)
    count = len(wall_ms)
    if count < 2:
        raise ValueError(f"세션이 {count}개뿐이라 세션 단위로 나눌 수 없음 — 세션을 더 모으거나 --mode blocks(같은 착용 안 낙관적 상한)로")
    order = [int(s) for s in np.argsort(np.asarray(wall_ms, dtype=np.float64), kind="stable")]  # 시각 순서
    fixed_val = [int(s) for s in (val or [])]
    wear = wear_episodes(wall_ms, rewear) if wear is None else np.asarray(wear, dtype=np.int64)
    notes = []
    if test is None:
        pool = [s for s in order if s not in fixed_val]
        episodes = sorted({int(wear[s]) for s in pool})
        if len(episodes) >= 2:
            test = [s for s in pool if int(wear[s]) == episodes[-1]]
            notes.append(f"시험 = 마지막 착용(워치를 다시 찬 뒤, meta rewear)의 세션 {len(test)}개 통째로(자료의 착용 {len(episodes)}개 중)"
                         " — 같은 착용의 세션은 학습·검증에 들지 않는다")
        else:
            k = max(1, int(round(fractions[2] * count)))
            test = pool[-k:]
            notes.append(f"착용이 하나뿐이라(사이에 다시 찬 표시 meta rewear가 없음) 시각 순 마지막 {k}세션(세션 수의 {fractions[2]:.0%}, 적어도 하나)을"
                         " 시험으로 — 같은 착용 안이라 낙관적이다")
    test = [s for s in order if s in [int(v) for v in test]]
    if set(test) & set(fixed_val):
        raise ValueError("같은 세션을 시험과 검증에 함께 둘 수 없음")
    rest = [s for s in order if s not in test and s not in fixed_val]
    val = [s for s in order if s in fixed_val] if val is not None else rest[-1:] if len(rest) >= 2 else []
    train = [s for s in order if s not in test and s not in val]
    if not train:
        raise ValueError("학습에 남는 세션이 없음")
    held = {int(wear[s]) for s in test}
    leaked = [s for s in train + val if int(wear[s]) in held]
    if leaked:
        notes.append(f"시험과 같은 착용의 세션 {len(leaked)}개가 학습·검증에 있음(고른 세션 때문 — 같은 착용 누수라 시험이 낙관적이다)")

    def frames(sets):
        idx = np.flatnonzero(np.isin(session, sets))
        return idx[np.lexsort((t_ns[idx], session[idx]))].astype(np.int64)

    parts = [frames(train), frames(val), frames(test)]
    if not val:
        notes.append(f"시험이 아닌 세션이 하나뿐이라 검증은 학습 세션의 끝 {fractions[1]:.0%}(앞 {gap_sec:g} s 비움) — 검증 손실이 같은 세션 안이라 낙관적이다")
        keep, held = [], []
        for s in train:
            idx = frames([s])
            t = t_ns[idx]
            k = min(int(round((1.0 - fractions[1]) * len(idx))), len(idx) - 1)
            keep.append(idx[:k])
            held.append(idx[t >= t[k] + gap_sec * 1e9])
        parts[0], parts[1] = np.concatenate(keep), np.concatenate(held)
    return tuple(parts), notes, (train, val, test)


def save(path, ds, with_x=False):
    """dataset.npz를 임시 파일에 다 쓴 뒤 이름을 바꾼다. 창 x는 with_x일 때만 쓴다(머리말)."""
    tmp = path[: -len(".npz")] + ".tmp.npz"
    np.savez(tmp, **{k: v for k, v in ds.items() if with_x or k != "x"})
    os.replace(tmp, path)


def load(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="세션들에서 학습 자료(창·목표·라벨)를 만든다")
    parser.add_argument("sessions", nargs="*", help="폰 세션 이름(없으면 쓸 수 있는 세션 모두)")
    parser.add_argument("--data", default=hs_io.DEFAULT_DATA)
    parser.add_argument("--out", help="dataset.npz 경로(없으면 보고만 하고 쓰지 않음)")
    parser.add_argument("--save-x", action="store_true", help="창 x도 dataset.npz에 쓴다(세션당 약 170 MB)")
    args = parser.parse_args()
    ds, lines = build(args.data, args.sessions)
    print("\n".join(lines))
    if ds is None:
        print("쓸 세션이 없음")
        return 1
    if len(ds["sessions"]) >= 2:  # train.py의 기본 나누기
        (train, val, test), notes, _ = session_split(ds["frame_ns"], ds["session"], ds["wall_ms"], ds["rewear"], wear=ds.get("wear"))
        how = "세션 나누기(" + "; ".join(notes) + ")"
    else:
        train, val, test = block_split(ds["frame_ns"], ds["session"])
        how = f"블록 나누기({BLOCK_SEC:g} s 블록, 경계마다 {GAP_SEC:g} s 비움)"
    print(f"모두 {len(ds['y'])}프레임, 세션 {len(ds['sessions'])}개, {how}: 학습 {len(train)} · 검증 {len(val)} · 시험 {len(test)}")
    if args.out:
        save(args.out, ds, with_x=args.save_x)
        print(f"저장: {args.out}{'' if args.save_x else '(창 x 없이)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
