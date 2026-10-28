"""학습 자료: 카메라 프레임 하나에 에코 창 하나와 손 관절 목표 하나(손 소나 계획 4, 학습 설계 §2·§4).

    python -I tools/dataset.py [--data 데이터 폴더] [--out dataset.npz] [세션 이름 …]   (보통은 train.py가 부른다)

- 쓰는 세션(설계 §2): hs_io.chosen이 고른 세션 중 align.json이 "표지음 둘"(method two)이고, sync.json이 usable이며 그 입력 지문
  (align.json·rows.npz·landmarks.csv)이 지금 파일과 같고, pairs.csv가 있는 것. "표지음 하나" 세션은 쓰지 않는다.
- 창: pairs.csv의 짝 줄 r에서 끝나는 96줄(1.2 s, 인과적 정렬), 칸 300–359(60칸, 직접 경로부터 약 21.4 cm). 채널 0은 원본(부호 있는
  상관), 채널 1은 차분이다. 차분 열 j는 원본 줄 j+1의 변화라(pairs.py) 원본 열 r−95…r에 맞는 차분 열은 r−96…r−1이다.
  창마다 채널별로 평균을 빼고 표준편차로 나눈다. 모양 (2, 96, 60).
- 거르기: pairs.csv hand·direct가 1이고 video 칸이 있으며(영상 n = 줄 n), 창이 쓰는 원본 줄 r−96…r(차분 첫 열이 원본 줄 r−96도 쓰므로
  97줄)이 모두 rows.npz ok이고, 정답 거르기(joints.keep_frames: 점수·손목 튐·길이 ±3σ)를 지난 프레임만 쓴다. 길이 거르기가 동작
  종류별로 뺀 비율을 보고 줄에 적고, 한 종류가 치우쳐 빠지면(전체 비율의 2배 넘게) 그 사실도 적는다. 거울을 되돌린 손에서 엄지가
  손바닥 쪽인 프레임(joints.thumb_palmar)이 90% 아래인 세션은 손 방향 처리가 틀렸거나 반쯤 거울인 것이라 쓰지 않는다.
- 동작 라벨(평가용, 학습 목표 아님): pairs.csv prompt·classId·flag·edge. 라벨로 쓸 줄 = prompt ≥ 0, edge 0, 일시정지 아님,
  flag ≠ wrong(pairs.py와 같은 규칙). 자유 동작(−1)은 라벨이 있는 줄로 친다.
- 나누기(time_split, 설계 §4): 세션마다 프레임 시각 순서로 앞 70% 학습, 다음 15% 검증, 나머지 시험. 검증·시험은 앞쪽 경계 뒤
  2 s 안의 프레임을 버린다(창 1.2 s가 앞쪽 정답 시각과 겹치지 않게).
dataset.npz(allow_pickle 없이 읽힘): x (N, 2, 96, 60) float32, y (N, 60) float32(단위 손 손바닥 틀, 점 1–20 × xyz),
frame_ns (N,) int64, session (N,) int16, row (N,) int64, prompt (N,) int32, class_id (N,) int16(라벨 없음 −99), flag (N,) 문자열,
edge·labelled (N,) bool, v9 (N, 3) float32(m, 거울을 되돌린 카메라 좌표의 p9 − p0), sessions·hands (S,) 문자열,
length_mm (S,) float64(세션의 L), label_ids (K,) int16, label_names (K,) 문자열.
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

WINDOW = 96  # 창 줄 수(1.2 s)
BINS = slice(300, 360)  # 칸 300–359(설계 §8: 첫 판은 칸 300을 넣는다)
SPLIT = (0.70, 0.15, 0.15)  # 학습·검증·시험
GAP_SEC = 2.0  # 나눈 경계마다 비우는 시간
MIN_PALMAR = 0.9  # 거울을 되돌린 손에서 엄지가 손바닥 쪽인 프레임 비율 하한. 20261027-235412는 99.66%(되돌리기 전 0.34%)라 여유가 크고, 0.5면 프레임 40%가 뒤집힌 세션도 지난다
KINDS = (("finger", "손가락 굽히기"), ("asl", "ASL 숫자"), ("wrist", "손목"), ("free", "자유 동작"))  # 동작 종류: classId 0–4 · 5–14 · 15–17 · −1
KIND_KEYS = tuple(k for k, _ in KINDS) + ("none",)  # 마지막 none = 라벨 없음(동작 시작 0.5 s·쉬는 때·동작 밖)
SKEW_RATIO = 2.0  # 길이 거르기가 한 동작 종류에서 전체 비율의 이 배 넘게 빼면 치우쳤다고 알린다(20261027-235412: 손목 10%, 전체 2%)
SKEW_MIN_FRAMES = 5  # 몇 프레임뿐인 우연은 알리지 않는다


def kind_of(class_id, labelled):
    """프레임마다 동작 종류 번호 (N,): 0 손가락 굽히기(classId 0–4), 1 ASL 숫자(5–14), 2 손목(15–17), 3 자유 동작(−1), 4 라벨 없음."""
    cid = np.asarray(class_id)
    return np.where(~np.asarray(labelled, dtype=bool), 4, np.where(cid == -1, 3, np.where(cid >= 15, 2, np.where(cid >= 5, 1, 0))))


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

    return {"frameNs": ints("frameNs", 0), "row": ints("row", -1), "hand": ints("hand", 0), "direct": ints("direct", 0),
            "prompt": ints("prompt", -1), "classId": ints("classId", pairs.NO_CLASS), "edge": ints("edge", 0),
            "video": ints("video", -1), "flag": np.array([r["flag"] for r in rows], dtype=str)}


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


def windows(raw, diff, rows, block=512):
    """모델 입력 (N, 2, 96, 60) float32. raw (600, R)·diff (600, R−1)는 profiles.npy·diff_profiles.npy(mmap이어도 된다)."""
    raw60 = np.asarray(raw[BINS], dtype=np.float64).T  # (R, 60)
    diff60 = np.asarray(diff[BINS], dtype=np.float64).T  # (R − 1, 60)
    raw_cols, diff_cols = window_columns(rows)
    out = np.empty((len(raw_cols), 2, WINDOW, raw60.shape[1]), dtype=np.float32)
    for a in range(0, len(raw_cols), block):
        out[a: a + block, 0] = standardize(raw60[raw_cols[a: a + block]])
        out[a: a + block, 1] = standardize(diff60[diff_cols[a: a + block]])
    return out


def session_data(data, name):
    """세션 하나: (학습 자료 dict 또는 None, 보고 줄)."""
    out = os.path.join(data, "processed", name)
    folder = hs_io.session_folder(data, name)
    p = read_pairs(os.path.join(out, "pairs.csv"))
    ok = np.load(os.path.join(out, "rows.npz"), allow_pickle=False)["ok"]
    raw = np.load(os.path.join(out, "profiles.npy"), mmap_mode="r", allow_pickle=False)
    diff = np.load(os.path.join(out, "diff_profiles.npy"), mmap_mode="r", allow_pickle=False)
    if raw.shape != (600, len(ok)) or diff.shape != (600, len(ok) - 1):
        return None, [f"{name}: 건너뜀(profiles.npy {raw.shape}·diff_profiles.npy {diff.shape}와 rows.npz {len(ok)}줄이 맞지 않음 — profiles.py를 다시)"]
    marks = hs_io.read_landmarks(folder)
    if not np.array_equal(marks.frame_ns, p["frameNs"]):
        return None, [f"{name}: 건너뜀(pairs.csv와 landmarks.csv의 프레임이 다름 — pairs.py를 다시)"]
    hand = hs_io.read_session(folder).hand
    good, length = joints.keep_frames(marks.world, marks.score, marks.valid)
    base, _ = joints.scored(marks.world, marks.score, marks.valid)
    paired = (p["hand"] == 1) & (p["direct"] == 1) & (p["video"] >= 0)
    windowed = window_ok(ok, p["row"])
    use = paired & windowed & good
    scored = np.nan_to_num(marks.score, nan=-1.0) >= joints.MIN_SCORE
    lines = [f"{name}({hand}): 프레임 {len(use)}개 중 {int(use.sum())}개 씀",
             f"  뺀 것(겹침 있음): 짝 줄 못 씀(손·직접 경로·영상) {int((~paired).sum())}, 창 안 ok 아닌 줄·앞 줄 모자람 {int((~windowed).sum())},"
             f" 정답 거르기 {int((~good).sum())}(손 없음·점수 {joints.MIN_SCORE} 아래 {int((~scored).sum())},"
             f" 손이 둘인 줄 손목 튐 {int((scored & ~marks.valid).sum())},"
             f" 손목–새끼 MCP ±{joints.MAD_LIMIT:g}σ 밖 {int((base & ~good).sum())})"] + filter_lines(p["classId"], base, good, length)
    if not use.any():
        return None, lines
    y, v9, palmar = joints.targets(marks.world[use], hand)
    if palmar < MIN_PALMAR:
        return None, lines + [f"  건너뜀: 거울을 되돌린 손에서 엄지가 손바닥 쪽인 프레임 {palmar:.1%}(문턱 {MIN_PALMAR:.0%}. 실제 손 모양이 아니거나"
                              " 반쯤 거울 — 거울·손 처리를 볼 것)"]
    labelled = (p["prompt"] >= 0) & (p["edge"] == 0) & (p["classId"] != pairs.PAUSE_CLASS) & (p["flag"] != "wrong")
    length_mm = float(np.median(length[good])) * 1000.0
    lines.append(f"  L(손목–새끼 MCP 중앙값) {length_mm:.1f} mm, 엄지가 손바닥 쪽인 프레임 {palmar:.1%}, 라벨 붙은 프레임 {int((use & labelled).sum())}개")
    part = {"x": windows(raw, diff, p["row"][use]), "y": y.reshape(-1, 60).astype(np.float32), "frame_ns": p["frameNs"][use],
            "row": p["row"][use], "prompt": p["prompt"][use].astype(np.int32), "class_id": p["classId"][use].astype(np.int16),
            "flag": p["flag"][use], "edge": p["edge"][use] == 1, "labelled": labelled[use], "v9": v9.astype(np.float32),
            "name": name, "hand": hand, "length_mm": length_mm,
            "label_names": {q["classId"]: q["name"] for q in hs_io.read_prompts(folder)}}
    return part, lines


def merge(parts):
    """세션 dict 여럿을 dataset.npz 모양 하나로."""
    keys = ("x", "y", "frame_ns", "row", "prompt", "class_id", "flag", "edge", "labelled", "v9")
    ds = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    ds["session"] = np.concatenate([np.full(len(p["y"]), i, dtype=np.int16) for i, p in enumerate(parts)])
    ds["sessions"] = np.array([p["name"] for p in parts])
    ds["hands"] = np.array([p["hand"] for p in parts])
    ds["length_mm"] = np.array([p["length_mm"] for p in parts], dtype=np.float64)
    names = {}
    for p in parts:
        names.update(p["label_names"])
    ids = sorted(names)
    ds["label_ids"] = np.array(ids, dtype=np.int16)
    ds["label_names"] = np.array([names[i] for i in ids])
    return ds


def build(data, names=()):
    """(학습 자료 dict 또는 None, 보고 줄). 세션 이름이 없으면 쓸 수 있는 세션 모두."""
    parts, lines = [], []
    for name, why in hs_io.chosen(data, names):
        if not why:
            _, why = eligible(data, name)
        if why:
            lines.append(f"{name}: 건너뜀({why})")
            continue
        try:
            part, more = session_data(data, name)
        except Exception as e:  # noqa: BLE001  한 세션이 넘어져도 다른 세션은 쓴다
            part, more = None, [f"{name}: 처리 중 오류로 건너뜀({type(e).__name__}: {e})"]
        lines += more
        if part is not None:
            parts.append(part)
    return (merge(parts) if parts else None), lines


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


def save(path, ds):
    """dataset.npz를 임시 파일에 다 쓴 뒤 이름을 바꾼다."""
    tmp = path[: -len(".npz")] + ".tmp.npz"
    np.savez(tmp, **ds)
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
    args = parser.parse_args()
    ds, lines = build(args.data, args.sessions)
    print("\n".join(lines))
    if ds is None:
        print("쓸 세션이 없음")
        return 1
    train, val, test = time_split(ds["frame_ns"], ds["session"])
    print(f"모두 {len(ds['y'])}프레임, 시간 나누기 학습 {len(train)} · 검증 {len(val)} · 시험 {len(test)}(경계마다 {GAP_SEC:g} s 비움)")
    if args.out:
        save(args.out, ds)
        print(f"저장: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
