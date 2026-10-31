"""세션마다 카메라 프레임 ↔ 프로파일 줄 대응표 pairs.csv를 만든다(손 소나 계획 3, 설계 §8·§8.1).

    python -I tools/pairs.py [--data 데이터 폴더] [세션 이름 …]   (profiles.py와 verify_sync.py를 먼저 돌린다)

짝짓는 조건(pairs.csv는 학습 정답이라, 틀린 지연으로 짝짓느니 짓지 않는다): sync.json이 있고, 카메라–소리 지연이 1프레임(33 ms)
안이며, 상관 봉우리가 verify_sync.MIN_PEAK 이상이고, sync.json에 적힌 입력 지문(align.json·rows.npz·landmarks.csv의 SHA-256)이
지금 파일과 같아야 한다. 아니면 pairs.csv를 쓰지 않고 끝 코드 1로 끝난다. 그때 옛 pairs.csv가 있으면 지우지 않고 pairs.stale.csv로
바꾼다(쓸 수 없는 세션 — 휴지통·끝나지 않음 — 도 같다). pairs.csv는 임시 파일에 다 쓴 뒤 이름을 바꿔 만든다.

pairs.csv 한 줄 = landmarks.csv 데이터 줄 하나(열이 130개가 아닌 잘린 줄은 hs_io가 버린다).
- frame: landmarks.csv 데이터 줄 번호(0부터). 버린 줄은 세지 않으므로 파일의 줄 번호와 다를 수 있다. 학습은 frameNs로 landmarks.csv와 잇는다.
- video: meta video.ok이고 video.frames == stats.frames == 데이터 줄 수(버린 줄 없음)일 때만 frame과 같은 번호(영상 n번째 프레임 =
  줄 n, 설계 §8.1), 아니면 빈 칸. 영상은 가변 프레임률이라 시각으로 찾지 말고 처음부터 차례로 디코드해 센다.
- frameNs: 카메라 시각(폰 elapsedNs). soundNs = frameNs − 카메라–소리 지연(sync.json lagMs, +면 카메라가 늦게 찍힘).
- row: soundNs에 가장 가까운 원본 프로파일 줄(profiles.npy의 열 번호. rows.npz ns는 줄 가운데 시각). 첫 줄 앞·마지막 줄 뒤로 실제 이웃
  줄 간격의 반 넘게 벗어나면 −1. dtMs = 그 줄 가운데 시각 − soundNs.
  모델 입력 창(96줄)은 이 줄에서 끝난다: 원본 열 [row − 95, row], 차분 열 [row − 96, row − 1]. 차분 열 j = |원본 j+1| − |원본 j|라
  원본 줄 j+1에 붙고, 시각은 (ns[j] + ns[j+1])/2다.
- hand: 추적하는 손이 있고 손목이 이어지면 1(손이 둘 보인 줄은 다른 손을 골랐을 수 있다). direct: 그 줄이 rows.npz ok(직접 경로가
  제자리이고 따라간 칸을 옮긴 둘레·녹음 계단 둘레가 아님)면 1. 창 안 다른 줄의 ok는 rows.npz에서 따로 본다.
- prompt·classId·flag: frameNs를 담은 prompts.csv 줄, 그 동작 번호, flag(wrong: 틀림 표시, pause: 일시정지로 끊긴 동작이나 일시정지
  구간). 화면에 보인 시각과 같은 폰 시계라 지연을 빼지 않는다. 담은 줄이 없으면 −1과 빈 칸 둘.
- edge: 동작 경계라 라벨로 쓰지 않는 줄이면 1: 동작 줄 시작 0.5 s 안(화면 표시와 사람 반응이 늦다, 설계 §8.1), 또는 일시정지·취소로
  끊긴 동작의 멈춘 때 앞 0.5 s(손이 화면으로 가는 중).
  동작 라벨로 쓸 줄 = prompt ≥ 0, edge = 0, classId ≠ −2(일시정지), flag ≠ wrong. 자유 동작(−1)은 넣는다.
학습 정답은 영상 없이 이 표와 landmarks.csv frameNs로 맞춘다(설계 §8.1). 카메라가 끊긴 시간(frameNs 간격 0.1 s 넘음)은 따로 세어
알린다(stats.handRatio는 받은 프레임 기준이라 끊긴 시간을 세지 않는다).
"""
import argparse
import csv
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # python -I는 3.11부터 -P를 겸해 스크립트 폴더를 경로에 넣지 않는다
import hs_io  # noqa: E402  사용자 설치 numpy 경로도 여기서 붙는다
import numpy as np  # noqa: E402
import verify_sync  # noqa: E402

COLUMNS = ["frame", "frameNs", "soundNs", "row", "dtMs", "hand", "direct", "prompt", "classId", "flag", "edge", "video"]
EDGE_SEC = 0.5  # 동작 줄 시작 뒤, 그리고 끊긴 동작의 멈춘 때 앞 이 안의 프레임은 경계로 본다
GAP_SEC = 0.1  # frameNs 간격이 이보다 길면 카메라가 끊긴 것으로 센다
PAUSE_CLASS = -2  # prompts.csv 일시정지 줄(shared Labels.PAUSE_ID)
NEUTRAL_CLASS = -3  # prompts.csv 기준 자세 정면 줄(shared Labels.NEUTRAL_ID). 동작 라벨이 아니라 dataset이 라벨에서 뺀다
NEUTRAL_SIDE_CLASS = -4  # 기준 자세 옆면 줄(shared Labels.NEUTRAL_SIDE_ID)
CALIB_CLASSES = (NEUTRAL_SIDE_CLASS, NEUTRAL_CLASS)  # 기준 자세 줄(옆면 → 정면 차례)
NO_CLASS = -99  # prompts.csv 줄 밖 프레임의 메모리 안 동작 번호(엄지 0·자유 동작 −1·일시정지 −2와 겹치지 않게. CSV에는 빈 칸)


def nearest_rows(row_ns, t_ns):
    """(가장 가까운 줄 번호, 줄 가운데 − t ns). 첫 줄 앞·마지막 줄 뒤로 실제 이웃 줄 간격의 반 넘게 벗어나면 −1
    (줄 사이에 든 시각은 늘 반 안이다. 명목 12.5 ms가 아니라 실제 간격 12.5 ms × (1 + 시계 차)를 쓴다)."""
    j = np.clip(np.searchsorted(row_ns, t_ns), 1, len(row_ns) - 1)
    left = np.abs(t_ns - row_ns[j - 1]) <= np.abs(row_ns[j] - t_ns)
    row = np.where(left, j - 1, j)
    dt = row_ns[row] - t_ns
    half = (row_ns[j] - row_ns[j - 1]) / 2
    return np.where(np.abs(dt) <= half + 1, row, -1), dt


def prompt_rows(prompts, t_ns):
    """각 시각을 담은 prompts.csv 줄 번호(없으면 −1). 줄끼리 시간이 겹치지 않는다."""
    if not prompts:
        return np.full(len(t_ns), -1)
    order = np.argsort([p["startNs"] for p in prompts])
    starts = np.array([prompts[k]["startNs"] for k in order], dtype=np.float64)
    ends = np.array([prompts[k]["endNs"] for k in order], dtype=np.float64)
    i = np.searchsorted(starts, t_ns, side="right") - 1
    ic = np.clip(i, 0, len(order) - 1)
    return np.where((i >= 0) & (t_ns < ends[ic]), order[ic], -1)


def cut_short(prompts, meta):
    """동작 줄마다 일시정지·취소로 끊겼는가. 일시정지로 끊긴 동작은 flag pause(일시정지 줄 classId −2와 다름), 취소로 끊긴 동작은
    끝이 세션 끝(meta end = 취소 시각)과 같다(shared SessionRunner)."""
    end_ns = hs_io.nanos(meta.get("end"), "elapsedNs")
    cancelled = bool(meta.get("cancelled"))
    return [p["classId"] != PAUSE_CLASS and (p["flag"] == "pause" or (cancelled and end_ns is not None and p["endNs"] >= end_ns - 1_000_000))
            for p in prompts]


def _pairs(data, name):
    """(보고 줄, 됐는가). 됐을 때만 pairs.csv를 쓴다."""
    out = os.path.join(data, "processed", name)
    if not os.path.exists(os.path.join(out, "rows.npz")):
        return [f"{name}: 짝짓지 않음(rows.npz 없음 — profiles.py를 먼저)"], False
    sync_path = os.path.join(out, "sync.json")
    if not os.path.exists(sync_path):
        return [f"{name}: 짝짓지 않음(sync.json 없음 — verify_sync.py를 먼저)"], False
    sync = hs_io.read_json(sync_path)
    good, why = verify_sync.usable(sync)
    if not good:
        return [f"{name}: 짝짓지 않음({why})"], False
    now = verify_sync.inputs(data, name)
    changed = [f for f in verify_sync.INPUTS if (sync.get("inputs") or {}).get(f) != now[f]]
    if changed:
        return [f"{name}: 짝짓지 않음(sync.json을 잰 뒤 바뀌었거나 지문이 없는 입력: {', '.join(changed)} — verify_sync.py를 다시)"], False
    folder = hs_io.session_folder(data, name)
    meta = hs_io.read_json(os.path.join(folder, "meta.json"))
    rows = np.load(os.path.join(out, "rows.npz"), allow_pickle=False)
    row_ns, ok = rows["ns"], rows["ok"]
    if len(row_ns) < 2:
        return [f"{name}: 짝짓지 않음(프로파일 줄이 {len(row_ns)}개뿐)"], False
    lag_ns = hs_io.num(sync, "lagMs") * 1e6
    marks = hs_io.read_landmarks(folder)
    count = len(marks.frame_ns)
    if count == 0:
        return [f"{name}: 짝짓지 않음(landmarks.csv에 쓸 데이터 줄이 없음, 버린 줄 {marks.dropped})"], False
    prompts = hs_io.read_prompts(folder)
    frame_ns = marks.frame_ns.astype(np.float64)
    sound_ns = frame_ns - lag_ns
    row, dt = nearest_rows(row_ns, sound_ns)
    direct = (row >= 0) & ok[np.clip(row, 0, len(ok) - 1)]
    prompt = prompt_rows(prompts, frame_ns)
    class_id = np.array([prompts[k]["classId"] if k >= 0 else NO_CLASS for k in prompt], dtype=np.int64)
    flag = [prompts[k]["flag"] if k >= 0 else "" for k in prompt]
    cut = cut_short(prompts, meta)
    edge = np.array([k >= 0 and (t < prompts[k]["startNs"] + EDGE_SEC * 1e9 or (cut[k] and t >= prompts[k]["endNs"] - EDGE_SEC * 1e9))
                     for k, t in zip(prompt, frame_ns)], dtype=bool)
    video, stats = meta.get("video") or {}, meta.get("stats") or {}
    video_ok = bool(video.get("ok")) and int(hs_io.num(video, "frames", -1)) == int(hs_io.num(stats, "frames", -2)) == count and marks.dropped == 0
    path = os.path.join(out, "pairs.csv")
    with open(path + ".tmp", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(COLUMNS)
        for n in range(count):
            k = int(prompt[n])
            w.writerow([n, int(marks.frame_ns[n]), int(round(sound_ns[n])), int(row[n]), f"{dt[n] / 1e6:.3f}" if row[n] >= 0 else "",
                        int(marks.valid[n]), int(direct[n]), k, int(class_id[n]) if k >= 0 else "", flag[n], int(edge[n]), n if video_ok else ""])
    os.replace(path + ".tmp", path)
    gaps = np.diff(frame_ns) / 1e9
    camera = meta.get("camera") or {}
    measured = camera.get("measuredFps")
    joints = marks.valid & direct
    labelled = joints & (prompt >= 0) & ~edge & (class_id != PAUSE_CLASS) & (np.array(flag) != "wrong")
    rate = f"{(count - 1) / ((frame_ns[-1] - frame_ns[0]) / 1e9):.1f}" if count > 1 and frame_ns[-1] > frame_ns[0] else "-"
    lines = [f"{name}: 카메라 프레임 {count}개(버린 줄 {marks.dropped}), 초당 {rate}"
             f"(meta measuredFps {measured if measured is not None else '없음'}), 손 있음 {marks.valid.mean() * 100:.1f}%,"
             f" 카메라 끊김(간격 {GAP_SEC} s 넘음) {int((gaps > GAP_SEC).sum())}번 {gaps[gaps > GAP_SEC].sum():.1f} s",
             f"  프로파일 줄에 든 프레임 {int((row >= 0).sum())}개, 지연 {lag_ns / 1e6:+.1f} ms 뺌(상관 봉우리 {hs_io.num(sync, 'peak'):.2f})",
             f"  관절 학습에 쓸 줄(손·직접 경로·프로파일 안) {int(joints.sum())}개, 동작 라벨까지 붙은 줄(경계 {EDGE_SEC} s·끊긴 동작 끝·"
             f"일시정지·틀림 뺌) {int(labelled.sum())}개, 끊긴 동작 {sum(cut)}개, 틀림 표시 줄 {sum(p['flag'] == 'wrong' for p in prompts)}개",
             f"  영상 n = 줄 n: {'예' if video_ok else '아니오(video.ok·프레임 수·버린 줄을 볼 것)'}"]
    return lines, True


def run(data, name):
    """세션 하나: (보고 줄, 됐는가). 안 됐으면(오류 포함) 옛 pairs.csv를 지우지 않고 pairs.stale.csv로 바꾼다."""
    lines, done = hs_io.guarded(name, _pairs, data, name)
    if not done:
        lines += retire(data, name)
    return lines, done


def retire(data, name):
    """옛 pairs.csv를 pairs.stale.csv로(학습이 옛 대응표를 집지 않게). 알릴 줄 목록."""
    if hs_io.retire(os.path.join(data, "processed", name, "pairs.csv")):
        return ["  옛 pairs.csv를 pairs.stale.csv로 바꿈(지금 처리로는 짝짓지 않는 세션)"]
    return []


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="카메라 프레임 ↔ 프로파일 줄 대응표를 만든다")
    parser.add_argument("sessions", nargs="*", help="폰 세션 이름(없으면 모두)")
    parser.add_argument("--data", default=hs_io.DEFAULT_DATA)
    args = parser.parse_args()
    bad = 0
    for name, why in hs_io.chosen(args.data, args.sessions):
        if why:
            print("\n".join([f"{name}: 건너뜀({why})"] + retire(args.data, name)))
            continue
        lines, done = run(args.data, name)
        print("\n".join(lines))
        bad += not done
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
