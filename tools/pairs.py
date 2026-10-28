"""세션마다 카메라 프레임 ↔ 프로파일 줄 대응표 pairs.csv를 만든다(손 소나 계획 3, 설계 §8·§8.1).

    python -I tools/pairs.py [--data 데이터 폴더] [세션 이름 …]   (profiles.py와 verify_sync.py를 먼저 돌린다)

pairs.csv 한 줄 = landmarks.csv 데이터 줄 하나(열이 130개가 아닌 잘린 줄은 hs_io가 버린다).
- frame: landmarks.csv 데이터 줄 번호(0부터, 버린 줄은 세지 않음).
- video: meta video.ok이고 video.frames == stats.frames == 데이터 줄 수(버린 줄 없음)일 때만 frame과 같은 번호(영상 n번째 프레임 =
  줄 n, 설계 §8.1), 아니면 빈 칸. 영상은 가변 프레임률이라 시각으로 찾지 말고 처음부터 차례로 디코드해 센다.
- frameNs: 카메라 시각(폰 elapsedNs). soundNs = frameNs − 카메라–소리 지연(sync.json lagMs. 없으면 0으로 두고 알린다).
- row: soundNs에 가장 가까운 원본 프로파일 줄(profiles.npy의 열 번호). 프로파일 밖이면 −1. dtMs = 그 줄 가운데 시각 − soundNs.
  모델 입력 창(96줄)은 이 줄에서 끝나는 줄들로 자른다.
- hand: 추적하는 손이 있고 손목이 이어지면 1(손이 둘 보인 줄은 다른 손을 골랐을 수 있다). direct: 그 줄에 직접 경로가 있으면 1.
- prompt·classId: frameNs를 담은 prompts.csv 줄과 그 동작 번호(화면에 보인 시각과 같은 폰 시계라 지연을 빼지 않는다). 없으면 −1과 빈 칸.
  edge: 그 줄 시작 0.5 s 안이면 1(화면 표시와 사람 반응이 늦어 동작 경계 앞 0.3–0.5 s는 버린다, 설계 §8.1).
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

COLUMNS = ["frame", "frameNs", "soundNs", "row", "dtMs", "hand", "direct", "prompt", "classId", "edge", "video"]
EDGE_SEC = 0.5  # 동작 줄 시작 뒤 이 안의 프레임은 경계로 본다
GAP_SEC = 0.1  # frameNs 간격이 이보다 길면 카메라가 끊긴 것으로 센다
ROW_NS = 600 / 48_000 * 1e9  # 프로파일 줄 하나(12.5 ms)


def nearest_rows(row_ns, t_ns):
    """(가장 가까운 줄 번호, 줄 가운데 − t ns). 첫 줄 앞·마지막 줄 뒤로 반 줄 넘게 벗어나면 −1."""
    j = np.clip(np.searchsorted(row_ns, t_ns), 1, len(row_ns) - 1)
    left = np.abs(t_ns - row_ns[j - 1]) <= np.abs(row_ns[j] - t_ns)
    row = np.where(left, j - 1, j)
    dt = row_ns[row] - t_ns
    return np.where(np.abs(dt) <= ROW_NS / 2 + 1, row, -1), dt


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


def run(data, name):
    """세션 하나: (보고 줄, 됐는가)."""
    out = os.path.join(data, "processed", name)
    if not os.path.exists(os.path.join(out, "rows.npz")):
        return [f"{name}: 건너뜀(rows.npz 없음 — profiles.py를 먼저)"], False
    folder = hs_io.session_folder(data, name)
    meta = hs_io.read_json(os.path.join(folder, "meta.json"))
    rows = np.load(os.path.join(out, "rows.npz"), allow_pickle=False)
    row_ns, ok = rows["ns"], rows["ok"]
    sync_path = os.path.join(out, "sync.json")
    lag_ns = hs_io.num(hs_io.read_json(sync_path), "lagMs", 0.0) * 1e6 if os.path.exists(sync_path) else 0.0
    marks = hs_io.read_landmarks(folder)
    prompts = hs_io.read_prompts(folder)
    frame_ns = marks.frame_ns.astype(np.float64)
    sound_ns = frame_ns - lag_ns
    row, dt = nearest_rows(row_ns, sound_ns)
    direct = (row >= 0) & ok[np.clip(row, 0, len(ok) - 1)]
    prompt = prompt_rows(prompts, frame_ns)
    class_id = np.array([prompts[k]["classId"] if k >= 0 else 0 for k in prompt])
    edge = np.array([k >= 0 and t < prompts[k]["startNs"] + EDGE_SEC * 1e9 for k, t in zip(prompt, frame_ns)])
    video, stats = meta.get("video") or {}, meta.get("stats") or {}
    count = len(frame_ns)
    video_ok = bool(video.get("ok")) and int(hs_io.num(video, "frames", -1)) == int(hs_io.num(stats, "frames", -2)) == count and marks.dropped == 0
    with open(os.path.join(out, "pairs.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(COLUMNS)
        for n in range(count):
            k = int(prompt[n])
            w.writerow([n, int(marks.frame_ns[n]), int(round(sound_ns[n])), int(row[n]), f"{dt[n] / 1e6:.3f}" if row[n] >= 0 else "",
                        int(marks.valid[n]), int(direct[n]), k, class_id[n] if k >= 0 else "", int(edge[n]), n if video_ok else ""])
    gaps = np.diff(frame_ns) / 1e9
    camera = meta.get("camera") or {}
    measured = camera.get("measuredFps")
    joints = marks.valid & direct
    labelled = joints & (prompt >= 0) & ~edge & (class_id != -2)  # 일시정지 줄(classId −2)은 빼고, 자유 동작(−1)은 넣는다
    lines = [f"{name}: 카메라 프레임 {count}개(버린 줄 {marks.dropped}), 초당 {(count - 1) / ((frame_ns[-1] - frame_ns[0]) / 1e9):.1f}"
             f"(meta measuredFps {measured if measured is not None else '없음'}), 손 있음 {marks.valid.mean() * 100:.1f}%,"
             f" 카메라 끊김(간격 {GAP_SEC} s 넘음) {int((gaps > GAP_SEC).sum())}번 {gaps[gaps > GAP_SEC].sum():.1f} s",
             f"  프로파일 줄에 든 프레임 {int((row >= 0).sum())}개, 지연 {lag_ns / 1e6:+.1f} ms 뺌" + ("" if os.path.exists(sync_path) else "(sync.json 없음: 0으로 둠)"),
             f"  관절 학습에 쓸 줄(손·직접 경로·프로파일 안) {int(joints.sum())}개, 동작 라벨까지 붙은 줄(경계 {EDGE_SEC} s·일시정지 뺌) {int(labelled.sum())}개",
             f"  영상 n = 줄 n: {'예' if video_ok else '아니오(video.ok·프레임 수·버린 줄을 볼 것)'}"]
    return lines, True


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="카메라 프레임 ↔ 프로파일 줄 대응표를 만든다")
    parser.add_argument("sessions", nargs="*", help="폰 세션 이름(없으면 모두)")
    parser.add_argument("--data", default=hs_io.DEFAULT_DATA)
    args = parser.parse_args()
    bad = 0
    for name, why in hs_io.chosen(args.data, args.sessions):
        if why:
            print(f"{name}: 건너뜀({why})")
            continue
        lines, done = run(args.data, name)
        print("\n".join(lines))
        bad += not done
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
