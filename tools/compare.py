"""우리 워치8 프로파일과 공개 데이터 왼손 표본(워치7)을 같은 축으로 나란히 그린 그림 한 장(손 소나 계획 3, 설계 §8).

    python -I tools/compare.py <세션 이름> [--data 데이터 폴더] [--public 공개 표본 폴더] [--out 그림.png] [--seconds 20] [--public-at 초]

보기만 하는 그림이다. 처리가 같은 종류의 데이터를 내는지 눈으로 본다. 지표를 내거나 모양을 맞추지 않는다(설계 §1, 사용자 2026-10-27).
둘 다 같은 처리의 값이다(부호 있는 상관, 직접 경로 칸 300, 시간 3줄 중앙값, 차분 |원본[t]| − |원본[t−1]|).
- 위: |원본| dB(각자 칸 300의 중앙값 기준, −50…0 dB). 아래: 차분(각자 그 구간 |값|의 99백분위로 나눔, −1…1).
- 세로축은 칸 290–360(직접 경로 앞 3.6 cm ~ 뒤 21.4 cm, 칸 하나 = 343 m/s ÷ 48 kHz ÷ 2 = 3.57 mm), 가로축은 같은 길이(기본 20 s).
- 우리 구간은 첫 동작(prompts.csv에서 segment가 1 이상인 첫 줄) 1 s 앞부터, 공개 표본은 --public-at부터. 공개 표본은 처음 4 s에
  준비 움직임이 커서(가장 조용한 10% 대비 11–17배) 기본을 12 s(손가락 구간이 시작하는 무렵)로 둔다.
기본 그림 자리는 processed/<세션>/compare.png다. 영상 프레임은 넣지 않는다.
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # python -I는 3.11부터 -P를 겸해 스크립트 폴더를 경로에 넣지 않는다
import hs_io  # noqa: E402  사용자 설치 numpy 경로도 여기서 붙는다
import numpy as np  # noqa: E402

BINS = slice(290, 361)
ROW0 = 300
CM_PER_BIN = 343.0 / 48_000 / 2 * 100
ROWS_PER_SEC = 80  # 줄 하나 = 600샘플 = 12.5 ms


def public_files(folder):
    """(원본 npy, 차분 npy) 경로. 공개 데이터 이름은 `…_fmcw_16bit_profiles.npy`·`…_fmcw_16bit_diff_profiles.npy`다."""
    names = sorted(os.listdir(folder))
    diff = [n for n in names if n.endswith("_diff_profiles.npy")]
    raw = [n for n in names if n.endswith("_profiles.npy") and not n.endswith("_diff_profiles.npy")]
    if not raw or not diff:
        raise FileNotFoundError(f"{folder}에 *_profiles.npy와 *_diff_profiles.npy가 없음")
    return os.path.join(folder, raw[0]), os.path.join(folder, diff[0])


def excerpt(raw, diff, first, rows):
    """원본 열 [first, first + rows)의 칸 290–360과, 같은 시각의 차분 열(차분 열 i는 원본 줄 i+1의 변화)."""
    first = int(np.clip(first, 1, raw.shape[1] - rows))
    return np.asarray(raw[BINS, first: first + rows], dtype=np.float64), np.asarray(diff[BINS, first - 1: first - 1 + rows], dtype=np.float64)


def draw(axes, raw, diff, seconds, title):
    """한 열(위 |원본| dB, 아래 차분)을 그리고 두 그림 객체를 돌려준다."""
    extent = [0, seconds, (BINS.start - ROW0 - 0.5) * CM_PER_BIN, (BINS.stop - 1 - ROW0 + 0.5) * CM_PER_BIN]
    db = 20 * np.log10(np.abs(raw) / np.median(np.abs(raw[ROW0 - BINS.start])) + 1e-12)
    top = axes[0].imshow(db, aspect="auto", origin="lower", cmap="Blues", vmin=-50, vmax=0, extent=extent)
    lim = float(np.percentile(np.abs(diff), 99)) or 1.0
    bottom = axes[1].imshow(diff / lim, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-1, vmax=1, extent=extent)
    axes[0].set_title(title, fontsize=11)
    axes[1].set_xlabel("시간(s)")
    return top, bottom


def run(data, name, public, out, seconds, public_at):
    folder = os.path.join(data, "processed", name)
    raw = np.load(os.path.join(folder, "profiles.npy"), mmap_mode="r", allow_pickle=False)
    diff = np.load(os.path.join(folder, "diff_profiles.npy"), mmap_mode="r", allow_pickle=False)
    row_ns = np.load(os.path.join(folder, "rows.npz"), allow_pickle=False)["ns"]
    prompts = hs_io.read_prompts(hs_io.session_folder(data, name))
    start = next((p["startNs"] for p in prompts if p["segment"] > 0), row_ns[0] + 1e9) - 1e9
    public_raw, public_diff = (np.load(p, mmap_mode="r", allow_pickle=False) for p in public_files(public))
    rows = min(int(seconds * ROWS_PER_SEC), raw.shape[1] - 1, public_raw.shape[1] - 1)  # 짧은 세션이면 그만큼만
    seconds = rows / ROWS_PER_SEC
    ours = excerpt(raw, diff, int(np.searchsorted(row_ns, start)), rows)
    theirs = excerpt(public_raw, public_diff, int(public_at * ROWS_PER_SEC), rows)
    plt = hs_io.pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(13, 6.5), sharex=True, sharey=True, constrained_layout=True)
    draw(axes[:, 0], *theirs, seconds, f"공개 데이터 왼손(워치7) {os.path.basename(os.path.normpath(public))}, {public_at:g} s부터")
    top, bottom = draw(axes[:, 1], *ours, seconds, f"우리(워치8) {name}, 첫 동작 1 s 앞부터")
    for ax in axes[:, 0]:
        ax.set_ylabel("직접 경로에서 거리(cm)")
    fig.colorbar(top, ax=axes[0, :], label="|원본| dB(칸 300 중앙값 기준)", shrink=0.9)
    fig.colorbar(bottom, ax=axes[1, :], label="차분 / 99백분위", shrink=0.9)
    fig.suptitle("같은 처리(칸 300 맞춤 · 3줄 중앙값 · 절댓값 차분), 보기만 하는 그림", fontsize=12)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="우리 프로파일과 공개 데이터 왼손 표본을 나란히 그린다")
    parser.add_argument("session", help="폰 세션 이름(profiles.py를 먼저 돌린다)")
    parser.add_argument("--data", default=hs_io.DEFAULT_DATA)
    parser.add_argument("--public", default=hs_io.DEFAULT_PUBLIC, help="공개 데이터 표본 폴더(*_profiles.npy와 *_diff_profiles.npy)")
    parser.add_argument("--out", help="그림 경로(없으면 processed/<세션>/compare.png)")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--public-at", type=float, default=12.0, help="공개 표본에서 그릴 처음(초)")
    args = parser.parse_args()
    out = args.out or os.path.join(args.data, "processed", args.session, "compare.png")
    print(f"그림: {run(args.data, args.session, args.public, out, args.seconds, args.public_at)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
