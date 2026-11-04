"""process.json(process_check.py)을 그린다: 거리별 움직일 때·멈췄을 때 변화 세기(공개 대 우리), 처리별 오르내림(실측 대 모형 예상).

    python -I tools/experiments/knobs/process_figure.py --out <저장소 밖 폴더>
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402  tools 경로를 붙이고 --out·--data·--pub를 읽는다
import hs_io  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
GROUPS = (("공개 WatchHand", lambda s: s["who"] == "public", "#d98a1c", "public"),
          ("우리 10-31 톱니", lambda s: s["who"] == "ours" and s["window"] == "none" and s["shape"] == "saw", "#2a9d8f", "ours_saw"),
          ("우리 11-01 삼각", lambda s: s["session"].startswith("20261101"), "#8a6fc4", "ours_tri"),
          ("우리 10-28 한 창 톱니", lambda s: s["who"] == "ours" and s["window"] == "hann", "#5d8fd1", None))
PROCS = (("now", "지금 처리\n||원본|| 차분"), ("env", "포락선\n차분"), ("env/direct", "포락선 ÷\n직접 경로"),
         ("cplx/direct", "복소 ÷\n직접 경로"), ("cplx/fit", "복소, 줄 공통\n이득 빼기"))


def main():
    out = common.out("process.png")
    R = json.load(open(common.out("process.json"), encoding="utf-8"))
    cm = (np.array(R["distanceBins"]) - 300) * 343.0 / 48_000 / 2 * 100
    plt = hs_io.pyplot()
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": BASE,
                         "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 9, "text.color": INK})
    fig = plt.figure(figsize=(14, 11))
    fig.text(0.04, 0.965, "차이는 어디에 있나 — 거리별 변화 세기와, 줄 공통 흔들림을 빼는 처리", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.04, 0.945, "같은 원본(3줄 중앙값을 건 부호 있는 상관)에서 처리만 바꿨다. 움직임·멈춤은 MediaPipe 정답 속도로 가른 프레임(> 80 mm/s, < 15 mm/s).",
             fontsize=9.2, color=INK2)
    groups = [(name, [s for s in R["sessions"] if cond(s)], color, key) for name, cond, color, key in GROUPS]

    # ① 거리별 변화 세기(지금 처리)
    ax = fig.add_axes([0.06, 0.56, 0.9, 0.33])
    ax.axvspan(5, 21.4, color="#f1f0ea", zorder=0)
    ax.text(13.2, -47.5, "손 자리(5–21 cm, 값을 재는 곳)", ha="center", fontsize=8.6, color=INK2)
    for name, sel, color, _ in groups[:3]:
        for kind, ls, lw in (("move", "-", 1.8), ("rest", (0, (3, 2)), 1.2)):
            M = np.median([np.array(s["distance"]["now"][kind]) for s in sel], axis=0)
            ax.plot(cm, M, color=color, ls=ls, lw=lw, label=f"{name} {'움직일 때' if kind == 'move' else '멈췄을 때'}")
    pub = np.median([np.array(s["distance"]["now"]["move"]) for s in groups[0][1]], axis=0)
    saw = np.median([np.array(s["distance"]["now"]["move"]) for s in groups[1][1]], axis=0)
    band = (cm >= 5) & (cm < 15)
    gap = 10 * np.log10(np.mean(10 ** (pub[band] / 10))) - 10 * np.log10(np.mean(10 ** (saw[band] / 10)))
    ax.annotate(f"움직일 때 5–15 cm: 우리 톱니가 공개보다 {gap:.1f} dB 약하다", xy=(10, saw[np.argmin(np.abs(cm - 10))]), xytext=(23, -54),
                fontsize=9, color=groups[1][2], arrowprops=dict(arrowstyle="-", color=groups[1][2], lw=0.8))
    ax.annotate("멈췄을 때는 톱니끼리 비슷(10–15 cm는 우리가 더 낮다)", xy=(12, -69.6), xytext=(23, -76), fontsize=9, color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_xlim(cm[0], cm[-1])
    ax.set_ylim(-80, -45)
    ax.set_xlabel("직접 경로에서 거리(cm)")
    ax.set_ylabel("줄 사이 변화 세기(dB, 직접 경로 = 0)")
    ax.grid(color=GRID, lw=0.6)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="lower left")
    ax.set_title("① 거리별 변화 세기(세션 중앙값, 지금 처리): 차이는 '멈췄을 때 잡음'보다 '움직이는 손가락의 반사'에 있다", loc="left", fontsize=10.5, color=INK)

    # ② 처리별 오르내림 — 실측(점·선) 대 모형 예상(빈 네모)
    ax = fig.add_axes([0.06, 0.1, 0.9, 0.34])
    x = np.arange(len(PROCS))
    for gi, (name, sel, color, key) in enumerate(groups):
        vals = np.array([[s["procs"][p]["swing"] for p, _ in PROCS] for s in sel])
        off = (gi - 1.5) * 0.08
        for v in vals:
            ax.plot(x + off, v, "o", color=color, ms=2.5, alpha=0.35)
        med = np.median(vals, axis=0)
        ax.plot(x + off, med, "o-", color=color, lw=1.8, ms=7, label=f"{name} 실측(중앙값)")
        ax.text(x[-1] + off + 0.12, med[-1], f"{med[-1] - med[0]:+.1f} dB", va="center", fontsize=8.6, color=color)
        if key:
            sim = [R["sim"][key][p]["swing"] for p, _ in PROCS]
            ax.plot(x + off, sim, "s--", color=color, mfc="none", ms=7, lw=1.0, alpha=0.8, label=f"{name} 모형(줄 공통 흔들림)")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in PROCS], fontsize=8.6)
    ax.set_xlim(-0.4, len(PROCS) - 0.4)
    ax.set_ylabel("오르내림(dB)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(fontsize=7.8, frameon=False, ncol=2, loc="upper left")
    ax.set_title("② 줄 공통 세기·위상 흔들림을 빼는 처리: 모형은 우리에게 +4–5 dB를 예상했지만, 실측은 삼각 +1.5~2 dB, 톱니 +0.5~1 dB, 공개 -1~-2 dB",
                 loc="left", fontsize=10.5, color=INK)
    fig.text(0.04, 0.025, "→ 우리 손 자리의 멈춤 변화는 대부분 줄 전체가 함께 흔들리는 것이 아니다(그랬다면 ②의 처리로 지워졌다). 남는 큰 차이는 손가락 반사 자체의 세기라 처리로는 메우지 못한다.",
             fontsize=9.6, color=INK)
    for a in fig.axes:
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.savefig(out, dpi=130)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
