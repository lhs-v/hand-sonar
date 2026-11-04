"""knobs.json(knobs.py)을 녹음 설정 묶음별로 그린다: 소리 전체 흔들림, 멈췄을 때 손 자리, 먼 칸 바닥, 오르내림(움직일 때 − 멈췄을 때 손 자리).

    python -I tools/experiments/knobs/knobs_figure.py --out <저장소 밖 폴더>
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
GROUPS = (  # (이름표, 조건, 색)
    ("10-27\nVR·한 창·톱니\n볼륨 11", lambda r: r["session"].startswith("20261027"), "#9db8de"),
    ("10-28\nVR·한 창·톱니\n볼륨 13", lambda r: r["session"].startswith("20261028") and r.get("volume") == 13, "#5d8fd1"),
    ("10-28\nVR·한 창·톱니\n볼륨 15", lambda r: r["session"].startswith("20261028") and r.get("volume") == 15, "#1f4f9c"),
    ("10-31\n원음·창 없음·톱니\n세기 0.8", lambda r: r["session"].startswith("20261031") and r.get("shape") == "saw", "#2a9d8f"),
    ("10-31\n원음·창 없음·삼각\n세기 0.4·0.8", lambda r: r["session"].startswith("20261031") and r.get("shape") == "triangle", "#c2548a"),
    ("11-01\n원음·창 없음·삼각\n세기 0.5", lambda r: r["session"].startswith("20261101"), "#8a6fc4"),
    ("공개\nWatchHand", lambda r: r["who"] == "public", "#d98a1c"),
)
PANELS = (("jitter", "① 받은 소리 전체의 처프마다 흔들림(%, 직접 경로 세기)", "낮을수록 좋음", None),
          ("rest", "② 멈췄을 때 손 자리 |차분|(dB, 직접 경로 = 0)", "낮을수록 좋음", "rest_rn"),
          ("floor", "③ 먼 칸 바닥(dB, 54–107 cm)", "낮을수록 좋음", None),
          ("swing", "④ 오르내림 = 움직일 때 손 자리 - 멈췄을 때 손 자리(dB)", "클수록 좋음 — 손가락 움직임이 실제로 드러나는 정도", "swing_rn"))


def main():
    out = common.out("figure.png")
    rows = json.load(open(common.out("knobs.json"), encoding="utf-8"))
    plt = hs_io.pyplot()
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": BASE,
                         "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 9, "text.color": INK})
    fig, axes = plt.subplots(4, 1, figsize=(14, 15.5))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.07, hspace=0.55)
    fig.text(0.04, 0.975, "잡음 바닥·손 신호 값들은 무엇에 따라 움직였나 — 우리 세션 27개와 공개 10개를 같은 방식으로", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.04, 0.957, "점 = 세션(직접 경로 기준). 빈 동그라미 = 같은 세션에 줄 단위 정규화를 켠 값(처리 손잡이). VR = VOICE_RECOGNITION 음원, 원음 = UNPROCESSED."
             " 자세 기록(집·앉음·팔 받침 없음·시계줄 보통)은 모든 세션이 같아 가를 수 없다.", fontsize=9.2, color=INK2)
    for ax, (key, title, note, rn) in zip(axes, PANELS):
        for j, (label, cond, color) in enumerate(GROUPS):
            sel = [r for r in rows if cond(r)]
            v = np.array([r[key] for r in sel])
            x = j + np.linspace(-0.18, 0.18, len(v)) if len(v) > 1 else np.array([float(j)])
            ax.plot(x, v, "o", color=color, ms=7, zorder=3)
            if rn:
                w = np.array([r[rn] for r in sel])
                ax.plot(x, w, "o", mfc="none", mec=color, ms=7, zorder=3)
                for a, b, c in zip(x, v, w):
                    ax.plot([a, a], [b, c], color=color, lw=0.8, alpha=0.6)
            ax.text(j, np.nanmax(v if not rn else np.r_[v, [r[rn] for r in sel]]) + (0.03 if key == "jitter" else 0.8),
                    f"{np.median(v):.2f}%" if key == "jitter" else f"{np.median(v):.1f}", ha="center", va="bottom", fontsize=8.6, color=INK)
        ax.set_xticks(range(len(GROUPS)))
        ax.set_xticklabels([g[0] for g in GROUPS], fontsize=8.2)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(f"{title}  —  {note}", loc="left", fontsize=10.5, color=INK)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if key == "jitter":
            ax.set_ylim(0, 0.58)
        else:
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi + 0.18 * (hi - lo))  # 맨 위 숫자가 제목과 겹치지 않게
    fig.savefig(out, dpi=130)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
