"""first_arrival.json을 그린다: 공개 세션의 칸 300이 어떤 도착인가, 첫 도착에 맞추면 공개와 우리가 거리별로 어떻게 다른가.

    python -I tools/experiments/knobs/first_arrival_figure.py --out <저장소 밖 폴더>
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
BIN_CM = 343.0 / 48_000 / 2 * 100
GROUPS = (("공개: 칸 300이 둘째 도착", lambda s: s["who"] == "public" and s["firstBin"] < 290, "#d98a1c"),
          ("공개: 칸 300이 첫 도착", lambda s: s["who"] == "public" and s["firstBin"] >= 290, "#8c5a10"),
          ("우리 10-31 톱니", lambda s: s["who"] == "ours" and s["shape"] == "saw", "#2a9d8f"),
          ("우리 삼각", lambda s: s["who"] == "ours" and s["shape"] == "triangle", "#8a6fc4"))


def main():
    out = common.out("first_arrival.png")
    R = json.load(open(common.out("first_arrival.json"), encoding="utf-8"))
    span = np.array(R["spanBins"]) * BIN_CM
    level_cm = (np.arange(250, 400) - 300) * BIN_CM
    groups = [(n, [s for s in R["sessions"] if c(s)], col) for n, c, col in GROUPS]
    plt = hs_io.pyplot()
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": BASE,
                         "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 9, "text.color": INK})
    fig = plt.figure(figsize=(14, 11))
    fig.text(0.04, 0.965, "공개 세션의 칸 300은 한 가지 도착이 아니다 — 첫 도착에 맞추면 차이가 더 뚜렷하다", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.04, 0.945, "처리는 가장 센 봉우리를 칸 300에 둔다. 공개 10세션 중 6세션은 약 23칸(8.2 cm) 뒤 둘째 봉우리가 첫째보다 세서, 칸 300이 둘째 도착이다. 값은 모두 첫 도착 = 0 dB.",
             fontsize=9.2, color=INK2)

    # ① 칸 300 기준 포락선
    ax = fig.add_axes([0.06, 0.56, 0.42, 0.33])
    for name, sel, color in groups[:3]:
        for s in sel:
            ax.plot(level_cm, s["level"], color=color, lw=0.5, alpha=0.35)
        ax.plot(level_cm, np.median([s["level"] for s in sel], axis=0), color=color, lw=1.8, label=f"{name}({len(sel)}세션)")
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlim(-14, 25)
    ax.set_ylim(-32, 3)
    ax.set_xlabel("칸 300에서 거리(cm)")
    ax.set_ylabel("포락선(dB, 칸 300 = 0)")
    ax.grid(color=GRID, lw=0.6)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title("① 칸 300 기준(지금 저장된 그대로)", loc="left", fontsize=10.5, color=INK)

    # ② 첫 도착 기준 정적 포락선
    ax = fig.add_axes([0.55, 0.56, 0.42, 0.33])
    ax.axvspan(5, 21.4, color="#f1f0ea", zorder=0)
    for name, sel, color in groups:
        ax.plot(span, np.median([s["static"] for s in sel], axis=0), color=color, lw=1.8, label=name)
    ax.set_xlim(-10, 54)
    ax.set_ylim(-34, 6)
    ax.set_xlabel("첫 도착에서 거리(cm)")
    ax.set_ylabel("평균 포락선(dB, 첫 도착 = 0)")
    ax.grid(color=GRID, lw=0.6)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_title("② 첫 도착 기준 정적 반사: 8 cm 봉우리가 공개에만 세다(회색 = 손 자리)", loc="left", fontsize=10.5, color=INK)

    # ③ 첫 도착 기준 변화 세기
    ax = fig.add_axes([0.06, 0.08, 0.42, 0.36])
    ax.axvspan(5, 21.4, color="#f1f0ea", zorder=0)
    for name, sel, color in groups[:3]:
        ax.plot(span, np.median([s["distanceMove"] for s in sel], axis=0), color=color, lw=1.8, label=f"{name} 움직일 때")
        ax.plot(span, np.median([s["distanceRest"] for s in sel], axis=0), color=color, lw=1.1, ls=(0, (3, 2)), label=f"{name} 멈췄을 때")
    ax.set_xlim(-10, 54)
    ax.set_ylim(-80, -38)
    ax.set_xlabel("첫 도착에서 거리(cm)")
    ax.set_ylabel("줄 사이 변화 세기(dB, 첫 도착 = 0)")
    ax.grid(color=GRID, lw=0.6)
    ax.legend(fontsize=7.6, frameon=False, ncol=1, loc="upper right")
    ax.set_title("③ 첫 도착 기준 변화 세기: 손 자리에서 움직임이 약 10 dB 약하다", loc="left", fontsize=10.5, color=INK)

    # ④ 표
    ax = fig.add_axes([0.55, 0.08, 0.42, 0.36])
    ax.axis("off")
    head = ["묶음", "멈춤", "움직임", "오르내림", "손 자리\n|원본|", "5–10 cm\n정적", "5–15 cm\n움직임"]
    rows = []

    def band(sel, key, a, b):
        m = np.median([s[key] for s in sel], axis=0)
        return 10 * np.log10(np.mean(10 ** (m[(span >= a) & (span < b)] / 10)))
    for name, sel, _ in groups:
        v = {k: float(np.median([s[k] for s in sel])) for k in ("rest", "move", "swing", "handRaw")}
        rows.append([f"{name}({len(sel)})", f"{v['rest']:.1f}", f"{v['move']:.1f}", f"{v['swing']:.1f}", f"{v['handRaw']:.1f}", f"{band(sel, 'static', 5, 10):.1f}", f"{band(sel, 'distanceMove', 5, 15):.1f}"])
    tbl = ax.table(cellText=rows, colLabels=head, loc="upper center", cellLoc="center", colWidths=[0.34, 0.1, 0.11, 0.11, 0.11, 0.11, 0.12])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.6)
    tbl.scale(1, 2.0)
    for (i, j), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        if i == 0:
            cell.set_facecolor("#f1f0ea")
            cell.set_text_props(fontweight="bold")
    ax.text(0.0, 0.42, "바로잡음(2026-11-04)\n"
            "• 11-03의 '정적 반사는 공개와 같다'는 틀렸다.\n"
            "   우리 톱니 손 자리 |원본|은 공개보다 약 7–12 dB 낮다.\n"
            "• '5–15 cm에서 7.7 dB'는 칸 300 기준이라 작게 나왔다.\n"
            "   첫 도착 기준으로는 움직임이 약 11 dB 약하다.\n"
            "• 오르내림 차이는 그대로다(공개 8.0–13.0 dB, 우리 톱니 5.3 dB).\n"
            "• 값은 모두 첫 도착 기준이다. 직접 경로 자체의 세기 차이(워치7·8)도\n"
            "   같은 방향으로 보이므로, 반사가 실제로 약한지는 이것만으로 못 가른다.\n"
            "• 우리 톱니의 8 cm 봉우리(약 -10 dB)는 직접 경로의 첫 곁엽(이론 -13 dB)과 같은 자리다.",
            transform=ax.transAxes, fontsize=8.8, color=INK, va="top", linespacing=1.5)
    ax.set_title("④ 첫 도착 기준 값(dB, 세션 중앙값)", loc="left", fontsize=10.5, color=INK)
    for a in fig.axes:
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.savefig(out, dpi=130)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
