"""칸 분류 머리(--head bins)의 구조·손실 배선 그림. 아래 칸은 실제 값(loss_terms.json, loss_compare.json).

    python -I tools/experiments/knobs/bins_head_figure.py --out <저장소 밖 폴더>
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402  tools 경로를 붙이고 --out·--data·--pub를 읽는다
import hs_io  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
LEARN = ("#dfeaf8", "#2a78d6")  # 학습하는 층
CALC = ("#f1f0ea", "#898781")  # 매개변수 없는 계산
FIXED = ("#fbeed5", "#d98a1c")  # 학습 자료에서 정해 고정
LOSS = ("#fbe3e8", "#c2548a")  # 손실
TRUTH = ("#e3f2ef", "#2a9d8f")  # 정답


def box(ax, x, y, w, h, text, style, size=8.8, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.8", fc=style[0], ec=style[1], lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK, fontweight=weight, linespacing=1.35)


def arrow(ax, a, b, color=INK2, lw=1.3, style="-|>", rad=0.0):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}"))


def main():
    out = common.out("bins_head.png")
    T = json.load(open(common.out("loss_terms.json"), encoding="utf-8"))
    C = json.load(open(common.out("loss_compare.json"), encoding="utf-8"))
    plt = hs_io.pyplot()
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": BASE,
                         "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 9, "text.color": INK})
    fig = plt.figure(figsize=(14, 13.2))
    fig.text(0.03, 0.974, "칸 분류 머리(--head bins)의 구조 — 분류로 내고, 기댓값으로 좌표를 만들어 MSE를 건다", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.03, 0.955, "입력 1.2 s 창 → 관절 각도 30개를 각각 48칸으로 분류 → 칸 확률의 기댓값이 각도 → 고정된 손 뼈대로 순기구학 → 점 20개. 분류지만 출력은 연속값이다.",
             fontsize=9.6, color=INK2)
    ax = fig.add_axes([0.02, 0.40, 0.96, 0.54])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 54)
    ax.axis("off")

    # 위 줄: 흐름
    y, h = 43, 8.5
    box(ax, 0.5, y, 9.5, h, "입력\n2 × 96 × 60\n원본 · 차분\n(줄 96 · 칸 60)", CALC)
    box(ax, 12, y, 11, h, "몸통 FastViT-T12\n매개변수 6.53M\n(아래 펼침)\n출력 1024값", LEARN, weight="bold")
    box(ax, 25, y, 9.5, h, "선형 출력층\n1024 → 1440\n매개변수 1.48M", LEARN)
    box(ax, 36.5, y, 9, h, "로짓\n30각 × 48칸\n(1440을 접음)", CALC)
    box(ax, 47.5, y, 10, h, "softmax\n각도마다\n칸 확률 p\n(합 1)", CALC)
    box(ax, 59.5, y, 12, h, "기댓값(soft-argmax)\n각도 = Σ p × 칸 중심\n각도 30값\n(뼈 15 × 굽힘·벌림)", CALC)
    box(ax, 73.5, y, 11, h, "순기구학\n뿌리부터 뼈 방향을\n이어 점을 만든다", CALC)
    box(ax, 87, y, 12.5, h, "출력\n점 20개 × xyz = 60값\n(단위 손 손바닥 틀)", CALC, weight="bold")
    for a, b in ((10, 12), (23, 25), (34.5, 36.5), (45.5, 47.5), (57.5, 59.5), (71.5, 73.5), (84.5, 87)):
        arrow(ax, (a, y + h / 2), (b, y + h / 2))

    # 고정 값(학습 자료에서)
    box(ax, 59.5, 35.5, 12, 4.8, "칸 중심 30 × 48\n학습 0.5–99.5% 범위 + 여유", FIXED, size=8)
    arrow(ax, (65.5, 40.3), (65.5, y), color=FIXED[1])
    box(ax, 73.5, 35.5, 11, 4.8, "손바닥 점 5(평균)\n뼈 15 길이(중앙값)", FIXED, size=8)
    arrow(ax, (79, 40.3), (79, y), color=FIXED[1])

    # 몸통 펼침
    ax.add_patch(FancyBboxPatch((0.5, 26.5), 55, 12.5, boxstyle="round,pad=0.0,rounding_size=0.8", fc="#f6f9fd", ec=LEARN[1], lw=0.9, ls=(0, (3, 2))))
    ax.text(1.2, 37.7, "몸통 펼침(입력 96 × 60 기준 특징 지도 크기, 채널@높이×폭)", fontsize=8.6, color=INK2, va="center")
    stages = (("stem", "64@24×15"), ("단계 1\nRepMixer ×2", "64@24×15"), ("단계 2\nRepMixer ×2", "128@12×8"), ("단계 3\nRepMixer ×6", "256@6×4"),
              ("단계 4\nRepMixer ×2", "512@3×2"), ("마지막 합성곱", "1024@3×2"), ("전역 평균", "1024"))
    x0, w = 1.2, 6.8
    for i, (name, size) in enumerate(stages):
        box(ax, x0 + i * (w + 0.95), 28, w, 7.8, f"{name}\n{size}", LEARN if i < 6 else CALC, size=8)
        if i:
            arrow(ax, (x0 + i * (w + 0.95) - 0.95, 31.9), (x0 + i * (w + 0.95), 31.9))
    arrow(ax, (17.5, 39), (17.5, y), color=LEARN[1], style="-")

    # 손실 배선
    ax.text(0.5, 23.6, "손실 배선", fontsize=10, fontweight="bold", color=INK)
    box(ax, 73.5, 13.2, 26, 8.6, "좌표 손실(주인)\nMSE(점, 정답 점) + 0.1 × 속도 손실\n(이웃 프레임 차분끼리 MSE)", LOSS, weight="bold")
    box(ax, 40, 13.2, 22.5, 8.6, "칸 손실(보조)\n0.01 × 교차 엔트로피 -Σ q log p\n(각도 30개 평균)", LOSS, weight="bold")
    arrow(ax, (93, y), (93, 21.8), color=LOSS[1], rad=0)
    arrow(ax, (56.6, y), (56.6, 21.8), color=LOSS[1], rad=0)
    box(ax, 87, 2.2, 12.5, 6.6, "정답 점 60값\nMediaPipe·HaMeR를\n단위 손 틀로", TRUTH, size=8.2)
    arrow(ax, (93, 8.8), (93, 13.2), color=TRUTH[1])
    box(ax, 65.5, 2.2, 18.5, 6.6, "순기구학의 역(IK)\n정답 점 → 정답 각도 30값", TRUTH, size=8.2)
    arrow(ax, (87, 5.5), (84, 5.5), color=TRUTH[1])
    box(ax, 40, 2.2, 22.5, 6.6, "칸 정답 q\n정답 각도에 칸 폭 1.5배 가우스", TRUTH, size=8.2)
    arrow(ax, (65.5, 5.5), (62.5, 5.5), color=TRUTH[1])
    arrow(ax, (51, 8.8), (51, 13.2), color=TRUTH[1])
    ax.text(0.5, 18.2, "학습 손실 = 좌표 손실 + 칸 손실\n검증·조기 종료·평가 = 좌표만(칸 손실은 안 봄)", fontsize=9.4, color=INK, va="center", linespacing=1.5)
    ax.text(0.5, 7.2, "색: 파랑 = 학습하는 층 · 회색 = 매개변수 없는 계산\n노랑 = 학습 자료에서 정해 고정 · 초록 = 정답 · 분홍 = 손실", fontsize=8.4, color=INK2, va="center", linespacing=1.5)

    # 아래 ①: 실제 칸 분포
    ax = fig.add_axes([0.06, 0.06, 0.28, 0.27])
    e = T["example"]
    c = np.array(e["centersDeg"])
    width = c[1] - c[0]
    ax.bar(c, e["p"], width=width * 0.9, color=LEARN[1], alpha=0.75, label="모델 칸 확률 p")
    ax.plot(c, e["q"], color=TRUTH[1], lw=1.8, label="칸 정답 q")
    ax.axvline(e["expectDeg"], color=INK, lw=1.4, ls="-", label=f"기댓값 {e['expectDeg']:.1f}°(출력)")
    ax.axvline(e["trueDeg"], color=TRUTH[1], lw=1.4, ls=(0, (3, 2)), label=f"정답 {e['trueDeg']:.1f}°")
    ax.set_xlabel("각도(°)")
    ax.set_ylabel("확률")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(fontsize=7.6, frameon=False, loc="upper right")
    ax.set_title(f"① 실제 분포: {e['angle']}(시험 프레임, 칸 폭 {e['binWidthDeg']:.1f}°)", loc="left", fontsize=10, color=INK)

    # 아래 ②: 항별 크기·기울기
    ax = fig.add_axes([0.39, 0.06, 0.25, 0.27])
    ax.axis("off")
    ini, tr = T["init"], T["trained"]
    rows = [["값: 좌표 MSE", f"{ini['mse']:.4f}", f"{tr['mse']:.4f}"],
            ["값: 0.1 × 속도", f"{ini['vel']:.5f}", f"{tr['vel']:.5f}"],
            ["값: 0.01 × 칸", f"{ini['ce']:.4f}", f"{tr['ce']:.4f}"],
            ["기울기: 좌표 쪽", f"{ini['gradCoord']:.3f}", f"{tr['gradCoord']:.3f}"],
            ["기울기: 칸 쪽", f"{ini['gradCe']:.3f}", f"{tr['gradCe']:.3f}"],
            ["두 기울기 코사인", f"{ini['cos']:.2f}", f"{tr['cos']:.2f}"]]
    tbl = ax.table(cellText=rows, colLabels=["항", "학습 전", "학습 뒤"], loc="upper center", cellLoc="center", colWidths=[0.5, 0.25, 0.25])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.8)
    tbl.scale(1, 1.75)
    for (i, j), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        if i == 0:
            cell.set_facecolor("#f1f0ea")
            cell.set_text_props(fontweight="bold")
    ax.set_title("② 같은 미니배치에서 잰 항별 크기", loc="left", fontsize=10, color=INK)

    # 아래 ③: 구성별 결과
    ax = fig.add_axes([0.70, 0.09, 0.28, 0.24])
    items = C["conditions"]
    for i, it in enumerate(items):
        v = it["ratio"]
        ax.bar(i, np.mean(v), 0.62, color=it["color"], alpha=0.8)
        ax.plot([i] * len(v), v, "o", color=INK, ms=3)
        ax.text(i, max(v) + 0.012, f"{np.mean(v):.3f}\n{np.mean(it['mm']):.1f} mm", ha="center", va="bottom", fontsize=7.8, color=INK)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([it["label"] for it in items], fontsize=7.8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("평균 자세 대비 비(낮을수록 좋음)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("③ 출력·손실 구성별 결과(톱니 두 세션, 씨앗값 3개)", loc="left", fontsize=10, color=INK)
    for a in fig.axes:
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.savefig(out, dpi=130)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
