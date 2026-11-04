"""band.json(band_check.py)·sim_band.json(sim_band.py)을 그린다: 공개 데이터의 실제 대역, 대역 안 모양(이론 템플릿과 견줌),
18–21 kHz를 셋으로 나눈 값(실측·모의), 대역을 옮기거나 넓힌 시뮬레이션.

    python -I tools/experiments/knobs/band_figure.py --out <저장소 밖 폴더>
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402  tools 경로를 붙이고 --out·--data·--pub를 읽는다
import hs_io  # noqa: E402
import profiles as P  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import sosfreqz  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
GROUPS = (  # (이름, 조건, 색, 이론 템플릿 (창, 모양))
    ("공개 WatchHand", lambda s: s["who"] == "public", "#d98a1c", ("none", "saw")),
    ("우리 10-31 톱니", lambda s: s["who"] == "ours" and s["window"] == "none" and s["shape"] == "saw", "#2a9d8f", ("none", "saw")),
    ("우리 10-31·11-01 삼각", lambda s: s["who"] == "ours" and s["shape"] == "triangle", "#8a6fc4", ("none", "triangle")),
    ("우리 10-27·10-28 한 창 톱니", lambda s: s["who"] == "ours" and s["window"] == "hann", "#5d8fd1", ("hann", "saw")),
)
THIRDS = ("18–19", "19–20", "20–21")
SIM = (("public", "공개(맞춘 모형)", "#d98a1c"), ("ours_saw", "우리 톱니(맞춘 모형)", "#2a9d8f"), ("ours_tri", "우리 삼각(맞춘 모형)", "#8a6fc4"))


def theory_db(freq, window, shape):
    """경로가 평평할 때 프로파일 칸 방향 세기 = |X(f)|⁴ · |대역 필터|⁴(템플릿 = 송신, sosfiltfilt는 필터를 두 번)."""
    X = np.fft.rfft(P.tx_template(window, 0.5, shape))
    _, h = sosfreqz(P.SOS, worN=freq, fs=P.RATE)
    p = np.abs(X) ** 4 * np.abs(h) ** 4
    return 10 * np.log10(p / p.max() + 1e-30)


def mean_db(sel):
    return 10 * np.log10(np.mean([10 ** (np.array(s["spectrumDb"]) / 10) for s in sel], axis=0) + 1e-30)


def main():
    out = common.out("band.png")
    B = json.load(open(common.out("band.json"), encoding="utf-8"))
    SB = json.load(open(common.out("sim_band.json"), encoding="utf-8"))
    hz = np.array(B["freqHz"])
    f = hz / 1000
    plt = hs_io.pyplot()
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": BASE,
                         "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 9, "text.color": INK})
    fig = plt.figure(figsize=(14, 15.5))
    fig.text(0.04, 0.978, "대역 확인 — 공개 데이터는 정말 18–21 kHz인가, 대역을 바꾸면 무엇이 움직이나", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.04, 0.962, "프로파일 한 줄 = 받은 줄과 송신 템플릿의 부호 있는 원형 상관이라 칸 방향 DFT = Y(f)·conj(X(f))(80 Hz 간격). 받은 소리와 템플릿에 함께 있는 대역만 남아,"
             " 원시 소리 없이도 실제로 쓰인 대역이 보인다.", fontsize=9.2, color=INK2)
    groups = [(name, [s for s in B["sessions"] if cond(s)], color, th) for name, cond, color, th in GROUPS]
    pub = groups[0][1]

    # ① 0–24 kHz 전체
    ax = fig.add_axes([0.06, 0.715, 0.41, 0.21])
    ax.axvspan(18, 21, color="#f1f0ea", zorder=0)
    for name, sel, color, _ in groups:
        ax.plot(f, mean_db(sel), color=color, lw=1.3, label=f"{name}({len(sel)}세션)")
    outside = (1 - np.median([s["inbandShare"] for s in pub])) * 100
    ax.text(0.5, -4, f"공개: 17.9–21.1 kHz 밖 에너지 {outside:.4f}%\n(설정 파일의 fmcw18000_b3000_l600_s48k · bandpass 18000–21000과 같다)", fontsize=8.6, color=INK, va="top")
    ax.set_xlim(0, 24)
    ax.set_ylim(-100, 5)
    ax.set_xlabel("주파수(kHz)")
    ax.set_ylabel("칸 방향 세기(dB, 최대 = 0)")
    ax.grid(color=GRID, lw=0.6)
    ax.legend(fontsize=7.8, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.84))
    ax.set_title("① 0–24 kHz 전체: 18–21 kHz(회색) 밖은 모두 -55 dB 아래", loc="left", fontsize=10, color=INK)

    # ② 대역 안 모양 — 실측(실선) 대 이론 템플릿(점선)
    ax = fig.add_axes([0.55, 0.715, 0.42, 0.21])
    for s in pub:
        ax.plot(f, s["spectrumDb"], color=groups[0][2], lw=0.4, alpha=0.3)
    means = {}
    for name, sel, color, (window, shape) in groups:
        means[name] = mean_db(sel)
        ax.plot(f, means[name], color=color, lw=1.6)
        ax.plot(f, theory_db(hz, window, shape), color=color, lw=1.0, ls=(0, (3, 2)), alpha=0.9)
    dip = (hz >= 19_200) & (hz <= 20_200)
    i_dip = np.flatnonzero(dip)[np.argmin(means[groups[0][0]][dip])]
    notch = (hz >= 19_200) & (hz <= 19_700)
    i_notch = np.flatnonzero(notch)[np.argmin(means[groups[2][0]][notch])]
    i_hann = int(np.argmin(np.abs(hz - 18_500)))
    notes = (("공개(주황)만 19.2–20.2 kHz가 톱니 이론보다 6–10 dB 낮다(세션 10개 모두)", groups[0][2], i_dip, groups[0][0]),
             ("삼각(보라): 같은 주파수를 두 번 지나 엇갈리는 곳이 팬다(이론과 같은 자리)", groups[2][2], i_notch, groups[2][0]),
             ("한 창(파랑): 양 끝이 깎여 실제로 쓰는 폭은 약 1.5 kHz", groups[3][2], i_hann, groups[3][0]))
    for k, (text, color, i, name) in enumerate(notes):
        ax.annotate(text, xy=(f[i], means[name][i]), xytext=(17.62, -66 - 9 * k), fontsize=8.0, color=color, va="center",
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0, alpha=0.9),
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.7, shrinkA=0, shrinkB=2, relpos=(0.5, 1.0)))
    ax.set_xlim(17.5, 21.5)
    ax.set_ylim(-92, 4)
    ax.set_xlabel("주파수(kHz)")
    ax.set_ylabel("dB(최대 = 0)")
    ax.grid(color=GRID, lw=0.6)
    ax.plot([], [], color=INK2, lw=1.6, label="실측(묶음 평균, 공개는 세션별 가는 선도)")
    ax.plot([], [], color=INK2, lw=1.0, ls=(0, (3, 2)), label="이론: 템플릿 |X|⁴ × 18–21 kHz 필터(경로 평평)")
    ax.legend(fontsize=7.8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax.set_title("② 18–21 kHz 안 모양: 우리 둘은 이론 템플릿과 거의 같다(워치8 경로가 평평)", loc="left", fontsize=10, color=INK)

    # ③ 실측: 셋으로 나눈 오르내림 · ④ 정적 반사 세기 · ⑤ 같은 나누기를 모의에
    fig.text(0.04, 0.655, "③–⑤ 18–21 kHz를 1 kHz씩 셋으로 나눠(칸 방향 스펙트럼 조각만 남김 = 그 조각만 쓰는 처프로 상관) 같은 값을 쟀다. 조각은 거리 분해능이 17 cm로"
             " 거칠어 값 자체는 전체 대역과 다르다 — 조각끼리만 견준다.", fontsize=9.2, color=INK)
    panels = (("swing", "③ 실측 오르내림 — 아래쪽 조각이 크다(한 창만 평평)", "오르내림(dB)"),
              ("hand_raw", "④ 실측 손 자리 정적 반사 — 우리 톱니는 평평", "dB(그 조각의 직접 경로 = 0)"))
    for j, (key, title, ylabel) in enumerate(panels):
        ax = fig.add_axes([0.06 + 0.32 * j, 0.41, 0.26, 0.215])
        for gi, (name, sel, color, _) in enumerate(groups):
            vals = np.array([[s["bands"][b][key] for b in THIRDS] for s in sel])
            x = np.arange(3) + (gi - 1.5) * 0.06
            for v in vals:
                ax.plot(x, v, "o", color=color, ms=2.5, alpha=0.35)
            ax.plot(x, np.median(vals, axis=0), "o-", color=color, lw=1.6, ms=6, label=name)
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"{b} kHz" for b in THIRDS])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        if j == 0:
            ax.legend(fontsize=7.6, frameon=False, loc="upper right")
        else:
            ax.text(0.98, 0.03, "공개의 19–20 kHz 볼록 = ②의 골(그 조각의 직접 경로가 약함)", transform=ax.transAxes, ha="right", fontsize=7.8, color=groups[0][2])
    ax = fig.add_axes([0.71, 0.41, 0.26, 0.215])
    for key, name, color in SIM:
        v = [SB["thirds"][key][b]["swing"] for b in THIRDS]
        ax.plot(range(3), v, "s--", color=color, mfc="none", ms=6, lw=1.3, label=name)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{b} kHz" for b in THIRDS])
    ax.set_ylabel("오르내림(dB)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(fontsize=7.6, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.72))
    ax.text(0.5, 0.36, "모의는 가장자리 조각이 18·21 kHz 필터 끝에 걸려\n좁아지므로 가운데가 크다. 실측은 그래도 18–19가\n가장 크다 → 모형에 없는 주파수 의존",
            transform=ax.transAxes, ha="center", va="center", fontsize=7.8, color=INK2)
    ax.set_title("⑤ 같은 나누기를 모의에 — 아래가 커지지 않는다", loc="left", fontsize=10, color=INK)

    # ⑥ 시뮬레이션: 3 kHz 폭을 옮기기 · ⑦ 폭 바꾸기
    fig.text(0.04, 0.362, f"⑥–⑦ sim.json에 맞춘 모형을 대역만 바꿔 같은 처리에 넣었다(정적 반사 무늬·손가락 자리 {SB['seeds']}벌 평균, 반사체는 칸 사이)."
             " 스피커·마이크 응답, 주변 소음, 주파수별 반사 세기는 모형에 없다 — 처리와 위상만의 효과.", fontsize=9.2, color=INK)
    w = SB["window"]
    K = json.load(open(common.out("knobs.json"), encoding="utf-8"))
    med = lambda cond, k: float(np.median([r[k] for r in K if cond(r)]))  # noqa: E731
    hann = lambda r: r["session"].startswith("20261028") and r.get("volume") == 13  # noqa: E731
    none = lambda r: r["session"].startswith("20261031") and r.get("shape") == "saw"  # noqa: E731
    fig.text(0.04, 0.344, f"창만 바꾸면(모형, 18–21 kHz 그대로): 한 창은 실효 폭 약 1.5 kHz라 멈춤 {w['hann']['rest'] - w['none']['rest']:+.1f}·"
             f"오르내림 {w['hann']['swing'] - w['none']['swing']:+.1f}·바닥 {w['hann']['floor'] - w['none']['floor']:+.1f} dB."
             f" 실측 10-28(한 창) 대 10-31(창 없음)은 멈춤 {med(hann, 'rest') - med(none, 'rest'):+.1f}·바닥 {med(hann, 'floor') - med(none, 'floor'):+.1f} dB"
             " → 멈춤의 절반만 창 탓, 바닥은 창 탓 아님.", fontsize=9.2, color=INK)
    for j, (kind, title, xlabel) in enumerate((("center", "⑥ 모의: 3 kHz 폭의 자리를 옮기면 — 1 dB 안에서 거의 그대로", "대역(kHz)"),
                                               ("width", "⑦ 모의: 가운데 19.5 kHz에서 폭을 바꾸면 — 좁히면 손해, 넓혀도 이득은 작다", "대역폭(kHz)"))):
        ax = fig.add_axes([0.06 + 0.49 * j, 0.07, 0.42, 0.235])
        for key, name, color in SIM:
            rows = SB[kind][key]
            ax.plot(range(len(rows)), [r["swing"] for r in rows], "o-", color=color, lw=1.6, ms=6, label=name)
            below = key == "ours_tri"  # 우리 톱니 숫자와 겹치지 않게 점 아래에
            for i, r in enumerate(rows):
                ax.text(i, r["swing"] + (-0.4 if below else 0.35), f"{r['swing']:.1f}", ha="center", va="top" if below else "bottom", fontsize=7.6, color=color)
        rows = SB[kind]["public"]
        if kind == "center":
            labels = [f"{r['f0Hz'] / 1000:.1f}–{(r['f0Hz'] + r['bwHz']) / 1000:.1f}" for r in rows]
        else:
            labels = [f"{r['bwHz'] / 1000:.1f}\n({r['f0Hz'] / 1000:.2f}–{(r['f0Hz'] + r['bwHz']) / 1000:.2f})" for r in rows]
            saw = SB["width"]["ours_saw"]
            ax.text(0.98, 0.5, f"우리 톱니 멈춤 {saw[0]['rest']:.1f} → {saw[1]['rest']:.1f} → {saw[-1]['rest']:.1f} dB:"
                    "\n좁을수록 직접 경로가 손 자리에 번진다", transform=ax.transAxes, ha="right", va="center", fontsize=8.2, color=INK2)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(labels, fontsize=8.4)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("오르내림(dB)")
        ax.set_ylim(0, 17)
        ax.grid(color=GRID, lw=0.6)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        if j == 0:
            ax.legend(fontsize=7.8, frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.58))
    for a in fig.axes:
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.savefig(out, dpi=130)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
