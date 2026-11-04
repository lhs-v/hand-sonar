"""sim.json(sim_floor.py)을 그린다: 이론(처프 자기상관), 손잡이 훑기, 실측 맞춤, 우리 삼각 → 공개로 하나씩 바꾸기.

    python -I tools/experiments/knobs/sim_figure.py --out <저장소 밖 폴더>
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
SAW_C, TRI_C, PUB_C = "#2a9d8f", "#8a6fc4", "#d98a1c"
BIN_CM = 343.0 / 48_000 / 2 * 100
MEAS = {"public": (0.12, -77.8, -69.1, 11.9), "ours_saw": (0.215, -77.3, -68.4, 5.3), "ours_tri": (0.29, -67.5, -63.9, 3.9)}


def main():
    out = common.out("sim.png")
    S = json.load(open(common.out("sim.json"), encoding="utf-8"))
    plt = hs_io.pyplot()
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": BASE,
                         "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 9, "text.color": INK})
    fig = plt.figure(figsize=(14, 15))
    fig.text(0.04, 0.978, "이론·시뮬레이션 — 실제 처리(profiles.compute)를 그대로 거친 가상 녹음으로 손잡이를 하나씩", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.04, 0.96, "수신 = 처프 × (정적 반사 + 움직이는 손가락) × (1 + 줄마다 세기 흔들림) × 줄마다 시각 흔들림 + 더해지는 잡음. 정적 반사는 실측 평균 |원본| 세기에 맞춤.",
             fontsize=9.2, color=INK2)

    # ① 이론: 처프 자기상관
    ax = fig.add_axes([0.06, 0.74, 0.42, 0.18])
    d = (np.arange(600) - 300) * BIN_CM
    for shape, c, name in (("saw", SAW_C, "톱니(18→21 kHz)"), ("triangle", TRI_C, "삼각(18→21→18 kHz)")):
        env = np.array(S["autocorr"][shape])
        ax.plot(d, 20 * np.log10(np.maximum(env, 1e-6)), color=c, lw=1.4, label=name)
    ax.axhline(-22.3, color=TRI_C, ls=":", lw=1)
    ax.text(105, -21.3, "실측 삼각 먼 칸 평균 |원본| -22.3 dB", ha="right", va="bottom", fontsize=8.2, color=TRI_C)
    ax.axhline(-37.7, color=SAW_C, ls=":", lw=1)
    ax.text(105, -36.7, "실측 톱니 먼 칸 -37~-38 dB(공개·우리)", ha="right", va="bottom", fontsize=8.2, color=SAW_C)
    ax.set_xlim(0, 107)
    ax.set_ylim(-70, 3)
    ax.set_xlabel("직접 경로에서 거리(cm)")
    ax.set_ylabel("자기상관 포락선(dB)")
    ax.grid(color=GRID, lw=0.6)
    ax.legend(fontsize=8.4, frameon=False, loc="lower left")
    ax.set_title("① 이론: 처프 하나의 자기상관 — 삼각은 오름×내림 엇갈림이 모든 거리에 약 -20 dB로 깔린다", loc="left", fontsize=10, color=INK)

    # ② 손잡이 훑기: 세기 흔들림 → 먼 칸 바닥(톱니·삼각)
    sw = S["sweeps"]
    ax = fig.add_axes([0.56, 0.74, 0.41, 0.18])
    for shape, c in (("saw", SAW_C), ("triangle", TRI_C)):
        rows = sw[f"sig_eps:{shape}"]
        ax.plot([r["value"] * 100 for r in rows], [r["floor"] for r in rows], "o-", color=c, label=f"{'톱니' if shape == 'saw' else '삼각'} 바닥")
        ax.plot([r["value"] * 100 for r in rows], [r["rest"] for r in rows], "s--", color=c, mfc="none", label=f"{'톱니' if shape == 'saw' else '삼각'} 멈췄을 때 손 자리")
    ax.set_xlabel("줄마다 세기 흔들림 σ(%)")
    ax.set_ylabel("dB(직접 경로 = 0)")
    ax.grid(color=GRID, lw=0.6)
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="upper left")
    ax.set_title("② 세기 흔들림을 키우면: 멈춤은 둘 다 오르고, 먼 칸 바닥은 삼각만 오른다", loc="left", fontsize=10, color=INK)

    # ③ 훑기: 시각 흔들림, 잡음, 손가락 세기 → 오르내림
    for j, (knob, label, scale) in enumerate((("sig_tau", "줄마다 시각 흔들림 σ(샘플)", 1.0), ("noise_db", "더해지는 잡음(dB, 직접 경로 rms 기준)", 1.0),
                                              ("a_f_db", "손가락 반사 세기(dB, 직접 경로 = 0)", 1.0))):
        ax = fig.add_axes([0.06 + 0.32 * j, 0.5, 0.27, 0.17])
        for shape, c in (("saw", SAW_C), ("triangle", TRI_C)):
            rows = sw[f"{knob}:{shape}"]
            x = [r["value"] * scale for r in rows]
            ax.plot(x, [r["swing"] for r in rows], "o-", color=c, label=f"{'톱니' if shape == 'saw' else '삼각'} 오르내림")
            ax.plot(x, [r["rest"] + 90 for r in rows], "s--", color=c, mfc="none", label=f"{'톱니' if shape == 'saw' else '삼각'} 멈춤 + 90")
        ax.set_xlabel(label, fontsize=8.6)
        ax.set_ylabel("dB")
        ax.grid(color=GRID, lw=0.6)
        if j == 0:
            ax.legend(fontsize=7.6, frameon=False, loc="upper right")
        ax.set_title(f"③-{j + 1} {label.split('(')[0]}", loc="left", fontsize=9.6, color=INK)
    fig.text(0.04, 0.685, "③ 다른 손잡이(기본: 세기 흔들림 0.15%, 잡음 -45 dB, 손가락 -40 dB): 시각 흔들림·잡음은 멈춤을 올려 오르내림을 깎고, 손가락 세기는 오르내림을 1:1로 올린다."
             " (멈춤은 + 90 dB로 옮겨 같은 축에)", fontsize=9.4, color=INK)

    # ④ 맞춤 표
    ax = fig.add_axes([0.04, 0.28, 0.92, 0.17])
    ax.axis("off")
    f = S["fit"]
    head = ["묶음", "세기 흔들림 ε(줄마다 σ)", "시각 흔들림(샘플)", "잡음", "흔들림 지표 실측 → 모의", "먼 칸 바닥 실측 → 모의", "멈춤 실측 → 모의", "손가락 세기(오르내림에 맞춤)"]
    rows = []
    for key, name in (("public", "공개(톱니)"), ("ours_saw", "우리 10-31 톱니"), ("ours_tri", "우리 11-01 삼각")):
        m = MEAS[key]
        r = f[key]
        rows.append([name, f"{r['eps'] * 100:.2f}%", f"{r['tau']:.4f}", f"{r['noiseDb']:.1f} dB" + (" (톱니 날 그대로)" if key == "ours_tri" else ""),
                     f"{m[0]:.2f}% → {r['jitter']:.2f}%", f"{m[1]:.1f} → {r['floor']:.1f}", f"{m[2]:.1f} → {r['rest']:.1f}", f"{f['fingerDb'][key]:.1f} dB"])
    tbl = ax.table(cellText=rows, colLabels=head, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.6)
    tbl.scale(1, 1.7)
    for (i, j), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        if i == 0:
            cell.set_facecolor("#f1f0ea")
            cell.set_text_props(fontweight="bold")
    pred = f["predictWithPublicFinger"]
    ax.text(0.0, -0.05, f"같은 손가락(공개에 맞춘 {f['fingerDb']['public']:.1f} dB)이면 우리 오르내림 예측: 톱니 {pred['ours_saw']['swing']:.1f} dB(실측 5.3), 삼각 {pred['ours_tri']['swing']:.1f} dB(실측 3.9)"
            f" — 실측이 4–6 dB 낮다 = 우리 손가락 반사가 약하다(우리에 맞추면 {f['fingerDb']['ours_saw']:.1f} · {f['fingerDb']['ours_tri']:.1f} dB). 잡음은 공개·우리 톱니가 같다.",
            transform=ax.transAxes, fontsize=9.2, color=INK, va="top")
    ax.set_title("④ 실측에 맞추기: ε은 흔들림 지표에, 잡음은 톱니 바닥에, 멈춤이 모자라면 시각 흔들림으로. 삼각 바닥은 맞추지 않고 예측(같은 기기 잡음)", loc="left", fontsize=10, color=INK)

    # ⑤ 하나씩 바꾸기
    ax = fig.add_axes([0.08, 0.05, 0.88, 0.17])
    steps = S["whatif"]
    vals = [s["swing"] for s in steps]
    ax.bar(range(len(steps)), vals, 0.6, color=[TRI_C] + [SAW_C] * (len(steps) - 2) + [PUB_C])
    for i, (s, v) in enumerate(zip(steps, vals)):
        delta = "" if i == 0 else f"\n({v - vals[i - 1]:+.1f})"
        ax.text(i, v + 0.3, f"{v:.1f} dB{delta}", ha="center", va="bottom", fontsize=8.6, color=INK)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([s["step"] for s in steps], fontsize=8.2)
    ax.set_ylabel("오르내림(dB)")
    ax.set_ylim(0, max(vals) + 3)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("⑤ 우리 11-01 삼각에서 하나씩 공개처럼 바꾸면 오르내림이 얼마나 오르나(괄호 = 앞 단계 대비)", loc="left", fontsize=10, color=INK)
    for a in fig.axes:
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.savefig(out, dpi=130)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
