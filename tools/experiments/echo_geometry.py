"""에코 프로파일의 한 칸이 공간에서 무엇인지 그린다(2026-11-03). 결과와 해석은 docs/echo-profile.md §2·§3.

    python -I tools/experiments/echo_geometry.py --out <저장소 밖 폴더> [--data data] [--session 20261028-215926]

- 같은 칸 = 스피커 → 반사점 → 마이크 경로 길이가 같은 점들 = 두 소리 구멍을 초점으로 한 회전 타원체 껍질. 칸 300 + k의 거리는
  (|PS| + |PM| − 구멍 간격) ÷ 2 = k × 3.57 mm다. 스피커와 마이크를 맞바꿔도 껍질은 같다.
- 워치 자리·구멍 간격은 hand_echo_sim과 같은 가정이다(손목에서 팔뚝 쪽 3.5 cm, 손등 쪽 2 cm, 구멍 간격 4.2 cm, 손 젖힘 25°,
  워치를 손바닥 틀에 붙임). 손은 세션의 MediaPipe 관절에서 손끝이 가장 벌어진 3%(편 손)와 가장 오므린 3%(쥔 손) 근처 줄 하나씩이다.
- (c)는 hand_echo_sim 반사점(손가락 마디·손바닥, 면적 무게)을 거리 칸에 모은 막대다. 퍼짐 감쇠(1 ÷ (스피커 거리 × 마이크 거리))만
  넣고 구멍 방향성·가려짐·간섭은 뺐다.
- (d)는 위에서 본 평면에서 두 번째 마이크(손목 옆면 12시 쪽, 가정)의 껍질이 지금 껍질과 어디서 겹치는지(두께 ±1칸) 그린다.
그림(--out): echo-shells.png.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # tools
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import hand_echo_sim as H  # noqa: E402
import joints  # noqa: E402
import sonar_signals as SG  # noqa: E402

HALF = 0.021  # 구멍 간격의 절반(m)
BIN = 343 / 48000 / 2  # 칸 하나(m)
SHELLS = (10, 20, 30, 40, 50, 60)  # 칸 300에서 센 칸 수
TIPS = (4, 8, 12, 16, 20)
LINES = tuple(joints.BONES) + ((5, 9), (9, 13), (13, 17))


def shell_axes(k):
    """칸 300 + k 껍질의 (팔뚝 축 방향 반지름, 옆 방향 반지름) cm(워치 가운데 원점)."""
    A = (k * BIN + HALF) * 100
    return A, float(np.sqrt(A ** 2 - (HALF * 100) ** 2))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="에코 프로파일 칸의 공간 모양 그리기")
    parser.add_argument("--out", required=True, help="그림을 쓸 폴더(저장소 밖)")
    parser.add_argument("--data", default=H.DEFAULT_DATA, help="데이터 묶음 폴더(기본: 저장소 data/)")
    parser.add_argument("--session", default="20261028-215926")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import matplotlib.ticker
    from matplotlib.patches import FancyBboxPatch

    plt = SG.korean_pyplot()
    plt.rcParams["axes.unicode_minus"] = False
    sim = H.Sim(args.data, args.session)
    a, n, y, Wc = sim.a, sim.n_a, sim.y_a, sim.Wc
    S, M = Wc + HALF * a, Wc - HALF * a
    Q = sim.Q

    def rng(P):
        return (np.linalg.norm(P - S, axis=-1) + np.linalg.norm(P - M, axis=-1) - 2 * HALF) / 2

    def proj(P, e2):
        d = P - Wc
        return d @ a * 100, d @ e2 * 100

    spread = np.linalg.norm(Q[:, list(TIPS)], axis=-1).mean(1)
    idx = np.flatnonzero(np.isfinite(spread))
    i_open = idx[np.argmin(np.abs(spread[idx] - np.percentile(spread[idx], 97)))]
    i_fist = idx[np.argmin(np.abs(spread[idx] - np.percentile(spread[idx], 3)))]
    poses = {"편 손": [Q[i_open], "#1f77b4"], "주먹": [Q[i_fist], "#d62728"]}
    print("1. 껍질 모양(워치 가운데 원점): 칸 · 거리 → 팔뚝 축 방향 반지름 / 옆 방향 반지름")
    for k in SHELLS:
        A, B = shell_axes(k)
        print(f"  칸 {300 + k}: {k * BIN * 100:4.1f} cm → {A:4.1f} / {B:4.1f} cm")
    print("2. 실제 관절의 거리(칸 300에서, cm)")
    for name, (P, _) in poses.items():
        r = rng(P) * 100
        print(f"  {name}: 관절 {r[1:].min():.1f}–{r[1:].max():.1f} cm, 손끝(엄지·검지·중지·약지·소지) {np.round(r[list(TIPS)], 1)}")

    fig = plt.figure(figsize=(15, 10.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.42, wspace=0.16)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axC, axD = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    t = np.linspace(0, 2 * np.pi, 400)

    def shells(ax, labels=True):
        for k in SHELLS:
            A, B = shell_axes(k)
            ax.plot(A * np.cos(t), B * np.sin(t), color="#888", lw=0.9 if k % 20 else 1.4, zorder=1)
            if labels:
                ang = np.deg2rad(62)
                ax.text(A * np.cos(ang), B * np.sin(ang), f"칸 {300 + k}\n{k * BIN * 100:.1f} cm", fontsize=7.5, color="#555",
                        ha="center", va="center", bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.5), zorder=2)

    def body(ax, top):
        if top:
            ax.add_patch(FancyBboxPatch((-18, -2.8), 21.5, 5.6, boxstyle="round,pad=0,rounding_size=1.5", fc="#f3e6d8", ec="#c9a98a", zorder=0))
            ax.add_patch(FancyBboxPatch((-2.2, -2.0), 4.4, 4.0, boxstyle="round,pad=0,rounding_size=1.2", fc="#444", ec="black", zorder=3))
        else:
            ax.add_patch(FancyBboxPatch((-18, -4.2), 21.5, 3.7, boxstyle="round,pad=0,rounding_size=1.2", fc="#f3e6d8", ec="#c9a98a", zorder=0))
            ax.add_patch(FancyBboxPatch((-2.2, -0.5), 4.4, 1.0, boxstyle="round,pad=0,rounding_size=0.3", fc="#444", ec="black", zorder=3))

    def hand(ax, e2):
        for name, (P, c) in poses.items():
            u, v = proj(P, e2)
            for i, j in LINES:
                ax.plot([u[i], u[j]], [v[i], v[j]], color=c, lw=2.2, solid_capstyle="round", zorder=5)
            ax.plot(u, v, "o", ms=3.2, color=c, zorder=6)
        ax.plot([], [], color=poses["편 손"][1], lw=2.2, label="편 손(실제 세션 관절)")
        ax.plot([], [], color=poses["주먹"][1], lw=2.2, label="주먹")

    # (a) 옆에서
    shells(axA)
    body(axA, top=False)
    axA.plot([-2.1, 2.1], [0, 0], "o", ms=6, mfc="#ffcc00", mec="black", zorder=4)
    hand(axA, n)
    P_ref = poses["편 손"][0][6]  # 검지 가운데 마디(PIP)
    k_ref = rng(P_ref) / BIN
    A_ref = (rng(P_ref) + HALF) * 100
    B_ref = np.sqrt(A_ref ** 2 - (HALF * 100) ** 2)
    axA.plot(A_ref * np.cos(t), B_ref * np.sin(t), color="#2ca02c", lw=2.0, zorder=4)
    ur, vr = proj(P_ref, n)
    for (px, py, txt, ha, off) in ((ur, vr, "검지 가운데 마디", "left", (30, 60)), (0, B_ref, "손등 위 같은 거리(소매·다른 손)", "center", (0, 9)),
                                   (-A_ref, 0.0, "팔뚝 쪽 같은 거리", "left", (8, 9))):
        axA.plot(px, py, "*", ms=15, mfc="#2ca02c", mec="black", zorder=7)
        axA.annotate(txt, (px, py), xytext=off, textcoords="offset points", fontsize=8.5, color="#1b6e1b", ha=ha, zorder=8,
                     bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.6),
                     arrowprops=dict(arrowstyle="-", color="#1b6e1b", lw=0.8) if off[1] > 20 else None)
    axA.text(-17.5, 21.5, f"초록 껍질(칸 {300 + k_ref:.0f}) 위 세 별은\n마이크 하나에 똑같이 들린다", fontsize=9, color="#1b6e1b", va="top")
    axA.set(xlim=(-18, 26), ylim=(-12, 23), aspect="equal", xlabel="팔뚝 축 (cm, + 손 쪽)", ylabel="손등 쪽 (cm)",
            title="(a) 옆에서 본 거리 껍질: 같은 칸 = 두 구멍을 초점으로 한 타원")
    axA.legend(loc="lower right", fontsize=8)
    axA.text(0, -1.3, "워치", ha="center", va="top", fontsize=8, color="#222")
    # (b) 위에서
    shells(axB, labels=False)
    body(axB, top=True)
    axB.plot([-2.1, 2.1], [0, 0], "o", ms=6, mfc="#ffcc00", mec="black", zorder=4)
    hand(axB, y)
    axB.text(-1.9, 2.5, "구멍 둘\n(스피커·마이크,\n맞바꿔도 같음)", fontsize=7.5, ha="right", va="bottom")
    axB.set(xlim=(-18, 26), ylim=(-16, 16), aspect="equal", xlabel="팔뚝 축 (cm, + 손 쪽)", ylabel="새끼 쪽 (cm, 12시)",
            title="(b) 위에서: 손 거리(3–20 cm)에서는 거의 워치 가운데 중심의 구")
    # (c) 3D 손 → 1D 거리 막대
    edges = np.arange(0, 61) - 0.5
    hists = {}
    print("3. 반사점 세기(면적 ÷ 퍼짐)의 거리 막대")
    for name, (P, c) in poses.items():
        pts, wts, _ = sim.points(P)
        amp = wts / (np.linalg.norm(pts - S, axis=1) * np.linalg.norm(pts - M, axis=1))
        hists[name], _ = np.histogram(rng(pts) / BIN, bins=edges, weights=amp)
        h = hists[name]
        print(f"  {name}: 가장 센 칸 {300 + int(np.argmax(h))}, 칸 338 너머(13.6 cm 넘음) 몫 {h[38:].sum() / h.sum() * 100:.1f}%, "
              f"칸 340–356 평균 ÷ 가장 센 칸 {h[40:57].mean() / h.max():.4f}")
    top = max(h.max() for h in hists.values())
    for off, (name, (_, c)) in zip((-0.21, 0.21), poses.items()):
        axC.bar(np.arange(60) + off, hists[name] / top, width=0.42, color=c, alpha=0.85, label=name)
    axC.set_yscale("log")
    axC.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axC.plot([4, 20], [1.6, 1.6], color="black", lw=2)
    axC.text(21, 1.6, "거리 분해능 5.7 cm(16칸): 이 폭 안의 반사는 한 덩어리로 번진다", ha="left", va="center", fontsize=8.5)
    axC.annotate("편 손 손끝(14–20 cm): 이 단순 모형에서\n손바닥 아래(칸 310 둘레)보다 100배 넘게 약하다.\n주먹은 칸 341 너머가 비어 있다",
                 (50, 0.004), xytext=(34, 0.06), fontsize=8.5, arrowprops=dict(arrowstyle="->", lw=0.8))
    sec = axC.secondary_xaxis("top", functions=(lambda b: b * BIN * 100, lambda c: c / (BIN * 100)))
    sec.set_xlabel("거리 (cm)")
    axC.set(xlim=(-1, 60), ylim=(1e-3, 3), xlabel="칸 − 300 (모델 입력 60칸)", ylabel="반사 세기 몫(상대, 로그)")
    axC.set_title("(c) 3D 손이 1D 거리 막대로 접힌다(퍼짐 감쇠만, 방향성·간섭 뺌)", pad=26)
    axC.legend(loc="upper right", fontsize=8.5, bbox_to_anchor=(1.0, 0.86))
    # (d) 듣는 자리가 둘이면(스피커 = 공통 초점, 팔꿈치 쪽 구멍으로 가정)
    T = np.array(proj(poses["편 손"][0][8], y))  # 검지 끝
    SP, M1, M2 = np.array([-2.1, 0.0]), np.array([2.1, 0.0]), np.array([0.0, 2.0])
    body(axD, top=True)
    axD.plot(*SP, "o", ms=9, mfc="#ffcc00", mec="black", zorder=5)
    axD.plot(*M1, "o", ms=9, mfc="#2ca02c", mec="black", zorder=5)
    axD.plot(*M2, "s", ms=8, mfc="#9467bd", mec="black", zorder=5)
    for xy, txt, off in ((SP, "스피커\n(팔꿈치 쪽 가정)", (-14, -30)), (M1, "마이크 1", (16, -30)), (M2, "마이크 2(가정)", (0, 10))):
        axD.annotate(txt, xy, xytext=off, textcoords="offset points", ha="center", va="center", fontsize=7.5, zorder=6,
                     bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.4))
    xx, yy = np.meshgrid(np.linspace(-18, 26, 1400), np.linspace(-16, 16, 1100))

    def dist(F):
        return np.hypot(xx - F[0], yy - F[1])
    d1, d2 = dist(SP) + dist(M1), dist(SP) + dist(M2)
    L1 = np.linalg.norm(T - SP) + np.linalg.norm(T - M1)
    L2 = np.linalg.norm(T - SP) + np.linalg.norm(T - M2)
    tol = 2 * BIN * 100  # 경로 ±7.1 mm = 거리 ±1칸
    b1, b2 = np.abs(d1 - L1) < tol, np.abs(d2 - L2) < tol
    both = b1 & b2
    axD.contourf(xx, yy, b1.astype(float), levels=[0.5, 1.5], colors=["#2ca02c"], alpha=0.35)
    axD.contourf(xx, yy, b2.astype(float), levels=[0.5, 1.5], colors=["#9467bd"], alpha=0.35)
    axD.contourf(xx, yy, both.astype(float), levels=[0.5, 1.5], colors=["black"], alpha=0.85)
    axD.plot(*T, "*", ms=15, mfc="gold", mec="black", zorder=7)
    ang = np.degrees(np.arctan2(yy[both], xx[both]))
    span = float(np.ptp(ang[np.abs(ang - np.degrees(np.arctan2(T[1], T[0]))) < 60]))
    arc = np.radians(span) * np.linalg.norm(T)
    g1 = (T - SP) / np.linalg.norm(T - SP) + (T - M1) / np.linalg.norm(T - M1)
    g2 = (T - SP) / np.linalg.norm(T - SP) + (T - M2) / np.linalg.norm(T - M2)
    cross_deg = float(np.degrees(np.arccos(g1 @ g2 / np.linalg.norm(g1) / np.linalg.norm(g2))))
    print(f"4. 두 번째 마이크(손목 옆면 12시 쪽, 가정): 검지 끝 {np.linalg.norm(T):.1f} cm에서 두 껍질이 {cross_deg:.1f}°로 만나고, "
          f"±1칸 두께면 겹친 곳이 {span:.0f}°(약 {arc:.0f} cm)")
    axD.annotate(f"검지 끝 ★ 둘레에서 두 껍질이 {cross_deg:.0f}°로 비스듬히 만나\n±1칸 두께면 겹친 곳(검정)이 {span:.0f}°, 약 {arc:.0f} cm 길이\n"
                 "→ 워치 크기 안의 두 마이크는 방향을 거칠게만 준다",
                 tuple(T), xytext=(-250, 95), textcoords="offset points", fontsize=8.5,
                 bbox=dict(fc="white", ec="#999", alpha=0.9), arrowprops=dict(arrowstyle="->", lw=0.8))
    axD.plot([], [], color="#2ca02c", lw=6, alpha=0.5, label="스피커–마이크 1(지금): 같은 칸 띠")
    axD.plot([], [], color="#9467bd", lw=6, alpha=0.5, label="스피커–마이크 2(가정): 같은 칸 띠")
    axD.legend(loc="lower left", fontsize=8.2)
    axD.set(xlim=(-18, 26), ylim=(-16, 16), aspect="equal", xlabel="팔뚝 축 (cm, + 손 쪽)", ylabel="새끼 쪽 (cm)",
            title="(d) 방향을 얻으려면 듣는 자리가 둘 이상: 두 껍질이 겹치는 곳")
    for ax in (axA, axB, axD):
        ax.grid(alpha=0.25)
    fig.suptitle(f"에코 프로파일의 한 칸 = 공간의 껍질 하나(워치 가운데 원점, 세션 {args.session} 관절, 워치 자리는 시뮬레이션 가정)",
                 fontsize=12.5, y=0.985)
    path = os.path.join(args.out, "echo-shells.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"그림 {path}")


if __name__ == "__main__":
    main()
