"""실제 녹음에서 편 손(숫자 5)과 말아 쥔 손(숫자 0)의 반사 세기를 견준다(2026-11-03). 결과와 해석은 docs/echo-profile.md §5.

    python -I tools/experiments/echo_hand_shape.py --out <저장소 밖 폴더> [--data data]

데이터 묶음(data/)만으로 돈다: 세션 echo.npz·pairs.csv.gz·landmarks.csv.gz(대응표가 없는 세션은 건너뜀).
- 동작 한 번(2 s)에서 앞 0.5 s(pairs edge)를 뺀, 손을 멈춘 구간의 워치 줄(rows_ok)을 모은다. 세션마다 숫자 5·숫자 0이 4번씩이다.
- 반사 세기: 줄마다 포락선(칸 방향 힐베르트 크기)을 칸 300 세기로 나눠 제곱하고, 구간 안에서 평균한 dB(직접 경로 = 0 dB).
- 두 가지 칸 300으로 잰다. ① profiles.py 그대로(따라간 봉우리 = 그 줄 포락선 최대). ② 첫 도착: second_arrival.py가 칸 300에 둘째
  도착(약 42칸 뒤)을 둔 줄로 찾은 줄을 그만큼 뒤로 밀어 첫 도착을 칸 300에 둔다.
- 세션마다 편 4번·쥔 4번 평균의 차이(손 자리 칸 301–360 RMS)를, 8번을 4·4로 나누는 70가지 모든 나누기(라벨 섞기)와 견준다(정확한 순열
  검정, 가장 작은 p = 2/70 = 0.029). 같은 대조를 손 밖 칸 420–479(43–64 cm)에서도 한다.
- 맞히기: 손 자리 60칸 dB로 가까운 라벨 평균(세션 안 한 번 빼기, 다른 세션: 세션 평균을 뺀 모양을 다른 세션들 평균에 견줌).
- 손끝 거리(단위 손 손바닥 틀, 검지–소지 끝)로 두 손 모양이 실제로 달랐는지 본다.
그림(--out): echo-hand-shape.png. 숫자: echo_hand_shape.json.
"""
import argparse
import itertools
import json
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # tools
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import second_arrival as SA  # noqa: E402

OPEN, CLOSED = "10", "5"  # classId: 숫자 5(편 손), 숫자 0(손가락을 말아 쥔 O)
HAND = (301, 361)
FAR = (420, 480)
REGIONS = {"0–7 cm": (301, 321), "7–14 cm": (321, 341), "14–21 cm": (341, 361)}
BIN_CM = SA.BIN_CM


def instances(data, name, fix):
    """세션의 숫자 5·숫자 0 한 번마다 {"cls", "power"(칸별 평균 세기, 칸 250–479), "open"(손끝 거리), "later"(칸 300 = 둘째 도착인 줄 비율)}."""
    folder = os.path.join(data, "sessions", name)
    E = np.load(os.path.join(folder, "echo.npz"))
    raw, b0, ok = E["raw"].astype(np.float64), int(E["bin0"]), E["rows_ok"].astype(bool)
    sa = SA.second_arrival(raw, b0)
    env = SA.first_arrival_env(sa["env"], sa["shift"]) if fix else sa["env"]
    i = 300 - b0
    rel2 = (env / env[i - 3: i + 4].max(axis=0)[None]) ** 2
    prs = SA.read_pairs(folder)
    opn = SA.read_openness(folder, prs)
    out = []
    for q in sorted({int(p["prompt"]) for p in prs if p["classId"] in (OPEN, CLOSED)}):
        fr = [k for k, p in enumerate(prs) if p["prompt"] == str(q) and p["edge"] == "0" and p["hand"] == "1" and p["direct"] == "1"
              and p["row"] != ""]
        if len(fr) < 10:
            continue
        rows = np.arange(min(int(prs[k]["row"]) for k in fr), max(int(prs[k]["row"]) for k in fr) + 1)
        rows = rows[ok[rows]]
        if len(rows) < 40:
            continue
        out.append({"cls": prs[fr[0]]["classId"], "t": int(prs[fr[0]]["frameNs"]), "rows": int(len(rows)),
                    "power": np.nanmean(rel2[:, rows], axis=1), "open": float(np.nanmean(opn[fr])),
                    "later": float(sa["later"][rows].mean())})
    return b0, out


def stats(b0, insts):
    """세션 하나: 라벨 평균 차이 D(dB), 70가지 4·4 나누기 차이 S, 순열 p(손 자리·손 밖), 같은 모양·다른 모양 한 번씩 차이 RMS, 구역 차이."""
    P = 10 * np.log10(np.array([x["power"] for x in insts]))
    lab = np.array([x["cls"] == OPEN for x in insts])
    n = len(insts)
    hz, fz = slice(HAND[0] - b0, HAND[1] - b0), slice(FAR[0] - b0, FAR[1] - b0)

    def diff(mask):
        return np.nanmean(P[mask], 0) - np.nanmean(P[~mask], 0)

    def rms(d, z):
        return float(np.sqrt(np.nanmean(d[z] ** 2)))
    D = diff(lab)
    S = []
    for comb in itertools.combinations(range(n), int(lab.sum())):
        m = np.zeros(n, bool)
        m[list(comb)] = True
        S.append(diff(m))
    S = np.array(S)
    pairs = list(itertools.combinations(range(n), 2))
    same = np.array([P[i] - P[j] for i, j in pairs if lab[i] == lab[j]])
    other = np.array([P[i] - P[j] if lab[i] else P[j] - P[i] for i, j in pairs if lab[i] != lab[j]])

    def region(mask, lo, hi):
        return 10 * np.log10(np.nanmean(10 ** (np.nanmean(P[mask], 0)[lo - b0: hi - b0] / 10)))
    return {"P": P, "lab": lab, "D": D, "S": S,
            "rmsHand": rms(D, hz), "nullHand": float(np.median([rms(s, hz) for s in S])),
            "pHand": float(np.mean([rms(s, hz) >= rms(D, hz) - 1e-12 for s in S])),
            "pFar": float(np.mean([rms(s, fz) >= rms(D, fz) - 1e-12 for s in S])),
            "sameRms": float(np.sqrt(np.nanmean(same[:, hz] ** 2))), "otherRms": float(np.sqrt(np.nanmean(other[:, hz] ** 2))),
            "regions": {k: float(region(lab, *v) - region(~lab, *v)) for k, v in REGIONS.items()}}


def classify(per):
    """(세션 안 한 번 빼기 맞힘, 다른 세션 맞힘, 개수). 손 자리 60칸 dB, 가까운 라벨 평균."""
    within, across, feats = [], [], []
    for b0, st in per:
        X, y = st["P"][:, HAND[0] - b0: HAND[1] - b0], st["lab"]
        for i in range(len(X)):
            m = np.ones(len(X), bool)
            m[i] = False
            c1, c0 = X[m & y].mean(0), X[m & ~y].mean(0)
            within.append((np.linalg.norm(X[i] - c1) < np.linalg.norm(X[i] - c0)) == y[i])
        feats.append((X - X.mean(0), y))
    for k in range(len(feats)):
        Xtr = np.concatenate([f[0] for j, f in enumerate(feats) if j != k])
        ytr = np.concatenate([f[1] for j, f in enumerate(feats) if j != k])
        c1, c0 = Xtr[ytr].mean(0), Xtr[~ytr].mean(0)
        Xte, yte = feats[k]
        across += list((np.linalg.norm(Xte - c1, axis=1) < np.linalg.norm(Xte - c0, axis=1)) == yte)
    return float(np.mean(within)), float(np.mean(across)), len(within)


def analyse(data, names, fix):
    per, rows = [], []
    for name in names:
        b0, insts = instances(data, name, fix)
        if sum(x["cls"] == OPEN for x in insts) < 2 or sum(x["cls"] == CLOSED for x in insts) < 2:
            continue
        st = stats(b0, insts)
        per.append((b0, st))
        rows.append({"name": name, "st": st, "insts": [(x["t"], x["cls"], x["later"]) for x in insts],
                     "open": float(np.mean([x["open"] for x in insts if x["cls"] == OPEN])),
                     "closed": float(np.mean([x["open"] for x in insts if x["cls"] == CLOSED]))})
    within, across, n = classify(per)
    return per, rows, within, across, n


def report(label, run):
    per, rows, within, across, n = run
    print(f"\n== {label}")
    for r in rows:
        st = r["st"]
        print(f"  {r['name']}: 손끝 거리 편 {r['open']:.2f} 쥔 {r['closed']:.2f} · 손 자리 편−쥔 RMS {st['rmsHand']:.2f} dB(라벨 섞기 중앙 "
              f"{st['nullHand']:.2f}, p={st['pHand']:.3f}) · 손 밖 p={st['pFar']:.3f} · 한 번씩: 같은 모양 {st['sameRms']:.2f} / 다른 모양 "
              f"{st['otherRms']:.2f} dB · " + " ".join(f"{k} {v:+.2f}" for k, v in st["regions"].items()))
    regs = np.array([[r["st"]["regions"][k] for k in REGIONS] for r in rows])
    print(f"  맞히기: 세션 안 {within * 100:.0f}% ({n}번), 다른 세션으로 {across * 100:.0f}% (우연 50%) · 구역 평균 편 − 쥔 "
          + ", ".join(f"{k} {regs[:, i].mean():+.2f} dB(편이 센 세션 {int((regs[:, i] > 0).sum())}/{len(regs)})" for i, k in enumerate(REGIONS))
          + f" · p<0.05 손 자리 {sum(r['st']['pHand'] < 0.05 for r in rows)}/{len(rows)}, 손 밖 {sum(r['st']['pFar'] < 0.05 for r in rows)}")
    return {"within": within, "across": across, "count": n,
            "sessions": {r["name"]: {k: v for k, v in r["st"].items() if k not in ("P", "lab", "D", "S")} | {"openness": [r["open"], r["closed"]]}
                         for r in rows}}


def figure(out_dir, raw_run, fix_run):
    import matplotlib.ticker
    from matplotlib.patches import Patch
    import sonar_signals as SG

    plt = SG.korean_pyplot()
    plt.rcParams["axes.unicode_minus"] = False
    per, rows, within, across, _ = fix_run
    b0 = per[0][0]
    bins = np.arange(b0, b0 + 230)
    x = (bins - 300) * BIN_CM
    sel = (bins >= 292) & (bins <= 372)
    fig, axs = plt.subplots(2, 2, figsize=(14.5, 9.8), gridspec_kw=dict(hspace=0.42, wspace=0.22))
    a, b, c, d = axs.ravel()
    mo = np.array([np.nanmean(st["P"][st["lab"]], 0) for _, st in per])
    mc = np.array([np.nanmean(st["P"][~st["lab"]], 0) for _, st in per])
    for M, col, name in ((mo, "#1f77b4", "숫자 5(편 손)"), (mc, "#d62728", "숫자 0(말아 쥔 손)")):
        for m in M:
            a.plot(x[sel], m[sel], color=col, lw=0.5, alpha=0.35)
        a.plot(x[sel], M.mean(0)[sel], color=col, lw=2.4, label=f"{name}: 세션 {len(M)}개 평균")
    a.axvspan(0, 21.4, color="#f2e6c9", alpha=0.5, zorder=0)
    a.annotate("둘째 도착(14–16 cm): 세션마다\n첫 도착과 거의 같은 세기", (15.0, -3.2), xytext=(17.5, 1.5), fontsize=8.5,
               arrowprops=dict(arrowstyle="->", lw=0.8))
    a.text(10.7, 3.0, "손 자리(모델 입력 0–21 cm)", ha="center", fontsize=9)
    a.set(xlabel="거리 (cm, 0 = 첫 도착)", ylabel="반사 세기 (dB, 첫 도착 = 0)", ylim=(-23, 4.5),
          title="(a) 반사 세기: 편 손과 쥔 손이 거의 겹친다(가는 선 = 세션)")
    a.legend(fontsize=8.5, loc="lower left")
    a.grid(alpha=0.25)
    D = np.array([st["D"] for _, st in per])
    null95 = np.array([np.nanpercentile(np.abs(st["S"]), 95, axis=0) for _, st in per])
    for dd in D:
        b.plot(x[sel], dd[sel], color="#555", lw=0.6, alpha=0.5)
    b.fill_between(x[sel], -null95.mean(0)[sel], null95.mean(0)[sel], color="#bbb", alpha=0.5,
                   label="라벨을 섞어 4·4로 나눠도 나오는 차이(95%, 세션 평균)")
    b.plot(x[sel], np.nanmean(D, 0)[sel], color="black", lw=2.4, label=f"편 − 쥔: 세션 {len(D)}개 평균")
    b.plot([], [], color="#555", lw=0.6, label="세션 하나")
    b.axhline(0, color="black", lw=0.6)
    b.axvspan(0, 21.4, color="#f2e6c9", alpha=0.5, zorder=0)
    yl = max(3.0, float(np.nanmax(np.abs(D[:, sel]))) * 1.05)
    b.set(xlabel="거리 (cm)", ylabel="편 손 − 쥔 손 (dB)", ylim=(-yl, yl), title="(b) 차이(1 dB ≈ 세기 12%): 대부분 라벨을 섞어도 나오는 크기")
    b.legend(fontsize=8, loc="lower left")
    b.grid(alpha=0.25)
    names = [r["name"][4:] for r in rows]
    k = np.arange(len(rows))
    raw_rms = {r["name"]: r["st"]["rmsHand"] for r in raw_run[1]}
    c.bar(k - 0.27, [raw_rms[r["name"]] for r in rows], width=0.27, color="#bdbdbd", label="따라간 봉우리 그대로(지금 profiles.py)")
    c.bar(k, [r["st"]["rmsHand"] for r in rows], width=0.27, color="#e6a03c", label="첫 도착으로 다시 맞춤")
    c.bar(k + 0.27, [r["st"]["nullHand"] for r in rows], width=0.27, color="#9ecae1", label="라벨을 섞은 나누기의 중앙(같은 모양 다시 하기 수준)")
    for i, r in enumerate(rows):
        if r["st"]["pHand"] < 0.05:
            c.text(i, r["st"]["rmsHand"] * 1.12, "*", ha="center", fontsize=13)
    c.set_yscale("log")
    c.set_ylim(0.15, 60)
    c.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    c.set_xticks(k, names, rotation=35, fontsize=8)
    c.set(ylabel="손 자리 60칸 편−쥔 RMS (dB, 로그)", title="(c) 세션별 차이 크기(* = 70가지 라벨 섞기 중 가장 큼, p = 0.03)")
    c.legend(fontsize=7.8, loc="upper left")
    c.grid(axis="y", alpha=0.25)
    show = [r for r in raw_run[1] if max(v[2] for v in r["insts"]) > 0.2]
    for j, r in enumerate(show):
        for q, (t, cls, later) in enumerate(sorted(r["insts"])):
            d.bar(j * 9 + q, later * 100, color="#1f77b4" if cls == OPEN else "#d62728", width=0.85)
        d.text(j * 9 + 3.5, 106, r["name"][4:], ha="center", fontsize=8.5)
    d.legend(handles=[Patch(color="#1f77b4", label="숫자 5(편 손) 한 번"), Patch(color="#d62728", label="숫자 0(쥔 손) 한 번")],
             fontsize=8.5, loc="center right")
    d.set_xticks([])
    d.set(ylim=(0, 115), ylabel="둘째 도착이 칸 300이 된 줄 (%)", xlabel="세션별 동작 순서(시간순)",
          title="(d) 손 모양에 따라 기준점(칸 300)이 둘째 도착으로 약 42칸 뛴다\n(214654·215102는 편 손일 때, 214248은 쥔 손일 때)")
    fig.suptitle(f"실제 녹음(저장소 data/ {len(rows)}세션: 한 창·음성 인식 음원·볼륨 11–13)에서 숫자 5와 숫자 0의 반사 세기, 세션마다 각 4번",
                 fontsize=12.5, y=0.995)
    path = os.path.join(out_dir, "echo-hand-shape.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # nan 칸(밀린 줄의 앞 칸) 평균
    parser = argparse.ArgumentParser(description="실제 녹음에서 편 손과 쥔 손의 반사 세기 견주기")
    parser.add_argument("--out", required=True, help="그림·숫자를 쓸 폴더(저장소 밖)")
    parser.add_argument("--data", default=SA.DEFAULT_DATA, help="데이터 묶음 폴더(기본: 저장소 data/)")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    names = [n for n in sorted(os.listdir(os.path.join(args.data, "sessions")))
             if os.path.exists(os.path.join(args.data, "sessions", n, "pairs.csv.gz"))]
    raw_run = analyse(args.data, names, fix=False)
    fix_run = analyse(args.data, names, fix=True)
    result = {"tracked": report("칸 300 = 따라간 봉우리 그대로(profiles.py)", raw_run),
              "firstArrival": report("칸 300 = 첫 도착으로 다시 맞춤", fix_run)}
    path = figure(args.out, raw_run, fix_run)
    with open(os.path.join(args.out, "echo_hand_shape.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"\n그림 {path}")


if __name__ == "__main__":
    main()
