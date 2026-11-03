"""직접 경로 뒤 약 42칸(14–16 cm)의 둘째 도착('둔덕')을 세션·시험 녹음에서 찾고, 손 자세와의 관계와 칸 300 기준점이 뛰는 줄을 본다
(2026-11-03). 결과와 해석은 docs/echo-profile.md §1·§6.

    python -I tools/experiments/second_arrival.py --out <저장소 밖 폴더> [--data data] [--no-sim]

데이터 묶음(data/, export_data.py)만으로 돈다: 세션 echo.npz(정렬된 칸 250–479)·pairs.csv.gz·landmarks.csv.gz, 시험 녹음 echo.npz·rec.json.
1. 줄마다 정렬 좌표(칸 300 = profiles.py가 따라간 봉우리 = 그 줄 포락선 최대)의 포락선(칸 방향 힐베르트 크기, dB)에서 칸 300 말고
   진짜 국소 최대(±3칸 안 최대, 칸 300 쪽 골짜기보다 1.5 dB 넘게 솟음)를 찾는다.
   - 앞(칸 252–290)에 칸 300보다 25칸 넘게 앞서고 −10 dB 넘는 봉우리가 있으면 그 줄은 둘째 도착을 칸 300에 둔 줄이다(첫 도착 = 앞 봉우리).
   - 아니면 뒤(칸 312–368)의 봉우리를 둘째 도착으로 본다. 간격 = 둘째 − 첫(칸, 포물선 보간), 세기 = 둘째 − 첫(dB).
2. 세션·시험 녹음마다: 둘째 도착이 −6 dB 넘게 선 줄 비율, 간격·세기 중앙값, 둘째 도착을 칸 300에 둔 줄 비율, 손끝 거리(단위 손)와
   세기의 상관(손가락·숫자 동작 프레임). 시험 녹음은 칸 300 뒤 20·42·60칸의 포락선 중앙값도 낸다.
3. 자세(숫자 5·숫자 0·손목 젖히기·굽히기·새끼 쪽 꺾기, 동작 앞 0.5 s 뺌)마다 둔덕 자리·세기 중앙값.
4. --no-sim이 아니면 hand_echo_sim(워치를 손바닥 틀에 붙인 모형, 손 메아리 비례 상수는 창 없음 시험 녹음에 맞춘 값)으로
   - 손가락 메아리였다면 둔덕이 얼마나 움직였어야 하나: 손가락 반사점의 세기 가중 거리, 숫자 5 → 숫자 0, 숫자 5 손을 팔뚝은 두고 손목에서
     젖히기·굽히기 30°, 새끼 쪽·엄지 쪽 25° 돌렸을 때.
   - 손 메아리만의 세기(한 창 송신, 숫자 5·숫자 0 줄 평균 포락선, 직접 경로 = 0 dB).
그림(--out): echo-row-hump.png((a) 줄 하나와 반사 세기, (b) 자세별 둔덕 자리, (c) 자세별 둔덕 세기). 숫자: second_arrival.json.
"""
import argparse
import csv
import gzip
import json
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # tools
sys.path.insert(0, HERE)  # experiments
import hs_io  # noqa: E402,F401  사용자 설치 numpy·scipy 경로도 붙는다
import joints  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import hilbert  # noqa: E402

DEFAULT_DATA = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data")
BIN_CM = 343 / 48000 / 2 * 100
EARLY = (252, 291)  # 앞 봉우리 찾는 칸(칸 300을 둘째 도착에 둔 줄의 첫 도착)
LATE = (312, 369)  # 뒤 봉우리 찾는 칸(둘째 도착)
MIN_GAP = 25  # 이보다 가까운 봉우리는 직접 경로 주봉의 어깨로 보고 따로 세지 않는다(한 창 반높이 폭 약 42칸)
EARLY_MIN_DB = -10.0  # 앞 봉우리가 이보다 세야 첫 도착으로 본다
PROMINENCE_DB = 1.5
STRONG_DB = -6.0  # '둘째 도착이 선 줄'의 문턱
POSES = (("10", "숫자 5(편 손)"), ("5", "숫자 0(쥔 손)"), ("15", "손목 젖히기"), ("16", "손목 굽히기"), ("17", "새끼 쪽 꺾기"))
FINGER_CLASSES = {str(c) for c in range(15)}  # 손가락 굽히기·숫자(손목은 그대로)
ROW_EXAMPLE = "20261028-201138"  # 그림 (a)의 세션(기준점이 뛰지 않은 세션)


def envelope(raw):
    """정렬된 부호 있는 상관 (칸, 줄) → 칸 방향 힐베르트 크기(포락선)."""
    return np.abs(hilbert(raw, axis=0))


def level_db(env, b0):
    """포락선을 칸 296–304 최대(따라간 봉우리) = 0 dB로 둔 dB(0인 칸은 −inf)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return 20 * np.log10(env / env[296 - b0: 305 - b0].max(0))


def _local_max(L):
    lm = np.ones_like(L, bool)
    for s in (1, 2, 3):
        lm[s:] &= L[s:] >= L[:-s]
        lm[:-s] &= L[:-s] >= L[s:]
    return lm


def _pick(L, lm, b0, lo, hi):
    """칸 lo…hi−1의 진짜 국소 최대 중 가장 센 것: (세기 dB, 칸(포물선 보간)). 없으면 nan."""
    i0, i1 = lo - b0, hi - b0
    c = 300 - b0
    cols = np.arange(L.shape[1])
    cand = np.where(lm[i0:i1], L[i0:i1], -np.inf)
    k = cand.argmax(0) + i0
    lvl = cand.max(0)
    valley = np.array([L[min(p, c): max(p, c) + 1, r].min() if np.isfinite(lvl[r]) else np.nan for r, p in enumerate(k)])
    ok = np.isfinite(lvl) & (lvl - valley >= PROMINENCE_DB)
    km = np.clip(k, 1, L.shape[0] - 2)
    y0, y1, y2 = L[km - 1, cols], L[km, cols], L[km + 1, cols]
    den = y0 - 2 * y1 + y2
    frac = np.where(np.abs(den) > 1e-9, 0.5 * (y0 - y2) / np.where(np.abs(den) > 1e-9, den, 1.0), 0.0)
    return np.where(ok, lvl, np.nan), np.where(ok, km + np.clip(frac, -0.5, 0.5) + b0, np.nan)


def second_arrival(raw, b0):
    """줄마다 {"gap": 둘째 − 첫(칸), "rel": 둘째 − 첫(dB), "later": 칸 300이 둘째 도착인 줄, "shift": 첫 도착을 칸 300에 두려면 밀 칸,
    "env": 포락선}. 둘째 도착이 없으면 gap·rel은 nan."""
    env = envelope(raw)
    L = level_db(env, b0)
    lm = _local_max(L)
    le, pe = _pick(L, lm, b0, *EARLY)
    ll, pl = _pick(L, lm, b0, *LATE)
    later = np.isfinite(le) & (le > EARLY_MIN_DB) & (300 - pe >= MIN_GAP)
    gap = np.where(later, 300 - pe, pl - 300)
    rel = np.where(later, -le, ll)
    shift = np.where(later, np.rint(300 - np.nan_to_num(pe, nan=300.0)), 0).astype(int)
    return {"gap": gap, "rel": rel, "later": later, "shift": shift, "env": env}


def first_arrival_env(env, shift):
    """칸 300에 둘째 도착을 둔 줄을 shift칸 뒤로 밀어 첫 도착을 칸 300에 둔 포락선(밀려 비는 앞 칸은 nan)."""
    out = env.copy()
    for r in np.flatnonzero(shift > 0):
        g = int(shift[r])
        out[g:, r] = env[:-g, r]
        out[:g, r] = np.nan
    return out


def read_pairs(folder):
    with gzip.open(os.path.join(folder, "pairs.csv.gz"), "rt", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_openness(folder, prs):
    """프레임마다 검지–소지 끝까지 거리(단위 손 손바닥 틀, 거른 프레임만, 나머지 nan)."""
    with gzip.open(os.path.join(folder, "landmarks.csv.gz"), "rt", encoding="utf-8") as f:
        lrows = list(csv.reader(f))[1:]
    if len(lrows) != len(prs):
        raise ValueError(f"{folder}: landmarks {len(lrows)}줄과 pairs {len(prs)}줄이 다르다")
    world = np.array([[float(v) for v in r[67:130]] if r[2] else [np.nan] * 63 for r in lrows]).reshape(-1, 21, 3)
    score = np.array([float(r[3]) if r[2] else np.nan for r in lrows])
    valid = np.isfinite(world).all(axis=(1, 2))
    good, _ = joints.keep_frames(world, score, valid)
    out = np.full(len(prs), np.nan)
    idx = np.flatnonzero(good)
    if len(idx):
        unit, _, _ = joints.targets(world[idx], "left")
        out[idx] = np.linalg.norm(unit[:, [7, 11, 15, 19]], axis=2).mean(1)
    return out


def pose_rows(prs, ok, cid):
    """동작 cid의 프레임(앞 0.5 s·손 없음·직접 경로 없음 뺌)에 짝지은 줄(rows_ok만)."""
    rw = np.array([int(p["row"]) for p in prs if p["classId"] == cid and p["edge"] == "0" and p["hand"] == "1"
                   and p["direct"] == "1" and p["row"] != ""], dtype=int)
    return rw[ok[rw]] if len(rw) else rw


def session_summary(data, name):
    folder = os.path.join(data, "sessions", name)
    E = np.load(os.path.join(folder, "echo.npz"))
    raw, b0, ok = E["raw"].astype(np.float64), int(E["bin0"]), E["rows_ok"].astype(bool)
    sa = second_arrival(raw, b0)
    strong = (sa["rel"] > STRONG_DB) & (sa["gap"] >= MIN_GAP) & ok
    out = {"rows": int(ok.sum()), "strongPct": float(strong.sum() / ok.sum() * 100),
           "gapMedian": float(np.nanmedian(sa["gap"][strong])) if strong.any() else None,
           "relMedian": float(np.nanmedian(sa["rel"][strong])) if strong.any() else None,
           "laterPct": float((sa["later"] & ok).sum() / ok.sum() * 100)}
    if not os.path.exists(os.path.join(folder, "pairs.csv.gz")):
        out["pairs"] = False
        return out, sa, None
    prs = read_pairs(folder)
    out["pairs"] = True
    opn = read_openness(folder, prs)
    use = [k for k, p in enumerate(prs) if p["classId"] in FINGER_CLASSES and p["edge"] == "0" and p["hand"] == "1"
           and p["direct"] == "1" and p["row"] != "" and np.isfinite(opn[k])]
    rows = np.array([int(prs[k]["row"]) for k in use], dtype=int)
    m = ok[rows] & np.isfinite(sa["rel"][rows]) & (sa["gap"][rows] >= MIN_GAP)
    out["opennessCorr"] = float(np.corrcoef(opn[use][m], sa["rel"][rows][m])[0, 1]) if m.sum() > 50 else None
    poses = {}
    for cid, _ in POSES:
        rw = pose_rows(prs, ok, cid)
        g, r_ = sa["gap"][rw], sa["rel"][rw]
        mm = np.isfinite(g) & (g >= MIN_GAP)
        poses[cid] = {"gap": float(np.median(g[mm])) if mm.mean() > 0.3 else None,
                      "rel": float(np.median(r_[mm])) if mm.mean() > 0.3 else None,
                      "laterPct": float(sa["later"][rw].mean() * 100) if len(rw) else None, "rows": int(len(rw))}
    out["poses"] = poses
    return out, sa, prs


def recording_summary(data, name):
    folder = os.path.join(data, "recordings", name)
    E = np.load(os.path.join(folder, "echo.npz"))
    rec = hs_io.read_json(os.path.join(folder, "rec.json"))
    raw, b0, ok = E["raw"].astype(np.float64), int(E["bin0"]), E["rows_ok"].astype(bool)
    sa = second_arrival(raw, b0)
    strong = (sa["rel"] > STRONG_DB) & (sa["gap"] >= MIN_GAP) & ok
    L = level_db(sa["env"][:, ok], b0)
    med = np.median(L, axis=1)
    i = 300 - b0
    tx, vol = rec.get("tx") or {}, rec.get("volume") or {}
    return {"window": tx.get("window"), "volume": vol.get("index") if isinstance(vol, dict) else vol,
            "source": (rec.get("audio") or {}).get("source") if isinstance(rec.get("audio"), dict) else None,
            "strongPct": float(strong.sum() / ok.sum() * 100),
            "gapMedian": float(np.nanmedian(sa["gap"][strong])) if strong.any() else None,
            "relMedian": float(np.nanmedian(sa["rel"][strong])) if strong.any() else None,
            "envAt": {f"+{k}": float(med[i + k]) for k in (20, 42, 60)}}


def simulate(data, session):
    """hand_echo_sim으로 손가락 메아리였다면의 둔덕 이동(칸)과 손 메아리만의 세기(dB)."""
    import hand_echo_sim as H
    import sonar_signals as SG

    sim = H.Sim(data, session)
    S, M = sim.Wc + 0.021 * sim.a, sim.Wc - 0.021 * sim.a
    prs = read_pairs(os.path.join(data, "sessions", session))
    rows = {cid: np.array(sorted({int(p["row"]) for p in prs if p["classId"] == cid and p["edge"] == "0" and p["hand"] == "1"
                                  and p["row"] != ""}), dtype=int) for cid in ("10", "5")}

    def rot(axis, deg):
        axis = axis / np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        th = np.radians(deg)
        return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K

    def finger_range(rws, R=np.eye(3)):
        vals = []
        for r in rws[::4]:
            P, w, g = sim.points(sim.Q[r] @ R.T)
            d1, d2 = np.linalg.norm(P - S, axis=1), np.linalg.norm(P - M, axis=1)
            rng = (d1 + d2 - 0.042) / 2
            amp = w / (d1 * d2)
            f = g < 5
            vals.append((rng[f] * amp[f]).sum() / amp[f].sum())
        return float(np.median(vals) * 100 / BIN_CM)

    base = finger_range(rows["10"])
    ext_axis = np.cross(sim.a, sim.n_a)  # +각이면 손이 손등 쪽(워치 쪽)으로 젖혀진다
    expect = {"10": 0.0, "5": finger_range(rows["5"]) - base, "15": finger_range(rows["10"], rot(ext_axis, 30)) - base,
              "16": finger_range(rows["10"], rot(ext_axis, -30)) - base, "17": finger_range(rows["10"], rot(sim.n_a, 25)) - base,
              "thumbSide25": finger_range(rows["10"], rot(sim.n_a, -25)) - base}
    Hh0, _ = sim.channel("along", 0.0)
    x = SG.repo("hann", "saw")

    def env_mean(Hm, rws):
        med, _ = sim.process(x, x, SG.BAND, Hm, noisy=False)
        return envelope(med[rws].T).mean(1)

    level = {}
    direct = env_mean(np.ones_like(Hh0), rows["10"])
    for cid in ("10", "5"):
        eh = env_mean(H.SH * Hh0, rows[cid])
        hz = slice(301, 361)
        k = int(np.argmax(eh[hz])) + 301
        level[cid] = {"peakBin": k, "peakDb": float(20 * np.log10(eh[k] / direct[300])), "bin342Db": float(20 * np.log10(eh[342] / direct[300]))}
    level["directTail342Db"] = float(20 * np.log10(direct[342] / direct[300]))
    return {"session": session, "fingerRangeBins": base, "expectShift": expect, "handOnly": level}


def figure(out_dir, data, sessions, sim_res):
    import sonar_signals as SG

    plt = SG.korean_pyplot()
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(15, 9.6))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.18)
    ax_a, ax_b, ax_c = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    # (a) 줄 하나와 반사 세기
    folder = os.path.join(data, "sessions", ROW_EXAMPLE)
    E = np.load(os.path.join(folder, "echo.npz"))
    raw, b0, ok = E["raw"].astype(np.float64), int(E["bin0"]), E["rows_ok"].astype(bool)
    prs = read_pairs(folder)
    q = min(int(p["prompt"]) for p in prs if p["classId"] == "10")
    fr = [int(p["row"]) for p in prs if p["prompt"] == str(q) and p["edge"] == "0" and p["row"] != ""]
    rows = np.arange(min(fr), max(fr) + 1)
    rows = rows[ok[rows]]
    env = envelope(raw[:, rows])
    ref = env[296 - b0: 305 - b0].max(0)
    j = len(rows) // 2
    bins = np.arange(b0, b0 + raw.shape[0])
    x = (bins - 300) * BIN_CM
    sel = (bins >= 290) & (bins <= 372)
    avg = np.sqrt(np.mean((env / ref[None]) ** 2, axis=1))
    ax_a.plot(x[sel], (raw[:, rows[j]] / ref[j])[sel], color="#888", lw=0.9,
              label="줄 하나(처프 한 번, 12.5 ms)의 부호 있는 상관 값: 반송파(19.5 kHz) 때문에 칸마다 +/−로 떤다")
    ax_a.plot(x[sel], (env[:, j] / ref[j])[sel], color="#ff7f0e", lw=1.6, ls="--", zorder=5,
              label="그 줄의 포락선(떨림을 걷은 크기): 평균과 거의 같다(대부분 늘 있는 소리)")
    ax_a.plot(x[sel], avg[sel], color="#1f77b4", lw=2.6, label=f"반사 세기 = 포락선을 손을 멈춘 {len(rows)}줄(약 {len(rows) * 12.5 / 1000:.1f} s) 동안 평균")
    ax_a.axvspan(0, 21.4, color="#f2e6c9", alpha=0.45, zorder=0)
    ax_a.axhline(0, color="black", lw=0.5)
    ax_a.annotate("둔덕(둘째 도착)", (15.2, avg[342 - b0]), xytext=(18.5, 0.95), fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax_a.set(xlim=(-3.6, 25.8), ylim=(-1.15, 1.25), xlabel="거리 (cm, 0 = 첫 도착 = 칸 300)", ylabel="첫 도착 = 1",
             title=f"(a) 에코 프로파일 '줄' 하나와 '반사 세기'(세션 {ROW_EXAMPLE}, 숫자 5 한 번). 모델 입력 채널 0은 회색 같은 줄 96개, 채널 1은 이웃 줄 크기 차이")
    ax_a.legend(fontsize=8.6, loc="lower right")
    ax_a.grid(alpha=0.25)
    # (b)(c) 자세별
    xs = np.arange(len(POSES))
    G, Rl = [], []
    for name, s in sessions.items():
        if not s.get("pairs") or s["poses"]["10"]["gap"] is None:
            continue
        g = np.array([np.nan if s["poses"][c]["gap"] is None else s["poses"][c]["gap"] for c, _ in POSES])
        r_ = np.array([np.nan if s["poses"][c]["rel"] is None else s["poses"][c]["rel"] for c, _ in POSES])
        flip = max(v["laterPct"] or 0 for v in s["poses"].values()) > 20
        ax_b.plot(xs, g - g[0], ls="--" if flip else "-", marker="o", ms=4, lw=1, color="#555", alpha=0.8)
        ax_c.plot(xs, r_ - r_[0], ls="--" if flip else "-", marker="o", ms=4, lw=1, color="#555", alpha=0.8)
        G.append(g - g[0])
        Rl.append(r_ - r_[0])
    G, Rl = np.array(G), np.array(Rl)
    if sim_res:
        ax_b.plot(xs, [sim_res["expectShift"][c] for c, _ in POSES], color="#d62728", lw=2.4, ls=":", marker="s",
                  label="손가락 메아리였다면(시뮬레이션: 손목 30°·25° 꺾음)")
    ax_b.plot(xs, np.nanmedian(G, 0), color="black", lw=2.6, marker="o", label=f"실제 둔덕: 세션 {len(G)}개 중앙값")
    ax_b.plot([], [], color="#555", lw=1, label="세션 하나(점선 = 기준점이 뛴 세션)")
    ax_b.axhline(0, color="black", lw=0.5)
    labels = [p.replace("(", "\n(") if "숫자" in p else p.replace("새끼 쪽 ", "새끼 쪽\n") for _, p in POSES]
    ax_b.set_xticks(xs, labels, fontsize=8.5)
    ax_b.set(ylabel="둔덕 자리 변화 (칸, 숫자 5 기준, 1칸 = 3.6 mm)", ylim=(-9, 4), title="(b) 둔덕 자리: 손 모양·손목을 바꿔도 거의 그대로")
    ax_b.legend(fontsize=8, loc="lower left")
    ax_b.grid(alpha=0.25)
    ax_c.plot(xs, np.nanmedian(Rl, 0), color="black", lw=2.6, marker="o", label="세션 중앙값")
    ax_c.plot([], [], color="#555", lw=1, label="세션 하나")
    ax_c.axhline(0, color="black", lw=0.5)
    ax_c.set_xticks(xs, labels, fontsize=8.5)
    ax_c.set(ylabel="둔덕 세기 변화 (dB, 숫자 5 기준)", ylim=(-4, 2), title="(c) 둔덕 세기: 손을 펴서 똑바로 둘 때 가장 세다(1 dB ≈ 12%)")
    ax_c.legend(fontsize=8.5, loc="lower left")
    ax_c.grid(alpha=0.25)
    fig.suptitle("에코 프로파일의 '줄'과 '반사 세기', 그리고 15 cm 둔덕은 손이 만드는가(저장소 data/ 한 창 세션)", fontsize=12.5, y=0.995)
    path = os.path.join(out_dir, "echo-row-hump.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # nan 칸(밀린 줄의 앞 칸) 평균
    parser = argparse.ArgumentParser(description="둘째 도착(15 cm 둔덕) 찾기와 손 자세와의 관계")
    parser.add_argument("--out", required=True, help="그림·숫자를 쓸 폴더(저장소 밖)")
    parser.add_argument("--data", default=DEFAULT_DATA, help="데이터 묶음 폴더(기본: 저장소 data/)")
    parser.add_argument("--no-sim", action="store_true", help="hand_echo_sim 계산(약 1분)을 건너뛴다")
    parser.add_argument("--sim-session", default="20261028-215926")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    result = {"sessions": {}, "recordings": {}}
    print("1. 세션: −6 dB 넘는 둘째 도착이 선 줄 · 간격(칸, cm) · 세기(둘째 − 첫) · 칸 300을 둘째 도착에 둔 줄 · 손끝 거리–세기 상관")
    for name in sorted(os.listdir(os.path.join(args.data, "sessions"))):
        s, _, _ = session_summary(args.data, name)
        result["sessions"][name] = s
        gap = "—" if s["gapMedian"] is None else f"{s['gapMedian']:.0f}칸({s['gapMedian'] * BIN_CM:.1f} cm)"
        rel = "—" if s["relMedian"] is None else f"{s['relMedian']:+.1f} dB"
        corr = "" if s.get("opennessCorr") is None else f" · 상관 {s['opennessCorr']:+.2f}"
        print(f"  {name}: {s['strongPct']:5.1f}% · {gap} · {rel} · 칸 300 = 둘째 {s['laterPct']:4.1f}%{corr}"
              + ("" if s["pairs"] else " (대응표 없음)"), flush=True)
    print("\n2. 시험 녹음(손 정답 없음): 둘째 도착이 선 줄 · 간격 · 세기 · 칸 300 뒤 +20/+42/+60칸 포락선 중앙값(dB)")
    for name in sorted(os.listdir(os.path.join(args.data, "recordings"))):
        r = recording_summary(args.data, name)
        result["recordings"][name] = r
        gap = "—" if r["gapMedian"] is None else f"{r['gapMedian']:.0f}칸"
        rel = "—" if r["relMedian"] is None else f"{r['relMedian']:+.1f} dB"
        print(f"  {name} 창 {r['window']} 볼륨 {r['volume']}: {r['strongPct']:5.1f}% · {gap} · {rel} · "
              + "/".join(f"{v:+.1f}" for v in r["envAt"].values()), flush=True)
    print("\n3. 자세별 둔덕 자리(칸)·세기(dB, 둘째 − 첫)·칸 300을 둘째 도착에 둔 줄(%)")
    for name, s in result["sessions"].items():
        if not s.get("pairs"):
            continue
        print(f"  {name}: " + " | ".join(
            f"{lab.split('(')[0]} " + ("—" if s["poses"][c]["gap"] is None else f"{s['poses'][c]['gap']:.1f}칸/{s['poses'][c]['rel']:+.1f}dB")
            + f"/{s['poses'][c]['laterPct']:.0f}%" for c, lab in POSES))
    with_hump = [s for s in result["sessions"].values() if s.get("pairs") and s["poses"]["10"]["gap"] is not None]
    for key, unit in (("gap", "칸"), ("rel", "dB")):
        print(f"  숫자 5 대비 {key}({unit}) 세션 중앙값: " + ", ".join(
            f"{lab.split('(')[0]} {np.nanmedian([np.nan if s['poses'][c][key] is None else s['poses'][c][key] - s['poses']['10'][key] for s in with_hump]):+.1f}"
            for c, lab in POSES[1:]) + f" ({len(with_hump)}세션)")
    sim_res = None
    if not args.no_sim:
        print("\n4. 시뮬레이션(hand_echo_sim, 세션 " + args.sim_session + ")", flush=True)
        sim_res = simulate(args.data, args.sim_session)
        e = sim_res["expectShift"]
        print(f"  손가락 반사점 세기 가중 거리(숫자 5) {sim_res['fingerRangeBins']:.1f}칸 · 손가락 메아리였다면 둔덕 이동(칸): 숫자 0 {e['5']:+.1f}, "
              f"손목 젖히기 30° {e['15']:+.1f}, 굽히기 30° {e['16']:+.1f}, 새끼 쪽 25° {e['17']:+.1f}, 엄지 쪽 25° {e['thumbSide25']:+.1f}")
        h = sim_res["handOnly"]
        print(f"  손 메아리만(한 창, 직접 경로 = 0 dB): 숫자 5 가장 센 칸 {h['10']['peakBin']}({(h['10']['peakBin'] - 300) * BIN_CM:.1f} cm) "
              f"{h['10']['peakDb']:+.1f} dB, 칸 342 {h['10']['bin342Db']:+.1f} dB · 숫자 0 칸 342 {h['5']['bin342Db']:+.1f} dB · "
              f"직접 경로 꼬리만 칸 342 {h['directTail342Db']:+.1f} dB")
        result["simulation"] = sim_res
    path = figure(args.out, args.data, result["sessions"], sim_res)
    with open(os.path.join(args.out, "second_arrival.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"\n그림 {path}")


if __name__ == "__main__":
    main()
