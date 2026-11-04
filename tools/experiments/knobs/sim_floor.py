"""이론·시뮬레이션: 잡음 바닥·손 신호 값들이 어떤 손잡이로 움직이나. 실제 처리(profiles.tx_template·profiles.compute)를 그대로 거친다.

    python -I tools/experiments/knobs/sim_floor.py --out <저장소 밖 폴더> [--data 우리 자료] [--pub 공개 자료]

수신 줄 t(600샘플, 주기 처프의 정상 상태라 원형):
    Y_t(f) = TX(f) · e^{−2πi f δ_t/600} · (1 + ε_t) · [H_s(f) + a_f · e^{−2πi f τ_f(t)/600}],  x = 줄들을 이은 것 + σ_n · rms(tx) · 흰 잡음
- H_s: 정적 반사(직접 경로 = 1, 손 자리 지연 14–60샘플과 먼 지연 61–299샘플에 가우스 탭 — 톱니 처리에서 잰 평균 |원본| 세기에 맞춤)
- 손가락: 지연 30샘플(약 10.7 cm)의 반사체 하나, 20 mm를 0.3 s 동안 올림 코사인으로 갔다 1.2 s 멈추고 되돌아옴(최고 약 105 mm/s)
- ε_t: 줄마다 세기 흔들림, δ_t: 줄마다 시각 흔들림(샘플), σ_n: 더해지는 잡음
값은 knobs.py와 같은 정의(직접 경로 = 0 dB): 바닥(칸 450–599 |차분|), 움직일 때·멈췄을 때 손 자리(칸 315–359, 5줄 평활), 흔들림(직접 경로
|원본| 줄 변화 중앙값), 오르내림 = 움직일 때 − 멈췄을 때. 결과는 sim.json.
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
from scipy.signal import hilbert  # noqa: E402

N, FS = P.N, P.RATE
ROW_SEC = N / FS
MM_PER_BIN = 343.0 / FS / 2 * 1000  # 반사체가 이만큼(mm) 멀어지면 지연 1샘플(왕복)
HAND, FAR = slice(315, 360), slice(450, 600)
SKIP = 40  # 필터 가장자리 줄
F = np.arange(N // 2 + 1)
MEASURED = {  # knobs.json 중앙값(세션 묶음별)과 평균 |원본| 세기(직접 경로 기준)
    "public": {"shape": "saw", "jitter": 0.12, "floor": -77.8, "rest": -69.1, "move": -57.2, "swing": 11.9, "hand_raw": -13.5, "far_raw": -38.1},
    "ours_saw": {"shape": "saw", "jitter": 0.215, "floor": -77.3, "rest": -68.4, "move": -63.2, "swing": 5.3, "hand_raw": -19.2, "far_raw": -37.3},
    "ours_tri": {"shape": "triangle", "jitter": 0.29, "floor": -67.5, "rest": -63.9, "move": -60.2, "swing": 3.9, "hand_raw": -15.5, "far_raw": -22.3},
}


def taps(hand_db, far_db, seed=1):
    """정적 반사 탭 (600,): 직접 경로 1, 손 자리(14–60)·먼 지연(61–299) 가우스(세기는 탭 rms, dB)."""
    rng = np.random.default_rng(seed)
    h = np.zeros(N)
    h[0] = 1.0
    h[14:61] = rng.standard_normal(47) * 10 ** (hand_db / 20)
    h[61:300] = rng.standard_normal(239) * 10 ** (far_db / 20) * np.exp(-np.arange(239) / 400)
    return h


def taps_frac(hand_db, far_db, seed=1):
    """taps()와 같은 세기의 반사체를 칸 사이 무작위 자리(소수 지연)에 둔 스펙트럼 (301,) complex. 정수 칸에만 둔 반사체는 반송파와 칸 격자가
    맞물려 대역 자리에 따라 오르내림이 ±1 dB 엇갈렸다(sim_band.py 시험) — 실제 반사체는 칸 사이 어디에나 있다."""
    h = taps(hand_db, far_db, seed)
    u = np.random.default_rng(100 + seed).uniform(0.0, 1.0, N)
    u[0] = 0.0  # 직접 경로는 칸 0
    return (h[:, None] * np.exp(-2j * np.pi * (np.arange(N) + u)[:, None] * F[None, :] / N)).sum(axis=0)


def finger_track(rows):
    """줄마다 손가락 지연 변화(샘플)와 속도(mm/s). 0.3 s 이동 · 1.2 s 멈춤을 되풀이(갔다 오기)."""
    t = np.arange(rows) * ROW_SEC
    period, move, stroke = 3.0, 0.3, 20.0
    ph = t % period
    pos = np.where(ph < move, 0.5 - 0.5 * np.cos(np.pi * ph / move), 1.0)
    back = (ph >= 1.5) & (ph < 1.5 + move)
    pos = np.where(ph >= 1.5, 1.0, pos)
    pos = np.where(back, 0.5 + 0.5 * np.cos(np.pi * (ph - 1.5) / move), pos)
    pos = np.where(ph >= 1.5 + move, 0.0, pos)
    mm = stroke * pos
    speed = np.abs(np.gradient(mm, ROW_SEC))
    return mm / MM_PER_BIN, speed


def simulate_raw(shape, sig_eps, sig_tau, noise_db, a_f_db, H, seconds=30.0, seed=0, finger_delay=30.0):
    """가상 녹음을 실제 처리에 넣은 원본(600, 줄 — 필터 가장자리 줄 뺌)과 차분 열마다 움직임·멈춤 (열,) bool.
    H = 정적 반사 시간 탭 (600,)(정수 지연, taps) 또는 스펙트럼 (301,) complex(소수 지연, taps_frac).
    finger_delay = 손가락이 멈춰 있을 때의 지연(샘플, 기본 30 = 약 10.7 cm)."""
    rng = np.random.default_rng(seed)
    tx = P.tx_template("none", 0.5, shape)
    TX = np.fft.rfft(tx)
    R = int(seconds / ROW_SEC)
    delay, speed = finger_track(R)
    Hs = H if np.iscomplexobj(H) else np.fft.rfft(H)
    a_f = 10 ** (a_f_db / 20) if a_f_db is not None else 0.0
    Hf = a_f * np.exp(-2j * np.pi * F[None, :] * (finger_delay + delay)[:, None] / N)
    gain = 1 + sig_eps * rng.standard_normal(R)
    shift = np.exp(-2j * np.pi * F[None, :] * (sig_tau * rng.standard_normal(R))[:, None] / N)
    rows = np.fft.irfft(TX[None, :] * shift * gain[:, None] * (Hs[None, :] + Hf), n=N, axis=1)
    x = rows.reshape(-1) + 10 ** (noise_db / 20) * np.sqrt(np.mean(tx ** 2)) * rng.standard_normal(R * N)
    p = P.compute(x, tx, direct_floor=P.direct_min(shape))
    raw = p.raw[:, SKIP:-SKIP]
    sp = speed[SKIP + 1: SKIP + raw.shape[1]]
    moving = sp > 80.0
    t = np.arange(len(sp)) * ROW_SEC
    near = np.zeros(len(sp), dtype=bool)
    for tm in t[moving]:
        near |= np.abs(t - tm) < 0.3
    return raw, moving, (sp < 15.0) & ~near


def simulate(shape, sig_eps, sig_tau, noise_db, a_f_db, H, seconds=30.0, seed=0, row_norm=False):
    raw, moving, rest = simulate_raw(shape, sig_eps, sig_tau, noise_db, a_f_db, H, seconds, seed)
    if row_norm:
        import dataset
        raw = dataset.row_normalized(raw)[0]
    return values(raw, moving, rest, a_f_db is not None)


def values(raw, moving, rest, finger=True):
    """knobs.py와 같은 정의의 값(직접 경로 = 0 dB). finger가 거짓이면 움직임은 nan."""
    d = np.abs(raw[:, 1:]) - np.abs(raw[:, :-1])
    direct = np.abs(raw[300]).mean()
    e_hand = np.convolve(np.abs(d[HAND]).mean(axis=0), np.ones(5) / 5, mode="same")
    e_far = np.abs(d[FAR]).mean(axis=0)
    db = lambda v: float(20 * np.log10(v / direct))  # noqa: E731
    out = {"floor": db(np.median(e_far)), "rest": db(np.median(e_hand[rest])), "move": db(np.median(e_hand[moving])) if finger else float("nan"),
           "jitter": float(np.median(np.abs(np.diff(np.abs(raw[300]))) / np.abs(raw[300])[1:]) * 100),
           "hand_raw": db(np.abs(raw[HAND]).mean()), "far_raw": db(np.abs(raw[FAR]).mean())}
    out["swing"] = out["move"] - out["rest"]
    return out


def autocorr_env(shape):
    tx = P.tx_template("none", 0.5, shape)
    c = np.fft.irfft(np.abs(np.fft.rfft(tx)) ** 2, n=N)
    env = np.abs(hilbert(c))
    return np.roll(env / env.max(), N // 2)  # 칸 300 = 0 지연


def fit_static(shape, hand_target, far_target, ref=(-45.0, -65.0)):
    """톱니 처리에서 잰 평균 |원본| 세기(손 자리·먼 칸)에 맞는 탭 세기(dB). 직접 경로를 넘지 않는 작은 기준 탭으로 한 번 재고 선형으로 옮긴다
    (상관 출력은 탭에 선형. 반복해서 맞추면 손 자리 반사가 직접 경로를 넘어 맞춤이 엉뚱한 봉우리를 잡았다)."""
    r = simulate(shape, 0.0, 0.0, -150, None, taps(*ref), seconds=4.0)
    return ref[0] + hand_target - r["hand_raw"], ref[1] + far_target - r["far_raw"]


def main():
    res = {"autocorr": {s: autocorr_env(s).tolist() for s in ("saw", "triangle")}}
    # 1) 정적 반사: 공개·우리 톱니 실측에 맞춘 탭(물리는 같은데 삼각으로 처리하면 먼 칸이 얼마나 오르나)
    stat = {k: fit_static("saw", MEASURED[k]["hand_raw"], MEASURED[k]["far_raw"]) for k in ("public", "ours_saw")}
    H_pub, H_ours = taps(*stat["public"]), taps(*stat["ours_saw"])
    tri_static = simulate("triangle", 0.0, 0.0, -150, None, H_ours, seconds=4.0)
    res["static"] = {"tapsDb": stat, "oursTriangleRaw": tri_static}
    print(f"정적 반사 탭: 공개 {stat['public']}, 우리 {stat['ours_saw']} · 같은 반사를 삼각으로 처리하면 손 자리 {tri_static['hand_raw']:.1f} · 먼 칸 {tri_static['far_raw']:.1f} dB"
          f"(실측 삼각 {MEASURED['ours_tri']['hand_raw']} · {MEASURED['ours_tri']['far_raw']})")
    # 2) 손잡이 훑기(우리 정적 반사, 손가락 −40 dB): 세기 흔들림 · 시각 흔들림 · 더해지는 잡음 · 손가락 세기, 톱니·삼각
    sweeps = {}
    base = {"sig_eps": 0.0015, "sig_tau": 0.0, "noise_db": -45.0, "a_f_db": -40.0}
    grid = {"sig_eps": [0.0, 0.0005, 0.001, 0.002, 0.003, 0.005], "sig_tau": [0.0, 0.002, 0.005, 0.01, 0.02],
            "noise_db": [-70.0, -60.0, -50.0, -45.0, -40.0, -30.0], "a_f_db": [-55.0, -50.0, -45.0, -40.0, -35.0, -30.0]}
    for knob, values in grid.items():
        for shape in ("saw", "triangle"):
            rows = []
            for v in values:
                kw = dict(base, **{knob: v})
                rows.append({"value": v, **simulate(shape, kw["sig_eps"], kw["sig_tau"], kw["noise_db"], kw["a_f_db"], H_ours)})
            sweeps[f"{knob}:{shape}"] = rows
            print(f"훑기 {knob} {shape}: " + " · ".join(f"{r['value']}→바닥 {r['floor']:.1f} 멈춤 {r['rest']:.1f} 오르내림 {r['swing']:.1f} 흔들림 {r['jitter']:.2f}%" for r in rows))
    res["sweeps"] = sweeps
    # 3) 맞추기(잡음 쪽): 세기 흔들림 ε은 흔들림 지표에, 잡음은 톱니 바닥에(삼각은 같은 기기 톱니 날 잡음을 그대로 쓰고 바닥을 '예측'),
    #    멈췄을 때 손 자리가 모자라면 시각 흔들림 δ로 맞춘다. 손가락은 아직 같은 세기(−40 dB)
    def fit_noise_side(key, H, noise=None, a_f_db=-40.0):
        m = MEASURED[key]
        eps, tau, fixed = m["jitter"] / 100 / 0.2, 0.0, noise is not None
        noise = -50.0 if noise is None else noise
        for _ in range(5):
            r = simulate(m["shape"], eps, tau, noise, a_f_db, H, seconds=20.0)
            eps *= m["jitter"] / max(r["jitter"], 1e-6)
            if not fixed:
                noise = float(np.clip(noise + (m["floor"] - r["floor"]), -90.0, -15.0))
            if r["rest"] < m["rest"] - 0.5:  # 멈춤이 모자라면 시각 흔들림을 늘린다(로그 눈금)
                tau = max(tau, 0.001) * 10 ** ((m["rest"] - r["rest"]) / 20)
        return {"eps": eps, "tau": tau, "noiseDb": noise, **simulate(m["shape"], eps, tau, noise, a_f_db, H)}

    f_pub = fit_noise_side("public", H_pub)
    f_saw = fit_noise_side("ours_saw", H_ours)
    f_tri = fit_noise_side("ours_tri", H_ours, noise=f_saw["noiseDb"])
    # 4) 손가락 세기: 공개 오르내림(11.9 dB)에 맞추고, 같은 세기면 우리 오르내림이 몇 dB로 나오나(예측) · 우리 실측에 맞추면 몇 dB인가
    def fit_finger(key, f, H, a_f_db=-40.0):
        m = MEASURED[key]
        for _ in range(5):
            r = simulate(m["shape"], f["eps"], f["tau"], f["noiseDb"], a_f_db, H, seconds=20.0)
            a_f_db += m["swing"] - r["swing"]
        return a_f_db

    a_pub = fit_finger("public", f_pub, H_pub)
    pred = {k: simulate(MEASURED[k]["shape"], f["eps"], f["tau"], f["noiseDb"], a_pub, H) for k, f, H in (("ours_saw", f_saw, H_ours), ("ours_tri", f_tri, H_ours))}
    a_saw, a_tri = fit_finger("ours_saw", f_saw, H_ours), fit_finger("ours_tri", f_tri, H_ours)
    res["fit"] = {"public": f_pub, "ours_saw": f_saw, "ours_tri": f_tri, "fingerDb": {"public": a_pub, "ours_saw": a_saw, "ours_tri": a_tri},
                  "predictWithPublicFinger": pred}
    for k, f in (("공개", f_pub), ("우리 톱니", f_saw), ("우리 삼각(톱니 날 잡음 그대로)", f_tri)):
        m = MEASURED[{"공개": "public", "우리 톱니": "ours_saw"}.get(k, "ours_tri")]
        print(f"맞춤 {k}: ε {f['eps'] * 100:.2f}% · 시각 흔들림 {f['tau']:.4f}샘플 · 잡음 {f['noiseDb']:.1f} dB → 흔들림 {f['jitter']:.2f}%(실측 {m['jitter']}),"
              f" 바닥 {f['floor']:.1f}(실측 {m['floor']}), 멈춤 {f['rest']:.1f}(실측 {m['rest']})")
    print(f"손가락 세기: 공개 오르내림에 맞춤 {a_pub:.1f} dB → 같은 손가락이면 우리 톱니 오르내림 {pred['ours_saw']['swing']:.1f} dB(실측 5.3), "
          f"우리 삼각 {pred['ours_tri']['swing']:.1f} dB(실측 3.9). 우리 실측에 맞추면 톱니 {a_saw:.1f} · 삼각 {a_tri:.1f} dB")
    # 5) 줄 단위 정규화를 켜면(맞춘 손잡이 그대로, 손가락은 각자 맞춘 세기)
    res["rownorm"] = {k: simulate(MEASURED[k]["shape"], f["eps"], f["tau"], f["noiseDb"], a, H, row_norm=True)
                      for k, f, a, H in (("public", f_pub, a_pub, H_pub), ("ours_saw", f_saw, a_saw, H_ours), ("ours_tri", f_tri, a_tri, H_ours))}
    for k, r in res["rownorm"].items():
        print(f"정규화 {k}: 멈춤 {r['rest']:.1f} 오르내림 {r['swing']:.1f}")
    # 6) 우리 삼각에서 하나씩 공개처럼 바꿔 보기(손가락은 우리 삼각에 맞춘 세기)
    steps = [("우리 삼각 그대로", "triangle", f_tri["eps"], f_tri["tau"], f_tri["noiseDb"], a_tri, H_ours),
             ("+ 톱니", "saw", f_tri["eps"], f_tri["tau"], f_tri["noiseDb"], a_tri, H_ours),
             ("+ 공개만큼 작은 세기 흔들림", "saw", f_pub["eps"], f_tri["tau"], f_tri["noiseDb"], a_tri, H_ours),
             ("+ 공개 시각 흔들림", "saw", f_pub["eps"], f_pub["tau"], f_tri["noiseDb"], a_tri, H_ours),
             ("+ 공개 잡음", "saw", f_pub["eps"], f_pub["tau"], f_pub["noiseDb"], a_tri, H_ours),
             ("+ 공개 정적 반사", "saw", f_pub["eps"], f_pub["tau"], f_pub["noiseDb"], a_tri, H_pub),
             ("+ 공개 손가락 세기(= 공개)", "saw", f_pub["eps"], f_pub["tau"], f_pub["noiseDb"], a_pub, H_pub)]
    res["whatif"] = []
    for name, shape, e, tau, n, a, H in steps:
        r = simulate(shape, e, tau, n, a, H)
        res["whatif"].append({"step": name, **r})
        print(f"하나씩: {name}: 바닥 {r['floor']:.1f} 멈춤 {r['rest']:.1f} 움직임 {r['move']:.1f} 오르내림 {r['swing']:.1f}")
    json.dump(res, open(common.out("sim.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("저장: sim.json")


if __name__ == "__main__":
    main()
