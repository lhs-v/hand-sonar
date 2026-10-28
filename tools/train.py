"""FastViT-T12로 에코 창에서 손 관절 20점을 회귀하고 평가한다(손 소나 계획 4, 학습 설계 §4·§5).

    python -I tools/train.py --mode overfit|time [--run 이름] [--data 데이터 폴더] [--out 학습 결과 폴더] [--epochs 200] [--seed 0] [세션 …]

- 자료: dataset.build(설계 §2·§3). 결과는 <out>/<run>/(기본 <data>/processed/training/<run>/)에 dataset.npz·model.pt·metrics.json·
  report.png로 쓴다. 저장소에는 아무것도 쓰지 않는다.
- 모델: timm fastvit_t12(처음부터 학습, ImageNet 가중치 없음), 입력 (N, 2, 96, 60), 출력 60값(단위 손 손바닥 틀의 점 1–20).
- 손실: MSE(60값) + 0.1 × 속도 손실. 속도 손실은 묶음 안 이웃 프레임의 예측 차분과 정답 차분의 MSE다(예측 움직임 자체는 벌하지
  않는다 — 평균 쏠림을 키우지 않게). 미니배치는 연속 프레임 32개 묶음 여럿(모두 256프레임 안팎)이다. 덩어리(세션이 같고 이웃 프레임
  간격이 0.2 s 안)마다 묶음 시작을 epoch마다 무작위로 밀고, 묶음 순서를 섞는다.
- 최적화: Adam 1e-3. 검증 손실(같은 손실, 평가 모드)이 9 epoch 동안 나아지지 않으면 멈추고 가장 좋던 가중치로 되돌린다. 최대 --epochs.
  씨앗값을 고정하고 cuDNN 결정적 계산을 쓴다(benchmark 끔. 이 계획을 쓰며 잰 값: 한 걸음이 0.067 → 0.075 s).
- 나누기: overfit = 모든 프레임으로 학습·검증·평가(외울 수 있는가), time = dataset.time_split(세션마다 70/15/15, 경계마다 2 s 비움).
- 평가(설계 §5, 시험 프레임의 예측·정답, mm = 단위 손 × 세션 L): MPJPE와 움직이는 점 MPJPE, 학습 평균 자세·메도이드 기준선,
  평균 자세 대비 비, 붕괴 비, 뼈 길이 부족(마디별, 평균 자세의 값과 함께), 동작별 MPJPE, ASL-5(숫자 5) 쪽 예측 비율,
  정답 잡음 바닥. 잡음 바닥은 예측이 필요 없어 모든 프레임의 정답으로 잰다(시간 나누기의 시험 구간은 자유 동작뿐이라 2 s 동작이 없다).
- 그림 report.png 한 장: 시험 구간 처음 30 s의 손끝 셋(엄지·검지·약지) 손목까지 거리(정답·예측·평균 자세) + MPJPE 막대
  (평균 자세·메도이드·FastViT, 잡음 바닥 선) + 계기 숫자. 영상 프레임이나 사람 모습은 넣지 않는다.
"""
import argparse
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # python -I는 3.11부터 -P를 겸해 스크립트 폴더를 경로에 넣지 않는다
import hs_io  # noqa: E402  사용자 설치 numpy·torch·timm 경로도 여기서 붙는다
import dataset  # noqa: E402
import joints  # noqa: E402
import numpy as np  # noqa: E402
import timm  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

ARCH = "fastvit_t12"
LR = 1e-3
BATCH = 256  # 미니배치 프레임 수(안팎)
CHUNK = 32  # 연속 프레임 묶음
CHUNK_GAP_SEC = 0.2  # 이웃 프레임 간격이 이보다 길면 덩어리를 끊는다(15–30 fps 카메라 간격 33–67 ms의 3배 넘게)
VELOCITY_WEIGHT = 0.1
PATIENCE = 9
EPOCHS = 200  # 최대 epoch(설계에 없음. 조기 종료가 먼저 멈춘다)
EVAL_BLOCK = 512
ASL5 = 10  # prompts.csv classId: 숫자 5(편 손)
FLOOR_MIN_FRAMES = 5  # 잡음 바닥: 프레임이 이보다 적은 유지 구간은 뺀다
EXCERPT_SEC = 30.0
FIG_TIPS = (4, 8, 16)  # 그림의 손끝: 엄지·검지·약지
SURFACE, INK, INK2, MUTED, GRID, BASE, BLUE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#2a78d6"


def velocity_loss(pred, true, pair):
    """속도 손실: pair[k]가 참인 이웃 (k, k+1)마다 (예측 차분 − 정답 차분)²의 60값 평균, 그 쌍들의 평균. pred·true (B, 60), pair (B−1,) bool."""
    d = (pred[1:] - pred[:-1]) - (true[1:] - true[:-1])
    w = pair.to(d.dtype)
    return ((d ** 2).mean(dim=1) * w).sum() / w.sum().clamp(min=1.0)


def loss_fn(pred, true, pair):
    return F.mse_loss(pred, true) + VELOCITY_WEIGHT * velocity_loss(pred, true, pair)


def segments(t_ns, session, index, gap_sec=CHUNK_GAP_SEC):
    """index를 세션·시각 순서로 놓고, 세션이 바뀌거나 이웃 간격이 gap_sec를 넘는 곳에서 끊은 덩어리 목록."""
    index = np.asarray(index, dtype=np.int64)
    index = index[np.lexsort((t_ns[index], session[index]))]
    cut = np.flatnonzero((np.diff(t_ns[index]) > gap_sec * 1e9) | (np.diff(session[index]) != 0)) + 1
    return np.split(index, cut)


def pairs_of(pieces):
    """묶음들을 이은 순서에서 이웃 (k, k+1)이 같은 묶음 안인가 (총 − 1,)."""
    return np.concatenate([np.r_[np.ones(len(p) - 1, dtype=bool), False] for p in pieces])[:-1]


def batches(segs, rng, chunk=CHUNK, batch=BATCH):
    """한 epoch의 미니배치 [(프레임 번호, 이웃 쌍 표시)]. 덩어리마다 시작을 무작위로 밀어 chunk개씩 자르고, 묶음 순서를 섞어 고르게 나눈다."""
    pieces = []
    for seg in segs:
        first = int(rng.integers(chunk)) if len(seg) > chunk else 0
        pieces += [p for p in np.split(seg, [c for c in range(first, len(seg), chunk) if c > 0]) if len(p)]
    order = rng.permutation(len(pieces))
    count = max(1, int(np.ceil(sum(len(p) for p in pieces) / batch)))
    out = []
    for group in np.array_split(order, count):
        chosen = [pieces[k] for k in group]
        out.append((np.concatenate(chosen), pairs_of(chosen)))
    return out


def seed_all(seed):
    """씨앗값과 값싼 결정적 설정(cuDNN 결정적 알고리즘, benchmark 끔)."""
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU와 모든 GPU
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_model():
    return timm.create_model(ARCH, pretrained=False, in_chans=2, num_classes=60)


def predict(model, x, index):
    """index 프레임의 예측 (len, 60) float32 numpy(평가 모드)."""
    model.eval()
    idx = torch.as_tensor(np.asarray(index, dtype=np.int64), device=x.device)
    with torch.no_grad():
        out = [model(x[idx[a: a + EVAL_BLOCK]]).float().cpu() for a in range(0, len(idx), EVAL_BLOCK)]
    return torch.cat(out).numpy() if out else np.zeros((0, 60), dtype=np.float32)


def eval_loss(model, x, y, segs):
    """덩어리들의 손실(평가 모드, 이웃 쌍은 덩어리 안만)."""
    idx = np.concatenate(segs)
    pred = torch.as_tensor(predict(model, x, idx), device=x.device)
    return float(loss_fn(pred, y[torch.as_tensor(idx, device=x.device)], torch.as_tensor(pairs_of(segs), device=x.device)))


def fit(x, y, train_segs, val_segs, epochs, seed, log=print):
    """(가장 좋던 가중치의 모델, 기록, 가장 좋던 epoch). x (N, 2, 96, 60)·y (N, 60)는 같은 장치의 텐서."""
    torch.manual_seed(seed)
    model = make_model().to(x.device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(seed)
    best, best_epoch, best_state, history = float("inf"), 0, None, []
    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0
        for idx, pair in batches(train_segs, rng):
            i = torch.as_tensor(idx, device=x.device)
            loss = loss_fn(model(x[i]), y[i], torch.as_tensor(pair, device=x.device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
            count += len(idx)
        val = eval_loss(model, x, y, val_segs)
        history.append({"epoch": epoch, "train": total / count, "val": val})
        if val < best:
            best, best_epoch = val, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            log(f"  epoch {epoch}: 학습 손실 {total / count:.5f}, 검증 손실 {val:.5f}(최고 epoch {best_epoch})")
        if epoch - best_epoch >= PATIENCE:
            log(f"  epoch {epoch}: 검증 손실이 {PATIENCE} epoch 동안 나아지지 않아 멈춤(최고 epoch {best_epoch}, {best:.5f})")
            break
    model.load_state_dict(best_state)
    return model, history, best_epoch


def instruments(ds, pred, train, test):
    """설계 §5 계기 dict. pred (N, 60)는 ds 모든 프레임의 예측이다. 기준선과 ASL-5 평균은 학습 프레임에서, 나머지는 시험 프레임에서 잰다."""
    y = ds["y"].reshape(-1, 20, 3).astype(np.float64)
    p = np.asarray(pred, dtype=np.float64).reshape(-1, 20, 3)
    mm = ds["length_mm"][ds["session"]].astype(np.float64)  # 프레임마다 그 세션의 L
    true, guess, scale = y[test], p[test], mm[test]
    mean, med = joints.mean_pose(y[train]), joints.medoid(y[train])

    def mpjpe(q, points=None):
        return float(np.mean(joints.frame_errors(q, true, points) * scale))

    out = {"mpjpeMm": mpjpe(guess), "movingMpjpeMm": mpjpe(guess, joints.MOVING),
           "baselines": {"meanPoseMm": mpjpe(mean[None]), "medoidMm": mpjpe(med[None]),
                         "meanPoseMovingMm": mpjpe(mean[None], joints.MOVING), "medoidMovingMm": mpjpe(med[None], joints.MOVING)}}
    out["ratioToMeanPose"] = out["mpjpeMm"] / out["baselines"]["meanPoseMm"]
    ratio = joints.collapse(guess, true)
    finger = np.array([next(k for k, (_, idx) in enumerate(joints.FINGERS) if j + 1 in idx) for j in joints.MOVING])
    out["collapse"] = {"mean": float(ratio.mean()), "min": float(ratio.min()),
                       "byFinger": {joints.FINGERS[k][0]: float(ratio[finger == k].mean()) for k in range(5)}}
    out["boneShortfallPercent"] = {"model": dict(zip(joints.SEGMENTS, joints.by_segment(joints.bone_shortfall(guess, true)).tolist())),
                                   "meanPose": dict(zip(joints.SEGMENTS, joints.by_segment(joints.bone_shortfall(mean[None], true)).tolist()))}
    labelled, cls = ds["labelled"][test], ds["class_id"][test]
    e_model = joints.frame_errors(guess, true) * scale
    e_mean = joints.frame_errors(mean[None], true) * scale
    names = dict(zip(ds["label_ids"].tolist(), ds["label_names"].tolist()))
    out["perLabel"] = [{"classId": int(c), "name": names.get(int(c), "?"), "frames": int(m.sum()),
                        "mpjpeMm": float(e_model[m].mean()), "meanPoseMm": float(e_mean[m].mean())}
                       for c in np.unique(cls[labelled]) for m in [labelled & (cls == c)]]
    five = ds["labelled"][train] & (ds["class_id"][train] == ASL5)
    other = labelled & (cls != ASL5)
    if five.any() and other.any():
        template = y[train][five].mean(axis=0)
        out["asl5Closer"] = {"model": joints.closer_to(guess[other], true[other], template),
                             "meanPose": joints.closer_to(np.broadcast_to(mean, true[other].shape), true[other], template),
                             "frames": int(other.sum())}
    hold = ds["labelled"] & (ds["class_id"] >= 0)  # 2 s 동작(자유 동작 −1·일시정지 −2 아님)의 경계 0.5 s 뒤 유지 구간
    groups = ds["session"][hold].astype(np.int64) * 1_000_000 + ds["prompt"][hold]
    floor = joints.hold_floor(y[hold], groups, FLOOR_MIN_FRAMES) * mm[hold]
    out["noiseFloorMm"] = float(np.nanmean(floor)) if np.isfinite(floor).any() else None
    out["noiseFloorFrames"] = int(np.isfinite(floor).sum())
    return out


def broken(t, v, gap_sec=CHUNK_GAP_SEC):
    """시각 간격이 gap_sec를 넘는 곳에 nan을 끼워 선이 끊기게 한다."""
    k = np.flatnonzero(np.diff(t) > gap_sec) + 1
    return np.insert(t.astype(np.float64), k, np.nan), np.insert(v.astype(np.float64), k, np.nan)


def figure(path, ds, pred, train, test, m, title):
    """report.png 한 장(왼쪽 손끝 시계열 셋, 오른쪽 MPJPE 막대와 계기 숫자)."""
    plt = hs_io.pyplot()
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": BASE,
                         "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 9})
    y = ds["y"].reshape(-1, 20, 3)
    p = np.asarray(pred).reshape(-1, 20, 3)
    mean = joints.mean_pose(y[train])
    order = np.asarray(test)[np.lexsort((ds["frame_ns"][test], ds["session"][test]))]
    order = order[ds["session"][order] == ds["session"][order[0]]]  # 시험 구간의 첫 세션
    t = (ds["frame_ns"][order] - ds["frame_ns"][order[0]]) / 1e9
    order, t = order[t <= EXCERPT_SEC], t[t <= EXCERPT_SEC]
    scale = float(ds["length_mm"][ds["session"][order[0]]])
    fig = plt.figure(figsize=(13.5, 7.4), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, width_ratios=[2.2, 1.0])
    names = {tip: name for name, idx in joints.FINGERS for tip in idx[-1:]}
    for row, tip in enumerate(FIG_TIPS):
        ax = fig.add_subplot(grid[row, 0])
        ax.axhline(float(np.linalg.norm(mean[tip - 1])) * scale, color=MUTED, lw=1.0, ls="--", label="평균 자세(학습)")
        for q, color, label in ((y, INK, "정답(MediaPipe)"), (p, BLUE, "FastViT 예측")):
            ax.plot(*broken(t, np.linalg.norm(q[order, tip - 1], axis=1) * scale), color=color, lw=1.3, label=label)
        ax.set_ylabel(f"{names[tip]} 끝–손목(mm)")
        ax.grid(axis="y", color=GRID, lw=0.7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if row == 0:
            ax.legend(loc="upper right", ncol=3, fontsize=8.5, frameon=False)
            ax.set_title(f"시험 구간 처음 {t[-1]:.0f} s(손바닥 틀, 단위 손 × L {scale:.1f} mm)", loc="left", fontsize=10, color=INK)
    ax.set_xlabel("시험 구간 시작부터의 시간(s)")
    ax = fig.add_subplot(grid[0, 1])
    labels = ["평균 자세", "메도이드", "FastViT"]
    values = [m["baselines"]["meanPoseMm"], m["baselines"]["medoidMm"], m["mpjpeMm"]]
    bars = ax.bar(labels, values, color=[BASE, BASE, BLUE], width=0.6)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9, color=INK)
    if m["noiseFloorMm"] is not None:
        ax.axhline(m["noiseFloorMm"], color=INK2, lw=1.0, ls=":")
        ax.set_title(f"점선 = 정답 잡음 바닥 {m['noiseFloorMm']:.1f} mm", loc="left", fontsize=9, color=INK2)
    ax.set_ylabel("시험 MPJPE(mm)")
    ax.set_ylim(0, max(values) * 1.18)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax = fig.add_subplot(grid[1:, 1])
    ax.axis("off")
    c, s = m["collapse"], m["boneShortfallPercent"]
    text = [f"평균 자세 대비 비  {m['ratioToMeanPose']:.2f}",
            f"MPJPE {m['mpjpeMm']:.1f} mm, 움직이는 점 {m['movingMpjpeMm']:.1f} mm(평균 자세 {m['baselines']['meanPoseMovingMm']:.1f})",
            f"붕괴 비(예측 σ ÷ 정답 σ, 움직이는 점) 평균 {c['mean']:.2f}",
            "  " + " · ".join(f"{k} {v:.2f}" for k, v in c["byFinger"].items()),
            "뼈 길이 부족(%) 손바닥 · 첫 · 가운데 · 끝 마디",
            "  FastViT " + " · ".join(f"{v:.1f}" for v in s["model"].values()),
            "  평균 자세 " + " · ".join(f"{v:.1f}" for v in s["meanPose"].values())]
    if "asl5Closer" in m:
        a = m["asl5Closer"]
        text.append(f"정답보다 ASL-5 평균에 가까운 예측 {a['model'] * 100:.0f}%(평균 자세 {a['meanPose'] * 100:.0f}%, {a['frames']}프레임)")
    text.append(f"학습 {m['frames']['train']} · 검증 {m['frames']['val']} · 시험 {m['frames']['test']}프레임, epoch {m['bestEpoch']}/{m['epochs']}")
    for k, line in enumerate(text):
        ax.text(0.0, 1.0 - k * 0.085, line, transform=ax.transAxes, va="top", fontsize=11 if k == 0 else 9,
                fontweight="bold" if k == 0 else "normal", color=INK if k == 0 else INK2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def report(m):
    """출력할 요약 줄."""
    b, c, s = m["baselines"], m["collapse"], m["boneShortfallPercent"]
    lines = [f"평균 자세 대비 비 {m['ratioToMeanPose']:.3f}: FastViT {m['mpjpeMm']:.2f} mm / 평균 자세 {b['meanPoseMm']:.2f} mm(메도이드 {b['medoidMm']:.2f} mm)",
             f"  움직이는 점 MPJPE {m['movingMpjpeMm']:.2f} mm(평균 자세 {b['meanPoseMovingMm']:.2f}, 메도이드 {b['medoidMovingMm']:.2f})",
             f"  붕괴 비 평균 {c['mean']:.3f}(최소 {c['min']:.3f}), 손가락별 " + ", ".join(f"{k} {v:.2f}" for k, v in c["byFinger"].items()),
             "  뼈 길이 부족 % FastViT " + ", ".join(f"{k} {v:.1f}" for k, v in s["model"].items()),
             "                 평균 자세 " + ", ".join(f"{k} {v:.1f}" for k, v in s["meanPose"].items())]
    if "asl5Closer" in m:
        a = m["asl5Closer"]
        lines.append(f"  ASL-5가 아닌 동작에서 정답보다 ASL-5 평균에 가까운 예측 {a['model'] * 100:.1f}%(평균 자세 {a['meanPose'] * 100:.1f}%, {a['frames']}프레임)")
    if m["noiseFloorMm"] is not None:
        lines.append(f"  정답 잡음 바닥 {m['noiseFloorMm']:.2f} mm({m['noiseFloorFrames']}프레임)")
    lines.append("  동작별 MPJPE(mm, 평균 자세): " + ", ".join(f"{r['name']} {r['mpjpeMm']:.1f}({r['meanPoseMm']:.1f}, {r['frames']})" for r in m["perLabel"]))
    lines.append(f"  나누기별 MPJPE: " + ", ".join(f"{k} {v:.2f} mm" for k, v in m["splitMpjpeMm"].items()))
    return lines


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="FastViT-T12로 에코 창에서 손 관절 20점을 학습·평가한다")
    parser.add_argument("sessions", nargs="*", help="폰 세션 이름(없으면 쓸 수 있는 세션 모두)")
    parser.add_argument("--mode", choices=("overfit", "time"), required=True, help="overfit: 모든 프레임으로 학습·평가, time: 시간 나누기")
    parser.add_argument("--run", help="실행 이름(없으면 <날짜-시각>-<mode>)")
    parser.add_argument("--data", default=hs_io.DEFAULT_DATA)
    parser.add_argument("--out", help="학습 결과를 둘 폴더(없으면 <data>/processed/training)")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="최대 epoch")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run = args.run or f"{time.strftime('%Y%m%d-%H%M%S')}-{args.mode}"
    folder = os.path.join(args.out or os.path.join(args.data, "processed", "training"), run)
    ds, lines = dataset.build(args.data, args.sessions)
    print("\n".join(lines))
    if ds is None:
        print("쓸 세션이 없음")
        return 1
    n = len(ds["y"])
    if args.mode == "overfit":
        train = val = test = np.arange(n)
    else:
        train, val, test = dataset.time_split(ds["frame_ns"], ds["session"])
    if min(len(train), len(val), len(test)) == 0:
        print(f"나눈 자료가 비었음(학습 {len(train)} · 검증 {len(val)} · 시험 {len(test)})")
        return 1
    os.makedirs(folder, exist_ok=True)
    dataset.save(os.path.join(folder, "dataset.npz"), ds)
    seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{run}: {args.mode}, 학습 {len(train)} · 검증 {len(val)} · 시험 {len(test)}프레임, 장치 {device}"
          f"{'(' + torch.cuda.get_device_name(0) + ')' if device == 'cuda' else ' — GPU가 없어 아주 느리다'}")
    x = torch.as_tensor(ds["x"], device=device)
    y = torch.as_tensor(ds["y"], device=device)
    start = time.time()
    model, history, best_epoch = fit(x, y, segments(ds["frame_ns"], ds["session"], train), segments(ds["frame_ns"], ds["session"], val),
                                     args.epochs, args.seed)
    seconds = time.time() - start
    pred = predict(model, x, np.arange(n))
    m = instruments(ds, pred, train, test)
    yy = ds["y"].reshape(-1, 20, 3)
    mm = ds["length_mm"][ds["session"]]
    m["splitMpjpeMm"] = {k: float(np.mean(joints.frame_errors(pred.reshape(-1, 20, 3)[v], yy[v]) * mm[v]))
                         for k, v in (("train", train), ("val", val), ("test", test))}
    m.update({"format": 1, "run": run, "mode": args.mode, "sessions": ds["sessions"].tolist(), "lengthMm": ds["length_mm"].tolist(),
              "frames": {"train": int(len(train)), "val": int(len(val)), "test": int(len(test))}, "seed": args.seed,
              "epochs": len(history), "maxEpochs": args.epochs, "bestEpoch": best_epoch, "trainSeconds": seconds,
              "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu", "history": history})
    torch.save({"arch": ARCH, "in_chans": 2, "num_classes": 60, "bins": [dataset.BINS.start, dataset.BINS.stop], "window": dataset.WINDOW,
                "mode": args.mode, "seed": args.seed, "epoch": best_epoch, "sessions": ds["sessions"].tolist(), "state_dict": model.state_dict()},
               os.path.join(folder, "model.pt"))
    hs_io.write_json(os.path.join(folder, "metrics.json"), m)
    what = "과적합 점검(모든 프레임으로 학습·평가)" if args.mode == "overfit" else "시간 나누기(앞 70% 학습 · 15% 검증 · 끝 15% 시험, 경계 2 s 비움)"
    figure(os.path.join(folder, "report.png"), ds, pred, train, test, m, f"{run}: {what} — 세션 {', '.join(m['sessions'])}")
    print("\n".join([f"{run}: {len(history)} epoch, {seconds / 60:.1f}분, 최고 epoch {best_epoch}"] + report(m)))
    print(f"결과: {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
