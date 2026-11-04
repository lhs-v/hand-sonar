"""칸 머리 손실의 항별 크기와 기울기 — 무엇이 학습을 이끄나. 같은 학습 미니배치에서 세 항을 따로 재서, 처음(학습 전 가중치)과 학습 뒤(가장 좋던 가중치)를 견준다.

    python -I tools/experiments/knobs/loss_terms.py --out <저장소 밖 폴더> [--data 우리 자료] [--pub 공개 자료] [--run 실행 이름]

- 값: 좌표 MSE, 0.1 × 속도 손실, 0.01 × 칸 교차 엔트로피.
- 기울기: 항마다 따로 역전파한 전체 매개변수 기울기의 크기(L2), 그 가운데 출력층의 몫, 좌표 쪽(MSE + 속도)과 칸 쪽 기울기의 코사인.
결과는 loss_terms.json.
"""
import json
import os
import sys
from argparse import Namespace

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402  tools 경로를 붙이고 --out·--data·--pub를 읽는다
import hs_io  # noqa: E402
import dataset  # noqa: E402
import kinematics  # noqa: E402
import train  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

BATCHES = 20


def grads(model, loss):
    """모든 학습 매개변수의 기울기를 하나로 이은 벡터 (P,)와 출력층 몫."""
    params = [p for p in model.parameters() if p.requires_grad]
    gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    flat = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(gs, params)])
    return flat


def measure(model, x, y, angles, pieces, device):
    fc = sum(p.numel() for p in model.backbone.get_classifier().parameters())
    rows = []
    model.train()
    for idx, pair, _ in pieces:
        i = torch.as_tensor(idx, device=device)
        p = torch.as_tensor(pair, device=device)
        pred = model(x[i])
        mse = torch.nn.functional.mse_loss(pred[:, :60], y[i][:, :60])
        vel = train.VELOCITY_WEIGHT * train.velocity_loss(pred[:, :60], y[i][:, :60], p)
        ce = train.BIN_WEIGHT * kinematics.bin_loss(model.logits, angles[i], model.centers)
        gm, gv, gc = grads(model, mse), grads(model, vel), grads(model, ce)
        gcoord = gm + gv
        # 출력층은 마지막에 등록되어 있어 벡터의 끝 fc개다(몸통 → 머리 순서, 버퍼 제외)
        rows.append({"mse": mse.item(), "vel": vel.item(), "ce": ce.item(), "ceRaw": ce.item() / train.BIN_WEIGHT,
                     "gradMse": gm.norm().item(), "gradVel": gv.norm().item(), "gradCe": gc.norm().item(), "gradCoord": gcoord.norm().item(),
                     "fcShareCoord": (gcoord[-fc:].norm() / gcoord.norm()).item(), "fcShareCe": (gc[-fc:].norm() / gc.norm()).item(),
                     "cos": torch.nn.functional.cosine_similarity(gcoord, gc, dim=0).item()})
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def main():
    run = common.ARGS.run
    folder = os.path.join(common.ARGS.data, "processed", "training", run)
    m = json.load(open(os.path.join(folder, "metrics.json"), encoding="utf-8"))
    ds, _ = dataset.build(common.ARGS.data, m["sessions"], bins=dataset.BINS, standardized=True, gt=m["gt"], row_norm=m["rowNorm"], channels=m["channels"])
    (tr, va, te), _, _, _, _ = train.split_frames(ds, m["mode"], Namespace(offset=m["split"]["offset"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, y = torch.as_tensor(ds["x"], device=device), torch.as_tensor(ds["y"], device=device)
    n = len(ds["y"])
    angles = torch.as_tensor(kinematics.ik(ds["y"].reshape(-1, 20, 3)).reshape(n, kinematics.ANGLES), dtype=torch.float32, device=device)
    template = kinematics.fit_template(ds["y"][tr].reshape(-1, 20, 3), bins=train.HEAD_BINS)
    rng = np.random.default_rng(0)
    pieces = train.batch_pieces(train.segments(ds["frame_ns"], ds["session"], tr), rng)[:BATCHES]
    res = {"run": run, "frames": {"train": int(len(tr))}, "batches": len(pieces)}
    train.seed_all(0)
    fresh = train.make_model(60, in_chans=int(x.shape[1]), head="bins", template=template).to(device)
    res["init"] = measure(fresh, x, y, angles, pieces, device)
    saved = torch.load(os.path.join(folder, "model.pt"), map_location=device, weights_only=False)
    model = train.make_model(60, in_chans=int(x.shape[1]), head="bins", template=template).to(device)
    model.load_state_dict(saved.get("state_dict", saved.get("model", saved)))
    res["trained"] = measure(model, x, y, angles, pieces, device)
    # 그림용 예: 시험 프레임에서 검지 PIP 굽힘(뼈 4의 굽힘 = 자리 8) 칸 분포 — 오차가 중앙값쯤인 프레임 하나
    model.eval()
    i = torch.as_tensor(te, device=device)
    chunks = []
    with torch.no_grad():
        for a in range(0, len(i), train.EVAL_BLOCK):
            model(x[i[a: a + train.EVAL_BLOCK]])
            chunks.append(model.logits.clone())
    logits = torch.cat(chunks)
    k = 8
    prob = torch.softmax(logits, dim=-1)
    centers = model.centers
    expect = (prob * centers).sum(-1)
    width = centers[:, 1] - centers[:, 0]
    spread = torch.sqrt((prob * (centers - expect[..., None]) ** 2).sum(-1))
    err = (expect[:, k] - angles[i][:, k]).abs()
    pick = int(err.argsort()[len(err) // 2])
    q = kinematics.soft_labels(angles[i][pick: pick + 1], centers)[0, k]
    res["example"] = {"angle": "검지 PIP 굽힘", "frame": int(te[pick]), "centersDeg": np.degrees(centers[k].cpu().numpy()).tolist(), "p": prob[pick, k].cpu().numpy().tolist(),
                      "q": q.cpu().numpy().tolist(), "expectDeg": float(np.degrees(expect[pick, k].item())), "trueDeg": float(np.degrees(angles[te[pick], k].item())),
                      "spreadDeg": float(np.degrees(spread[pick, k].item())), "binWidthDeg": float(np.degrees(width[k].item()))}
    res["spreadBinsMedian"] = float((spread / width).median().item())
    for k in ("init", "trained"):
        r = res[k]
        print(f"{k}: 값 MSE {r['mse']:.5f} · 0.1×속도 {r['vel']:.6f} · 0.01×칸 {r['ce']:.4f}(칸 원값 {r['ceRaw']:.3f}) | 기울기 크기 MSE {r['gradMse']:.4f} · 속도 {r['gradVel']:.4f}"
              f" · 칸 {r['gradCe']:.4f} · 좌표 합 {r['gradCoord']:.4f} | 출력층 몫 좌표 {r['fcShareCoord']:.2f} · 칸 {r['fcShareCe']:.2f} | 코사인 {r['cos']:.3f}")
    json.dump(res, open(common.out("loss_terms.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("저장: loss_terms.json")


if __name__ == "__main__":
    main()
