"""잠재 상태를 넘기는 순서 모델(손 소나, 2026-11-01): 창마다 FastViT 잠재 → 상태(GRU 셀) → 칸 분류 머리, JEPA 손실로 상태를 배운다.

    python -I tools/train_seq.py --design none|state|world [--jepa 0.1] [--jepa-test] [--window 1.2] [--hop 0] [--mode session|blocks|time]
                                 [--run 이름] [--data 데이터 폴더] [--out 학습 결과 폴더] [--epochs 100] [--max-minutes 20] [--seed 0] [--offset 0]
                                 [--test 세션 …] [--val 세션 …] [--shift 5] [--mask] [--gt mediapipe|hamer] [--clip 걸음] [--dim 256] [세션 …]

1.2 s 창 하나로 그 순간 자세를 내면 같은 자세가 다른 입력, 다른 자세가 같은 입력이 될 수 있다(움직임 방향, 멈춘 동안). 이전 예측 좌표를
되먹이는 AR은 틀린 좌표가 다음 입력이 되고 두 봉우리의 기댓값(골짜기)만 넘어가므로, 좌표 대신 잠재를 넘긴다(보고서
reports/hand-sonar/2026-11-01-latent-ar-jepa).
- 인코더 E: train.ARCH 몸통(분류층 없음, 특징 1024) + 선형 사영 → 잠재 z_t(--dim). 창은 train.py와 같다(가운데 60칸, 증강은 학습 묶음만).
- --design state(잠재 AR): h_t = GRU셀(z_t, h_{t−1}), h_0 = 0. 예측기 P(h_t)가 DELTAS 프레임 뒤 목표 잠재를 맞힌다 — 학습 신호로만 쓰고
  추론 고리에는 없다.
- --design world(월드 모델식): ẑ_t = P(h_{t−1})가 이번 창의 목표 잠재를 예측하고 h_t = GRU셀(z_t, ẑ_t) — 예측 잠재가 다음 걸음의 숨은 상태로
  들어가 관측 z_t로 고쳐진다(h_0은 학습하는 시작 상태). 관측이 없으면 h ≈ ẑ이므로, 예측기를 관측 없이 ROLL 걸음 굴린 예측도 목표에 맞춘다.
- 목표 잠재 z̄ = LayerNorm(Ē(x)): Ē는 E의 EMA 사본(EMA, BatchNorm 통계는 그대로 복사)이고 증강 없는 가운데 창을 본다. 기울기를 막는다.
  JEPA 손실은 예측과 z̄의 L1(차원·프레임 평균)이다.
- 좌표: h_t에 kinematics.Head(선형 → 각도마다 train.HEAD_BINS칸 → 기댓값 → 순기구학). 좌표는 읽기만 하고 되먹이지 않는다.
- 손실 = 자세(train.loss_fn: MSE + 속도) + train.BIN_WEIGHT × 칸 교차 엔트로피 + --jepa × JEPA. 자세 손실은 학습 세션 프레임에만,
  JEPA는 묶음의 모든 프레임에 준다. --jepa-test면 시험 세션의 에코도 라벨 없이 JEPA 손실에 쓴다(실제 사용에서도 워치는 새 착용의 에코를
  늘 가진다 — 시험 입력을 보므로 따로 보고한다). --jepa 0이면 상태만(GRU) 있는 비교 판이고 목표 인코더를 돌리지 않는다.
- 학습 단위: train.batch_pieces로 덩어리(train.segments)를 --clip 프레임 구간으로 잘라 train.BATCH 프레임 안팎으로 묶는다. 구간마다 상태를
  새로 시작한다. 증강(--shift·--mask)은 구간마다 같은 이동·가리기(train.Augment)다.
- 평가: 덩어리마다 처음부터 끝까지 상태를 이어 굴린다(실시간과 같다). 검증 손실(조기 종료)은 자세 손실(MSE + 속도)만이고, 멈춤 규칙·
  벽시계 상한은 train.fit과 같다.
- --design none: 상태 없이 창마다 z_t → 머리(같은 틀의 비교 판, JEPA 없음).
- 창과 갱신(2026-11-01, 사용자 "짧게도 해보고, update를 0.6초로 해서 겹침을 반으로"): --window W s면 1.2 s 창의 마지막(가장 최근) W ÷ 12.5 ms
  줄만 잘라 창마다 다시 표준화한다(증강의 시간 가리기는 그 줄 안에, 폭은 W ÷ 1.2배 — train.Augment rows). --hop H s면 상태를 H s마다 한 번 갱신한다. 덩어리마다 간격 H의 시간 격자를 놓고 격자점마다 가장 가까운
  카메라 프레임을 걸음으로 고른다. 학습은 격자 시작을 H ÷ (프레임 간격 평균)개로 밀어 위상을 모두 쓰고(거의 모든 프레임이 든다), 검증·
  평가는 덩어리 첫 프레임에서 시작하는 위상 하나만 쓴다. 이때 계기는 그 걸음 프레임에서만 잰다. --hop 0이면 카메라 프레임마다(평균 50 ms,
  1.2 s 창이면 이웃 창과 약 96% 겹침). 걸음이 듬성하면(hop > 0) JEPA 앞 걸음은 STEP_DELTAS(걸음 단위)다.
- 결과: <out>/<run>/에 dataset.npz·model.pt·metrics.json(train.py 형식 2 + seq)·report.png(--hop 0만)와 예측 pred.npy(N, 60, 평가하지 않은
  프레임은 nan)를 쓴다.
"""
import argparse
import copy
import math
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # python -I는 3.11부터 -P를 겸해 스크립트 폴더를 경로에 넣지 않는다
import hs_io  # noqa: E402  사용자 설치 numpy·torch·timm 경로도 여기서 붙는다
import dataset  # noqa: E402
import joints  # noqa: E402
import kinematics  # noqa: E402
import numpy as np  # noqa: E402
import timm  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import train  # noqa: E402
from torch import nn  # noqa: E402

DESIGNS = ("none", "state", "world")
DIM = 256  # 잠재·상태 차원
HIDDEN = 512  # 예측기 숨은 층
CLIP = 64  # 학습 구간 걸음(--hop 0이면 프레임, 20 fps면 3.2 s)
CLIP_SEC = 9.6  # --hop > 0의 기본 구간 길이(s): 걸음 수 = 이것 ÷ hop(8걸음 이상)
DELTAS = (2, 5, 10)  # state 예측기가 맞히는 앞 걸음(프레임, 약 0.1·0.25·0.5 s)
STEP_DELTAS = (1, 2)  # --hop > 0일 때의 앞 걸음(걸음 단위)
ROLL = 3  # world 예측기를 관측 없이 굴려 맞히는 걸음 수
EMA = 0.996
JEPA_WEIGHT = 0.1
ROW_SEC = 600 / 48_000  # 줄 하나 = 처프 하나(profiles.N ÷ profiles.RATE = 12.5 ms)


def crop(x, rows):
    """창 (N, C, 줄, 칸)의 마지막 rows줄(가장 최근)만 남기고 창마다 다시 표준화. rows가 창 이상이면 x 그대로."""
    if rows is None or rows >= x.shape[2]:
        return x
    return train.standardize_t(x[:, :, -rows:, :])


def hop_sequences(seg, t, hop, all_phases=True):
    """덩어리 seg(시각 순 프레임 번호)를 hop s 간격 걸음으로 고른 부분 열 목록(머리말). t는 프레임 시각(s, 전체 배열). hop ≤ 0이면 [seg]."""
    if hop <= 0 or len(seg) < 2:
        return [seg]
    tt = t[seg]
    mean_gap = (tt[-1] - tt[0]) / (len(tt) - 1)  # 카메라 간격이 33/67 ms로 갈려 중앙값은 둘 중 하나로 튄다
    phases = max(1, int(round(hop / mean_gap))) if all_phases else 1
    out = []
    for j in range(phases):
        grid = tt[0] + j * hop / phases + np.arange(int((tt[-1] - tt[0]) / hop) + 1) * hop
        grid = grid[grid <= tt[-1]]
        k = np.clip(np.searchsorted(tt, grid), 1, len(tt) - 1)
        k = np.where(np.abs(tt[k - 1] - grid) <= np.abs(tt[k] - grid), k - 1, k)  # 격자점에 가장 가까운 프레임
        out.append(seg[np.unique(k)])
    return out


def to_padded(flat, lengths):
    """이어 붙인 (Σ길이, …) → (구간 수, 최대 길이, …)와 실제 걸음 표시 (구간 수, 최대 길이) bool."""
    out = flat.new_zeros((len(lengths), max(lengths)) + tuple(flat.shape[1:]))
    mask = torch.zeros((len(lengths), max(lengths)), dtype=torch.bool, device=flat.device)
    a = 0
    for k, n in enumerate(lengths):
        out[k, :n] = flat[a: a + n]
        mask[k, :n] = True
        a += n
    return out, mask


def from_padded(padded, lengths):
    return torch.cat([padded[k, :n] for k, n in enumerate(lengths)])


def delta_pairs(lengths, deltas):
    """state JEPA 쌍: 구간 안에서 (출발 프레임, deltas[k] 뒤 목표 프레임, k)의 이어 붙인 번호 배열 셋."""
    src, dst, which = [], [], []
    a = 0
    for n in lengths:
        for k, d in enumerate(deltas):
            if n > d:
                s = np.arange(a, a + n - d)
                src.append(s)
                dst.append(s + d)
                which.append(np.full(len(s), k))
        a += n
    if not src:
        return (np.zeros(0, dtype=np.int64),) * 3
    return np.concatenate(src), np.concatenate(dst), np.concatenate(which)


class Encoder(nn.Module):
    """창 (N, C, 96, 60) → 잠재 (N, dim): 분류층 없는 timm 몸통 + 선형 사영."""

    def __init__(self, in_chans, dim):
        super().__init__()
        self.backbone = timm.create_model(train.ARCH, pretrained=False, in_chans=in_chans, num_classes=0)
        self.proj = nn.Linear(self.backbone.num_features, dim)

    def forward(self, x):
        return self.proj(self.backbone(x))


def mlp(inputs, outputs, hidden=HIDDEN):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Linear(hidden, outputs))


class SeqModel(nn.Module):
    """머리말의 설계(none·state·world). forward(창, 구간 길이) → (점 60값 (Σ, 60), {"h": 상태 (Σ, dim) — none이면 z, "zhat": world의 예측 잠재
    (Σ, dim) 또는 None})."""

    def __init__(self, design, template, in_chans=2, dim=DIM, deltas=DELTAS, encoder=None):
        super().__init__()
        if design not in DESIGNS:
            raise ValueError(f"설계 {design!r}은 {'·'.join(DESIGNS)} 중 하나여야 함")
        if template.get("centers") is None:
            raise ValueError("칸 중심이 있는 템플릿이 필요함(kinematics.fit_template bins > 0)")
        self.design, self.dim, self.deltas = design, int(dim), tuple(deltas)
        self.encoder = encoder if encoder is not None else Encoder(in_chans, dim)
        self.target = copy.deepcopy(self.encoder)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.head = kinematics.Head(nn.Linear(dim, kinematics.ANGLES * int(np.shape(template["centers"])[1])), template)
        self.num_classes = self.head.num_classes
        if design == "none":
            return
        self.cell = nn.GRUCell(dim, dim)
        if design == "state":
            self.predictor = mlp(dim, dim * len(self.deltas))
        else:
            self.predictor = mlp(dim, dim)
            self.h0 = nn.Parameter(torch.zeros(dim))

    def prior(self, h):
        """world의 예측 잠재 ẑ = LayerNorm(P(h)): 목표 z̄와 같은 공간이고 크기가 묶인다. 정규화하지 않으면 P를 거듭할수록 커져 씨앗값 1·2가
        epoch 6에 발산했다(2026-11-01, 학습한 P는 걸음마다 약 10배)."""
        p = self.predictor(h)
        return F.layer_norm(p, (p.shape[-1],))

    def roll(self, z, mask):
        """잠재 (n, L, dim)을 걸음마다 넘긴다 → 상태 (n, L, dim), world면 각 걸음에 들어간 예측 잠재 (n, L, dim)(그 밖은 None). none은 z 그대로."""
        if self.design == "none":
            return z, None
        n, steps, _ = z.shape
        h = z.new_zeros((n, self.dim)) if self.design == "state" else self.h0.expand(n, -1)
        hs, priors = [], []
        for t in range(steps):
            if self.design == "world":
                prior = self.prior(h)
                priors.append(prior)
                new = self.cell(z[:, t], prior)
            else:
                new = self.cell(z[:, t], h)
            h = torch.where(mask[:, t: t + 1], new, h)
            hs.append(h)
        return torch.stack(hs, 1), (torch.stack(priors, 1) if priors else None)

    def forward(self, x, lengths):
        z, mask = to_padded(self.encoder(x), lengths)
        hp, priors = self.roll(z, mask)
        h = from_padded(hp, lengths)
        return self.head(h), {"h": h, "zhat": None if priors is None else from_padded(priors, lengths)}

    @torch.no_grad()
    def targets(self, x):
        """목표 잠재 LayerNorm(Ē(x)) (N, dim). x는 증강 없는 창."""
        self.target.eval()
        z = self.target(x)
        return F.layer_norm(z, (z.shape[-1],))

    def jepa_loss(self, aux, target, lengths):
        """예측과 목표 잠재의 L1. state: P(h_t)의 deltas 걸음 뒤. world: 한 걸음 ẑ_t(구간 첫 걸음 제외) + 관측 없이 2…ROLL 걸음 굴린 예측."""
        h = aux["h"]
        if self.design == "state":
            src, dst, which = delta_pairs(lengths, self.deltas)
            if not len(src):
                return h.sum() * 0.0
            pred = self.predictor(h[src]).reshape(len(src), len(self.deltas), self.dim)[torch.arange(len(src), device=h.device), torch.as_tensor(which, device=h.device)]
            return F.l1_loss(pred, target[torch.as_tensor(dst, device=h.device)])
        pos = np.concatenate([np.arange(n) for n in lengths])
        span = np.repeat(np.asarray(lengths), lengths)
        terms = []
        one = np.flatnonzero(pos > 0)
        if len(one):
            i = torch.as_tensor(one, device=h.device)
            terms.append(F.l1_loss(aux["zhat"][i], target[i]))
        src = np.flatnonzero(pos + 2 < span)
        if len(src) and ROLL >= 2:
            z = self.prior(h[torch.as_tensor(src, device=h.device)])  # 한 걸음(= 다음 ẑ)
            for r in range(2, ROLL + 1):
                z = self.prior(z)  # 관측 없이 굴림: 예측 잠재를 상태로
                ok = pos[src] + r < span[src]
                if ok.any():
                    terms.append(F.l1_loss(z[torch.as_tensor(np.flatnonzero(ok), device=h.device)],
                                           target[torch.as_tensor(src[ok] + r, device=h.device)]))
        return sum(terms) / len(terms) if terms else h.sum() * 0.0

    @torch.no_grad()
    def ema(self, momentum=EMA):
        """목표 인코더 ← momentum × 목표 + (1 − momentum) × 온라인. BatchNorm 통계 같은 버퍼는 그대로 복사한다."""
        for t, o in zip(self.target.parameters(), self.encoder.parameters()):
            t.mul_(momentum).add_(o.detach(), alpha=1.0 - momentum)
        for t, o in zip(self.target.buffers(), self.encoder.buffers()):
            t.copy_(o)


def predict_seq(model, x, segs, block=train.EVAL_BLOCK, rows=None):
    """덩어리(걸음 열)마다 처음부터 상태를 이어 굴린 예측 (len(x), 60) float32 numpy(평가 모드). 걸음이 아닌 프레임은 nan. rows는 crop."""
    model.eval()
    out = np.full((len(x), model.num_classes), np.nan, dtype=np.float32)
    segs = [np.asarray(s, dtype=np.int64) for s in segs if len(s)]
    if not segs:
        return out
    idx = np.concatenate(segs)
    lengths = [len(s) for s in segs]
    with torch.no_grad():
        i = torch.as_tensor(idx, device=x.device)
        z = torch.cat([model.encoder(crop(x[i[a: a + block]], rows)) for a in range(0, len(i), block)])
        hp, _ = model.roll(*to_padded(z, lengths))
        pred = model.head(from_padded(hp, lengths))
    out[idx] = pred.float().cpu().numpy()
    return out


def eval_loss(model, x, y, segs, rows=None):
    """덩어리(걸음 열)들의 자세 손실(MSE + 속도, 평가 모드, 상태는 덩어리마다 새로)."""
    segs = [s for s in segs if len(s)]
    idx = np.concatenate(segs)
    pred = torch.as_tensor(predict_seq(model, x, segs, rows=rows)[idx], device=x.device)
    return float(train.loss_fn(pred, y[torch.as_tensor(idx, device=x.device)], torch.as_tensor(train.pairs_of(segs), device=x.device)))


def fit(x, y, angles, train_segs, val_segs, epochs, seed, design, template, jepa=JEPA_WEIGHT, extra_segs=(), augment=None, log=print,
        max_minutes=train.MAX_MINUTES, encoder=None, dim=DIM, clip=CLIP, batch=train.BATCH, rows=None, deltas=DELTAS):
    """(가장 좋던 가중치의 모델, 기록, 가장 좋던 epoch, 멈춘 까닭). x는 가운데 창(평가·목표 입력), augment를 주면 학습 묶음 입력만 그것으로.
    덩어리 인자들은 걸음 열(hop_sequences)이다. extra_segs는 자세 손실 없이 JEPA에만 쓰는 덩어리(--jepa-test). rows는 창 자르기(crop),
    deltas는 state 예측기의 앞 걸음. none은 JEPA를 쓰지 않는다. 기록은 epoch마다 {"train": 자세 손실(칸 손실 포함), "jepa", "val"}."""
    torch.manual_seed(seed)
    model = SeqModel(design, template, in_chans=int(x.shape[1]), dim=dim, deltas=deltas, encoder=encoder).to(x.device)
    if design == "none":
        jepa = 0.0
    opt = torch.optim.Adam([p for name, p in model.named_parameters() if not name.startswith("target.")], lr=train.LR)
    rng = np.random.default_rng(seed)
    labelled = np.zeros(len(x), dtype=bool)
    labelled[np.concatenate(train_segs)] = True
    segs = list(train_segs) + list(extra_segs)
    best, best_epoch, best_state, history, stopped = float("inf"), 0, None, [], "epochs"
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        pose_sum, pose_n, jepa_sum, jepa_n = 0.0, 0, 0.0, 0
        for idx, pair, lengths in train.batch_pieces(segs, rng, chunk=clip, batch=batch):
            i = torch.as_tensor(idx, device=x.device)
            inputs = crop(x[i] if augment is None else augment.batch(i, lengths, rng), rows)
            pred, aux = model(inputs, lengths)
            loss = pred.sum() * 0.0
            sel = np.flatnonzero(labelled[idx])
            if len(sel):  # 구간은 한 덩어리 안이라 통째로 학습 세션이거나 아니다
                s = torch.as_tensor(sel, device=x.device)
                near = torch.as_tensor(pair[sel[:-1]] & (np.diff(sel) == 1), device=x.device)
                pose = train.loss_fn(pred[s], y[i[s]], near) + train.BIN_WEIGHT * kinematics.bin_loss(model.head.logits[s], angles[i[s]],
                                                                                                    model.head.centers)
                loss = loss + pose
                pose_sum += pose.item() * len(sel)
                pose_n += len(sel)
            if jepa > 0:
                j = model.jepa_loss(aux, model.targets(crop(x[i], rows)), lengths)
                loss = loss + jepa * j
                jepa_sum += j.item() * len(idx)
                jepa_n += len(idx)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if jepa > 0:
                model.ema()
            if not math.isfinite(loss.item()):
                raise FloatingPointError(f"epoch {epoch}: 학습 손실이 유한하지 않음({loss.item()})")
        val = eval_loss(model, x, y, val_segs, rows=rows)
        if not math.isfinite(val):
            raise FloatingPointError(f"epoch {epoch}: 검증 손실이 유한하지 않음({val})")
        history.append({"epoch": epoch, "train": pose_sum / max(pose_n, 1), "jepa": jepa_sum / jepa_n if jepa_n else None, "val": val})
        if val < best:
            best, best_epoch = val, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0:
            jl = f", JEPA {jepa_sum / jepa_n:.4f}" if jepa_n else ""
            log(f"  epoch {epoch}: 학습 자세 손실 {pose_sum / max(pose_n, 1):.5f}{jl}, 검증 손실 {val:.5f}(최고 epoch {best_epoch})")
        if epoch - best_epoch >= train.PATIENCE:
            log(f"  epoch {epoch}: 검증 손실이 {train.PATIENCE} epoch 동안 나아지지 않아 멈춤(최고 epoch {best_epoch}, {best:.5f})")
            stopped = "patience"
            break
        if epoch < epochs and time.time() - start >= max_minutes * 60:
            log(f"  epoch {epoch}: 벽시계 상한 {max_minutes:g}분이 지나 멈춤(최고 epoch {best_epoch}, {best:.5f})")
            stopped = "time"
            break
    model.load_state_dict(best_state)
    return model, history, best_epoch, stopped


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="잠재 상태를 넘기는 순서 모델(state·world)을 학습·평가한다")
    parser.add_argument("sessions", nargs="*", help="폰 세션 이름(없으면 쓸 수 있는 세션 모두)")
    parser.add_argument("--design", choices=DESIGNS, required=True, help="state: GRU 상태를 넘김(예측기는 학습 신호만), world: 예측 잠재를 다음 걸음 숨은 상태로")
    parser.add_argument("--jepa", type=float, default=JEPA_WEIGHT, metavar="무게", help=f"JEPA 손실 무게(기본 {JEPA_WEIGHT:g}, 0이면 상태만)")
    parser.add_argument("--jepa-test", action="store_true", help="시험 세션의 에코도 라벨 없이 JEPA 손실에 쓴다(따로 보고)")
    parser.add_argument("--mode", choices=("session", "blocks", "time"), default="session")
    parser.add_argument("--run", help="실행 이름(없으면 <날짜-시각>-seq-<design>)")
    parser.add_argument("--data", default=hs_io.DEFAULT_DATA)
    parser.add_argument("--out", help="학습 결과를 둘 폴더(없으면 <data>/processed/training)")
    parser.add_argument("--epochs", type=int, default=train.EPOCHS)
    parser.add_argument("--max-minutes", type=float, default=train.MAX_MINUTES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0, help="blocks 모드 무늬 offset")
    parser.add_argument("--test", action="append", metavar="세션")
    parser.add_argument("--val", action="append", metavar="세션")
    parser.add_argument("--shift", type=int, default=0, metavar="칸", help="학습 증강: 거리 축 ±칸 이동(train.py와 같음)")
    parser.add_argument("--mask", action="store_true", help="학습 증강: 시간·거리 구간 가리기(train.py와 같음)")
    parser.add_argument("--gt", choices=dataset.GT_SOURCES, default="mediapipe")
    parser.add_argument("--clip", type=int, help=f"학습 구간 걸음(기본: --hop 0이면 {CLIP}, 아니면 {CLIP_SEC:g} s ÷ hop, 8 이상)")
    parser.add_argument("--dim", type=int, default=DIM, help=f"잠재·상태 차원(기본 {DIM})")
    parser.add_argument("--window", type=float, default=dataset.WINDOW * ROW_SEC, metavar="s", help="창 길이(s, 기본 1.2 — 1.2 s 창의 최근 줄만 남김)")
    parser.add_argument("--hop", type=float, default=0.0, metavar="s", help="상태 갱신 간격(s, 0이면 카메라 프레임마다)")
    args = parser.parse_args()
    rows = int(round(args.window / ROW_SEC))
    clip = args.clip or (CLIP if args.hop <= 0 else max(8, int(round(CLIP_SEC / args.hop))))
    deltas = DELTAS if args.hop <= 0 else STEP_DELTAS
    if not 0 <= args.shift <= dataset.BINS.start or args.jepa < 0 or clip < 2 or not 8 <= rows <= dataset.WINDOW or args.hop < 0:
        print("--shift는 0…300칸, --jepa는 0 이상, --clip은 2 이상, --window는 0.1…1.2 s, --hop은 0 이상")
        return 1
    if args.design == "none" and args.jepa > 0:
        print("--design none은 JEPA를 쓰지 않는다(무게 0으로 돌림)")
        args.jepa = 0.0
    if args.jepa_test and (args.mode != "session" or args.jepa == 0):
        print("--jepa-test는 세션 나누기와 --jepa > 0에서만")
        return 1
    wide = slice(dataset.BINS.start - args.shift, dataset.BINS.stop + args.shift)
    ds, lines = dataset.build(args.data, args.sessions, bins=wide, standardized=args.shift == 0, gt=args.gt)
    print("\n".join(lines))
    if ds is None:
        print("쓸 세션이 없음")
        return 1
    n = len(ds["y"])
    run = args.run or f"{time.strftime('%Y%m%d-%H%M%S')}-seq-{args.design}"
    folder = os.path.join(args.out or os.path.join(args.data, "processed", "training"), run)
    try:
        parts, block, split, what, lines = train.split_frames(ds, args.mode, args)
    except ValueError as e:
        print(f"나누지 못함: {e}")
        return 1
    print("\n".join(lines))
    tr, va, te = parts
    if min(len(tr), len(va), len(te)) == 0:
        print(f"나눈 자료가 비었음(학습 {len(tr)} · 검증 {len(va)} · 시험 {len(te)})")
        return 1
    mix = train.kind_mix(ds, parts)
    os.makedirs(folder, exist_ok=True)
    dataset.save(os.path.join(folder, "dataset.npz"), ds)
    train.seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    every = "카메라 프레임마다" if args.hop <= 0 else f"{args.hop:g} s마다"
    print(f"{run}: 설계 {args.design}, JEPA 무게 {args.jepa:g}{'(시험 착용 에코 포함)' if args.jepa_test else ''}, 창 {rows * ROW_SEC:.2f} s({rows}줄), 갱신 {every},"
          f" {args.mode}, 정답 {dataset.GT_NAMES[args.gt]}, 학습 {len(tr)} · 검증 {len(va)} · 시험 {len(te)}프레임, 구간 {clip}걸음, 장치 {device}")
    x = torch.as_tensor(ds["x"], device=device)
    augment = None
    if args.shift or args.mask:
        augment = train.Augment(x, args.shift, args.mask, rows=rows)  # 짧은 창이면 시간 가리기를 남길 줄 안에, 폭을 비례해 줄인다
        x = augment.center()
    y = torch.as_tensor(ds["y"], device=device)
    template = kinematics.fit_template(ds["y"][tr].reshape(-1, 20, 3), bins=train.HEAD_BINS)
    angles = torch.as_tensor(kinematics.ik(ds["y"].reshape(-1, 20, 3)).reshape(n, kinematics.ANGLES), dtype=torch.float32, device=device)
    t_sec = ds["frame_ns"] / 1e9

    def seqs(idx, all_phases=True):
        return [s for g in train.segments(ds["frame_ns"], ds["session"], idx) for s in hop_sequences(g, t_sec, args.hop, all_phases)]

    start = time.time()
    try:
        model, history, best_epoch, stopped = fit(x, y, angles, seqs(tr), seqs(va, False), args.epochs, args.seed, args.design, template, jepa=args.jepa,
                                                  extra_segs=seqs(te) if args.jepa_test else (), augment=augment, max_minutes=args.max_minutes,
                                                  dim=args.dim, clip=clip, rows=rows, deltas=deltas)
    except FloatingPointError as e:
        print(f"학습을 멈춤: {e}")
        return 1
    seconds = time.time() - start
    pred = predict_seq(model, x, seqs(np.arange(n), False), rows=rows)
    np.save(os.path.join(folder, "pred.npy"), pred)
    done = np.isfinite(pred).all(axis=1)  # 평가한 걸음 프레임(--hop 0이면 모두)
    tr_e, va_e, te_e = (v[done[v]] for v in (tr, va, te))
    filled = np.where(done[:, None], pred, 0.0)  # 계기는 걸음 프레임만 본다
    m = train.instruments(ds, filled, tr, te_e)
    yy = ds["y"].reshape(-1, 20, 3)
    mm = ds["length_mm"][ds["session"]]
    m["splitMpjpeMm"] = {k: float(np.mean(joints.frame_errors(filled.reshape(-1, 20, 3)[v], yy[v]) * mm[v])) for k, v in (("train", tr_e), ("val", va_e), ("test", te_e))}
    m.update({"format": 2, "run": run, "mode": args.mode, "gt": args.gt, "rowNorm": False, "channels": "base", "sessions": ds["sessions"].tolist(),
              "lengthMm": ds["length_mm"].tolist(), "frames": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te)), "testEvaluated": int(len(te_e))},
              "seed": args.seed, "epochs": len(history), "maxEpochs": args.epochs, "maxMinutes": args.max_minutes, "stoppedBy": stopped, "bestEpoch": best_epoch,
              "trainSeconds": seconds, "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu", "history": history, "mix": mix, "split": split,
              "augment": {"shift": args.shift, "mask": args.mask, "fraction": train.AUG_FRACTION, "maskProb": train.MASK_PROB,
                          "maskRows": list(train.MASK_ROWS), "maskBins": list(train.MASK_BINS)} if augment else None,
              "seq": {"design": args.design, "jepa": args.jepa, "jepaTest": args.jepa_test, "clip": clip, "dim": args.dim, "deltas": list(deltas),
                      "roll": ROLL, "ema": EMA, "windowSec": rows * ROW_SEC, "windowRows": rows, "hopSec": args.hop}})
    hm = train.head_metrics(ds, filled, tr, te_e, template)
    m["angles"] = hm["angles"]
    m["head"] = {"type": "bins", "bins": train.HEAD_BINS, "binWeight": train.BIN_WEIGHT, "ceilingMm": hm.get("ceilingMm")}
    torch.save({"arch": train.ARCH, "in_chans": int(x.shape[1]), "design": args.design, "dim": args.dim, "deltas": list(deltas), "head_bins": train.HEAD_BINS,
                "bins": [dataset.BINS.start, dataset.BINS.stop], "window": dataset.WINDOW, "window_rows": rows, "hop": args.hop, "mode": args.mode,
                "seed": args.seed, "epoch": best_epoch, "sessions": ds["sessions"].tolist(), "gt": args.gt, "state_dict": model.state_dict()},
               os.path.join(folder, "model.pt"))
    hs_io.write_json(os.path.join(folder, "metrics.json"), m)
    if args.hop <= 0:  # 걸음이 듬성하면 시계열 그림이 비어 그리지 않는다
        pieces, piece_labels = train.figure_pieces(ds, te, block)
        train.figure(os.path.join(folder, "report.png"), ds, pred, parts, m, f"{run}: 순서 모델 {args.design} — {what}", pieces, piece_labels)
    why = {"patience": f"검증 손실이 {train.PATIENCE} epoch 동안 나아지지 않음", "epochs": f"최대 {args.epochs} epoch", "time": f"벽시계 상한 {args.max_minutes:g}분"}[stopped]
    print("\n".join([f"{run}: {len(history)} epoch({why}), {seconds / 60:.1f}분, 최고 epoch {best_epoch}"] + train.report(m)))
    print(f"결과: {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
