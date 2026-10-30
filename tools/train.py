"""FastViT-T12로 에코 창에서 손 관절 20점을 회귀하고 평가한다(손 소나 계획 4, 학습 설계 §4·§5).

    python -I tools/train.py [--mode session|blocks|time|overfit] [--run 이름] [--data 데이터 폴더] [--out 학습 결과 폴더] [--epochs 100]
                             [--max-minutes 20] [--seed 0] [--offset 0] [--test 세션 …] [--val 세션 …] [--shift 5] [--mask] [--wrist 0.1]
                             [--save-x] [세션 …]

- 자료: dataset.build(설계 §2·§3). 결과는 <out>/<run>/(기본 <data>/processed/training/<run>/)에 dataset.npz·model.pt·metrics.json·
  report.png로 쓴다. 저장소에는 아무것도 쓰지 않는다. dataset.npz는 기본으로 창 x 없이 row·목표·라벨·입력 지문만 쓴다(--save-x면 창도).
- 모델: timm fastvit_t12(처음부터 학습, ImageNet 가중치 없음), 입력 (N, 2, 96, 60), 출력 60값(단위 손 손바닥 틀의 점 1–20).
- 손실: MSE(60값) + 0.1 × 속도 손실. 속도 손실은 묶음 안 이웃 프레임의 예측 차분과 정답 차분의 MSE다(예측 움직임 자체는 벌하지
  않는다 — 평균 쏠림을 키우지 않게). 미니배치는 연속 프레임 32개 묶음 여럿(모두 256프레임 안팎)이다. 덩어리(세션이 같고 이웃 프레임
  간격이 0.2 s 안)마다 묶음 시작을 epoch마다 무작위로 밀고, 묶음 순서를 섞는다.
- 증강(--shift·--mask, 논문 §3.6.4, 2026-10-30): 학습 미니배치에만 건다. 묶음(연속 32프레임)마다 절반 확률로 증강하고, 증강한 묶음은
  거리 축으로 −shift…+shift칸 무작위 이동(창을 칸 300−shift…359+shift로 넓게 만들어 두고 60칸을 잘라 창마다 다시 표준화 — 빈자리를 0이 아닌
  실제 이웃 칸으로 채운다. 논문은 ±5칸 ≈ 1.8 cm로, 워치를 찬 위치가 바뀌는 것을 흉내 냈다)과, 80% 확률로 시간 5–15줄·거리 5–10칸 구간
  가리기(표준화 뒤 0)를 한다. 한 묶음은 같은 이동·가리기를 쓴다(1.2 s 동안 착용 위치는 그대로이고, 속도 손실이 이웃 프레임을 견준다). 논문의
  진폭 0.95–1.05배는 창마다 표준화하므로 효과가 없어 넣지 않았다. 검증·평가는 가운데 60칸(증강 없음)이라 증강 없는 학습과 입력이 같다.
  증강을 끄면 난수 소비가 예전과 같아 같은 명령이 같은 숫자를 낸다.
- 손목 상태 다중 과제(--wrist W, 2026-10-30): 몸통은 그대로 두고 출력을 63값으로 늘려 뒤 3값이 손목 정답 u(dataset 손목 정답 — 워치
  축으로 적은 손 방향)를 맞춘다. 손실에 W × 손목 손실을 더한다. 손목 손실은 (예측 − u)를 카메라 깊이 축 d 성분과 그림 평면 성분으로 나눠
  |평면|² + DEPTH_WEIGHT × 깊이²를 정답이 있는 프레임(wrist_ok)에서 평균한다. MediaPipe 3D의 깊이는 그림 투영에 맞춘 값이라 믿기 어려워 깊이
  성분을 가볍게 본다. 손가락 계기는 앞 60값으로 그대로 재고, 시험 프레임의 손목 각 오차(모델·학습 평균 방향, 젖힘·꺾임별, 그림 평면 성분)를
  metrics.json wrist에 더한다. 끄면(기본 0) 예전과 같다.
- 최적화: Adam 1e-3. 검증 손실(같은 손실, 평가 모드)이 9 epoch 동안 나아지지 않으면 멈추고 가장 좋던 가중치로 되돌린다. 최대 --epochs
  (100)와 벽시계 상한 --max-minutes(20분, 그 epoch 뒤 멈춤)를 둔다. 학습·검증 손실이 nan·inf면 이유를 적고 끝 코드 1로 멈춘다.
  씨앗값을 고정하고 cuDNN 결정적 계산을 쓴다(benchmark 끔. 이 계획을 쓰며 잰 값: 한 걸음이 0.067 → 0.075 s).
- 나누기(--mode, 없으면 세션이 둘 이상이면 session, 하나면 blocks. 규칙은 dataset 머리말):
  session = dataset.session_split(세션 단위, 설계 §4): 시험 = 마지막 착용(워치를 다시 찬 뒤, meta rewear)의 세션 통째로, 착용이 하나뿐이면
    시각 순 마지막 세션(들)(계획 6 검토 고침 M1). 검증 = 시험이 아닌 가장 늦은 세션(세션이 둘이면 학습 세션의 끝 15%). --test·--val로 세션을
    고를 수 있다. "다른 세션·착용에서도 손 모양이 나오는가"에 답하는 나누기다. 세션이 하나면 이유를 적고 멈춘다.
  blocks = dataset.block_split(10 s 블록을 고정 무늬로 학습·검증·시험에 흩음, 역할이 바뀌는 경계 뒤 2 s 비움. --offset은 무늬를 블록 단위로 돌림).
    세 자료가 세션 전체(손가락·ASL·손목·자유)에 걸친다. 이웃 블록이 맥락을 나누고 원본 채널이 세션 안 시각을 담고 있어 같은 착용 안의
    낙관적 상한으로만 읽는다(세션 단위 나누기를 대신하지 못한다).
  time = dataset.time_split(세션마다 70/15/15, 경계마다 2 s 비움): 세션 안 시간 나누기(낙관적 상한). 안내 시간표가 고정 순서라 시험이
    늘 자유 동작 꼬리이고, 세션이 여럿이어도 세션마다 이렇게 나눈다.
  overfit = 모든 프레임으로 학습·검증·평가(외울 수 있는가).
  나누기마다 프레임 수와 동작 종류 구성(손가락 굽히기·ASL 숫자·손목·자유 동작·라벨 없음)을 출력하고 metrics.json mix·split에 남긴다.
- 평가(설계 §5, 시험 프레임의 예측·정답, mm = 단위 손 × 세션 L). 맨 윗줄은 평균 자세 대비 비(모델 MPJPE ÷ 학습 평균 자세 MPJPE)다.
  MPJPE와 움직이는 점 MPJPE, 학습 평균 자세·메도이드 기준선, 동작별·동작 종류별·나누기별 MPJPE와 함께 아래를 낸다.
  - 붕괴: 주 계기는 관절·축마다 예측을 정답에 회귀한 기울기(cov ÷ var, 움직이는 점 16 × 3축)와 상관이다. 기울기는 정답과 무관한 떨림에
    부풀지 않는다(1 = 정답만큼 움직임, 0 = 정답과 무관). σ비(예측 σ ÷ 정답 σ)는 시험 덩어리 안에서 9프레임 평활한 예측으로 잰다
    (sigmaRatioSmoothed). 평활하지 않은 σ비는 참고(sigmaRatioRaw)다: 떨림이 σ를 키워 따라가지 못하는 모델이 "덜 쏠림"으로 읽힌다.
  - 뼈 길이 부족(마디별): 평활한 예측(modelSmoothed)이 주, 평활 전(modelRaw)은 참고, 학습 평균 자세의 값(meanPose)을 옆에 둔다.
  - ASL-5: 맨 윗줄은 투영 치우침 asl5Bias다. ASL-5(숫자 5)가 아닌 시험 라벨 프레임에서 (예측 − 정답)을 "학습 평균 자세 → 학습 ASL-5 평균"
    단위 방향에 투영한 평균(mm)을 모델과 평균 자세 예측기 각각 낸다. +면 편 손 쪽이다. 평균 자세의 값이 순수한 평균 쏠림의 기준이라
    그보다 큰 양수면 평균을 넘어 편 손 쪽으로 쏠린 것이다(자유 동작뿐인 시험에서도 잴 수 있다). 옛 비율("자기 정답보다 ASL-5 평균에
    가까운 예측")은 asl5CloserReference로 참고만 남긴다: 평균 자세가 편 손 쪽에 있어 평균 쏠림과 ASL-5 쏠림을 가르지 못하고, 떨림이
    비율을 50% 쪽으로 끌어 판단을 거꾸로 낼 수 있다(검토: 시간 나누기 모델 89%로 "덜 쏠림", 투영으로는 +16.0 mm 대 평균 자세 +2.1 mm).
  - 정답 잡음 바닥: 시험 프레임의 2 s 동작 유지 구간에서 잰다(noiseFloorSource "test". 세션 나누기면 시험 세션). 시험에 유지 구간이 없으면
    (시간 나누기의 자유 동작 꼬리) 모든 프레임의 정답으로 재어 "전체 정답 유지 구간 흔들림(참고)"이라 적고 그림에 선을 긋지 않는다("all").
- metrics.json 형식 2(형식 1에서 collapse·boneShortfallPercent의 구조가 바뀌고 asl5Closer가 asl5Bias·asl5CloserReference로 나뉨,
  noiseFloorSource·mix·split·stoppedBy를 더함).
- 그림 report.png 한 장: 시험 프레임의 손끝 셋(엄지·검지·약지) 손목까지 거리(정답·예측·평균 자세. 블록 나누기는 시험 블록을 이어
  붙이고, 다른 나누기는 시험 구간 첫 세션의 처음 30 s) + MPJPE 막대(평균 자세·메도이드·FastViT, 잡음 바닥 선) + 계기 숫자 + 세션
  시간표 띠(동작 종류와 나누기 역할). 영상 프레임이나 사람 모습은 넣지 않는다.
"""
import argparse
import math
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # python -I는 3.11부터 -P를 겸해 스크립트 폴더를 경로에 넣지 않는다
import hs_io  # noqa: E402  사용자 설치 numpy·torch·timm 경로도 여기서 붙는다
import dataset  # noqa: E402
import joints  # noqa: E402
import numpy as np  # noqa: E402
import pairs  # noqa: E402
import timm  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import wrist  # noqa: E402

ARCH = "fastvit_t12"
LR = 1e-3
BATCH = 256  # 미니배치 프레임 수(안팎)
CHUNK = 32  # 연속 프레임 묶음
CHUNK_GAP_SEC = 0.2  # 이웃 프레임 간격이 이보다 길면 덩어리를 끊는다(15–30 fps 카메라 간격 33–67 ms의 3배 넘게)
VELOCITY_WEIGHT = 0.1
PATIENCE = 9
EPOCHS = 100  # 최대 epoch(설계에 없음). 이 세션 실행은 조기 종료가 19–56 epoch에서 먼저 멈췄다. 약 10세션(36k프레임)이면 epoch당 7–11 s(검토 합성 측정)라 100 epoch가 12–18분으로 계획의 "실행 하나 20분 안쪽"에 든다(200이면 25–35분)
MAX_MINUTES = 20.0  # 벽시계 상한(계획의 실행 하나 20분): 넘으면 그 epoch 뒤 멈추고 가장 좋던 가중치로 되돌린다. 이때는 같은 명령도 같은 숫자를 보장하지 않아 metrics.json stoppedBy "time"으로 남긴다
EVAL_BLOCK = 512
ASL5 = 10  # prompts.csv classId: 숫자 5(편 손)
FLOOR_MIN_FRAMES = 5  # 잡음 바닥: 프레임이 이보다 적은 유지 구간은 뺀다
EXCERPT_SEC = 30.0
FIG_TIPS = (4, 8, 16)  # 그림의 손끝: 엄지·검지·약지
SURFACE, INK, INK2, MUTED, GRID, BASE, BLUE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#2a78d6"
MODES = ("session", "blocks", "time", "overfit")
KIND_COLORS = ("#2a9d8f", "#8a6fc4", "#c2548a", "#5b5a56", GRID)  # 동작 종류 넷(dataset.KINDS) + 라벨 없음(그림의 띠)
ROLE_COLORS = (BASE, "#e3a33b", BLUE)  # 학습 · 검증 · 시험(그림의 띠, dataset.ROLES 순서)
ROLE_NAMES = ("학습", "검증", "시험")
AUG_FRACTION = 0.5  # 증강하는 묶음의 몫(논문: 원본은 그대로 두고 사본에 증강)
MASK_PROB = 0.8  # 증강한 묶음에서 가리기를 할 확률(논문)
MASK_ROWS = (5, 15)  # 가리는 시간 폭(줄, 논문)
MASK_BINS = (5, 10)  # 가리는 거리 폭(칸, 논문)
GAP_DRAW = 1.0  # 시험 블록을 이어 붙여 그릴 때 블록 사이를 비우는 s
FINGER_OUT = 60  # 손가락 목표 값 수(점 1–20 × xyz)
WRIST_OUT = 3  # 손목 목표 값 수(워치 축 손 방향)
DEPTH_WEIGHT = 0.25  # 손목 손실에서 카메라 깊이 성분의 무게(MediaPipe 깊이는 투영에서 추정한 값이라 가볍게)


def velocity_loss(pred, true, pair):
    """속도 손실: pair[k]가 참인 이웃 (k, k+1)마다 (예측 차분 − 정답 차분)²의 60값 평균, 그 쌍들의 평균. pred·true (B, 60), pair (B−1,) bool."""
    d = (pred[1:] - pred[:-1]) - (true[1:] - true[:-1])
    w = pair.to(d.dtype)
    return ((d ** 2).mean(dim=1) * w).sum() / w.sum().clamp(min=1.0)


def wrist_loss(pred, u, depth, ok):
    """손목 손실: 프레임마다 e = 예측 − u를 깊이 축 depth 성분 δ와 나머지(그림 평면)로 나눈 |e − δ·depth|² + DEPTH_WEIGHT·δ², ok 프레임 평균.
    pred·u·depth (B, 3), ok (B,) 0/1."""
    e = pred - u
    delta = (e * depth).sum(dim=1, keepdim=True)
    per = ((e - delta * depth) ** 2).sum(dim=1) + DEPTH_WEIGHT * delta[:, 0] ** 2
    return (per * ok).sum() / ok.sum().clamp(min=1.0)


def loss_fn(pred, true, pair, wrist_weight=0.0):
    """손가락 MSE + 속도 손실(앞 60값). wrist_weight > 0이면 true는 (B, 67) = 손가락 60 · u 3 · 깊이 축 3 · ok 1이고 손목 손실을 더한다."""
    f = FINGER_OUT
    loss = F.mse_loss(pred[:, :f], true[:, :f]) + VELOCITY_WEIGHT * velocity_loss(pred[:, :f], true[:, :f], pair)
    if wrist_weight > 0:
        loss = loss + wrist_weight * wrist_loss(pred[:, f: f + 3], true[:, f: f + 3], true[:, f + 3: f + 6], true[:, f + 6])
    return loss


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
    return [(idx, pair) for idx, pair, _ in batch_pieces(segs, rng, chunk, batch)]


def batch_pieces(segs, rng, chunk=CHUNK, batch=BATCH):
    """batches와 같은 미니배치에 묶음 길이 목록을 더한 [(프레임 번호, 이웃 쌍 표시, 묶음 길이들)](난수 소비도 같다)."""
    pieces = []
    for seg in segs:
        first = int(rng.integers(chunk)) if len(seg) > chunk else 0
        pieces += [p for p in np.split(seg, [c for c in range(first, len(seg), chunk) if c > 0]) if len(p)]
    order = rng.permutation(len(pieces))
    count = max(1, int(np.ceil(sum(len(p) for p in pieces) / batch)))
    out = []
    for group in np.array_split(order, count):
        chosen = [pieces[k] for k in group]
        out.append((np.concatenate(chosen), pairs_of(chosen), [len(p) for p in chosen]))
    return out


def standardize_t(a):
    """dataset.standardize의 torch 판: 마지막 두 축(줄 × 칸)마다 평균을 빼고 표준편차로 나눈다(0이면 1로)."""
    mean = a.mean(dim=(-2, -1), keepdim=True)
    std = a.std(dim=(-2, -1), keepdim=True, correction=0)
    return (a - mean) / torch.where(std > 0, std, torch.ones_like(std))


class Augment:
    """학습 미니배치 증강(머리말). x는 shift > 0이면 칸 300−shift…359+shift의 표준화 전 창 (N, 2, 96, 60 + 2·shift),
    0이면 보통 창 (N, 2, 96, 60)이다."""

    def __init__(self, x, shift, mask):
        self.x, self.shift, self.mask = x, int(shift), bool(mask)
        self.width = dataset.BINS.stop - dataset.BINS.start
        self.cols = torch.arange(self.width, device=x.device)

    def center(self, block=4096):
        """검증·평가 입력: 가운데 60칸을 창마다 표준화(증강 없는 학습의 창과 같다)."""
        if not self.shift:
            return self.x
        s = self.shift
        return torch.cat([standardize_t(self.x[a: a + block, :, :, s: s + self.width]) for a in range(0, len(self.x), block)])

    def params(self, count, rng):
        """묶음 count개의 (가운데 기준 이동 칸 (count,), 시간 가리기 [시작, 끝) (count, 2), 거리 가리기 [시작, 끝) (count, 2)). 가리지 않으면 [0, 0)."""
        on = rng.random(count) < AUG_FRACTION
        move = np.where(on, rng.integers(-self.shift, self.shift + 1, count), 0)
        rows, bins = np.zeros((count, 2), dtype=np.int64), np.zeros((count, 2), dtype=np.int64)
        if self.mask:
            hide = on & (rng.random(count) < MASK_PROB)
            tw = rng.integers(MASK_ROWS[0], MASK_ROWS[1] + 1, count)
            t0 = (rng.random(count) * (dataset.WINDOW - tw + 1)).astype(np.int64)
            bw = rng.integers(MASK_BINS[0], MASK_BINS[1] + 1, count)
            b0 = (rng.random(count) * (self.width - bw + 1)).astype(np.int64)
            rows = np.where(hide[:, None], np.stack([t0, t0 + tw], axis=1), 0)
            bins = np.where(hide[:, None], np.stack([b0, b0 + bw], axis=1), 0)
        return move, rows, bins

    def batch(self, i, lengths, rng):
        """미니배치 입력 (B, 2, 96, 60). i = 프레임 번호 텐서, lengths = 묶음 길이(batch_pieces)."""
        move, rows, bins = (np.repeat(v, lengths, axis=0) for v in self.params(len(lengths), rng))
        dev = self.x.device
        sub = self.x[i]
        if self.shift:
            cols = torch.as_tensor(move + self.shift, device=dev)[:, None] + self.cols[None, :]
            sub = standardize_t(torch.gather(sub, 3, cols[:, None, None, :].expand(-1, sub.shape[1], sub.shape[2], -1)))
        if self.mask:
            r = torch.arange(dataset.WINDOW, device=dev)[None, :]
            c = self.cols[None, :]
            rows, bins = torch.as_tensor(rows, device=dev), torch.as_tensor(bins, device=dev)
            hidden = ((r >= rows[:, :1]) & (r < rows[:, 1:]))[:, None, :, None] | ((c >= bins[:, :1]) & (c < bins[:, 1:]))[:, None, None, :]
            sub = sub.masked_fill(hidden, 0.0)
        return sub


def seed_all(seed):
    """씨앗값과 값싼 결정적 설정(cuDNN 결정적 알고리즘, benchmark 끔)."""
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU와 모든 GPU
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_model(outputs=FINGER_OUT):
    return timm.create_model(ARCH, pretrained=False, in_chans=2, num_classes=outputs)


def predict(model, x, index):
    """index 프레임의 예측 (len, 출력 수) float32 numpy(평가 모드). 출력은 손가락 60값(--wrist면 뒤에 손목 3값)."""
    model.eval()
    idx = torch.as_tensor(np.asarray(index, dtype=np.int64), device=x.device)
    with torch.no_grad():
        out = [model(x[idx[a: a + EVAL_BLOCK]]).float().cpu() for a in range(0, len(idx), EVAL_BLOCK)]
    return torch.cat(out).numpy() if out else np.zeros((0, model.num_classes), dtype=np.float32)


def eval_loss(model, x, y, segs, wrist_weight=0.0):
    """덩어리들의 손실(평가 모드, 이웃 쌍은 덩어리 안만)."""
    idx = np.concatenate(segs)
    pred = torch.as_tensor(predict(model, x, idx), device=x.device)
    return float(loss_fn(pred, y[torch.as_tensor(idx, device=x.device)], torch.as_tensor(pairs_of(segs), device=x.device), wrist_weight))


def fit(x, y, train_segs, val_segs, epochs, seed, log=print, max_minutes=MAX_MINUTES, augment=None, wrist_weight=0.0):
    """(가장 좋던 가중치의 모델, 기록, 가장 좋던 epoch, 멈춘 까닭 "patience" | "epochs" | "time"). x (N, 2, 96, 60)·y (N, 60)는 같은 장치의
    텐서. 학습·검증 손실이 유한하지 않으면(nan·inf) 그 자리에서 이유를 담은 FloatingPointError로 멈춘다. augment(Augment)를 주면 학습
    미니배치만 그것으로 만들고 검증은 x 그대로다. wrist_weight > 0이면 y는 (N, 67)(loss_fn)이고 모델 출력은 63값이다."""
    torch.manual_seed(seed)
    model = make_model(FINGER_OUT + (WRIST_OUT if wrist_weight > 0 else 0)).to(x.device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(seed)
    best, best_epoch, best_state, history, stopped = float("inf"), 0, None, [], "epochs"
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0
        for idx, pair, lengths in batch_pieces(train_segs, rng):
            i = torch.as_tensor(idx, device=x.device)
            inputs = x[i] if augment is None else augment.batch(i, lengths, rng)
            loss = loss_fn(model(inputs), y[i], torch.as_tensor(pair, device=x.device), wrist_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            value = loss.item()
            if not math.isfinite(value):
                raise FloatingPointError(f"epoch {epoch}: 학습 손실이 유한하지 않음({value}) — 창·정답에 nan·inf가 있거나 학습이 발산했다")
            total += value * len(idx)
            count += len(idx)
        val = eval_loss(model, x, y, val_segs, wrist_weight)
        if not math.isfinite(val):
            raise FloatingPointError(f"epoch {epoch}: 검증 손실이 유한하지 않음({val}) — 검증 창·정답에 nan·inf가 있거나 학습이 발산했다")
        history.append({"epoch": epoch, "train": total / count, "val": val})
        if val < best:
            best, best_epoch = val, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            log(f"  epoch {epoch}: 학습 손실 {total / count:.5f}, 검증 손실 {val:.5f}(최고 epoch {best_epoch})")
        if epoch - best_epoch >= PATIENCE:
            log(f"  epoch {epoch}: 검증 손실이 {PATIENCE} epoch 동안 나아지지 않아 멈춤(최고 epoch {best_epoch}, {best:.5f})")
            stopped = "patience"
            break
        if epoch < epochs and time.time() - start >= max_minutes * 60:
            log(f"  epoch {epoch}: 벽시계 상한 {max_minutes:g}분이 지나 멈춤(최고 epoch {best_epoch}, {best:.5f}) — 이때는 같은 명령도 같은 숫자를 보장하지 않는다")
            stopped = "time"
            break
    model.load_state_dict(best_state)
    return model, history, best_epoch, stopped


def kind_mix(ds, parts):
    """나누기별 동작 종류 프레임 수 {"train": {"finger": n, "asl": n, "wrist": n, "free": n, "none": n, "total": n}, "val": …, "test": …}.
    parts는 (학습, 검증, 시험) 프레임 번호. 종류는 라벨 붙은 프레임만 세고 나머지는 none이다."""
    kind = dataset.kind_of(ds["class_id"], ds["labelled"])
    return {name: {**{key: int((kind[idx] == k).sum()) for k, key in enumerate(dataset.KIND_KEYS)}, "total": int(len(idx))}
            for name, idx in zip(dataset.ROLES, parts)}


def mix_lines(mix):
    """kind_mix 결과를 보여 줄 줄(괄호는 그 자료 안 비율)."""
    out = ["  나누기별 동작 종류 프레임(라벨 붙은 프레임만 종류에 세고, 괄호는 그 자료 안 비율):"]
    for name, label in zip(dataset.ROLES, ROLE_NAMES):
        row, total = mix[name], max(mix[name]["total"], 1)
        cells = [f"{text} {row[key]}({row[key] / total * 100:.0f}%)" for key, text in dataset.KINDS + (("none", "라벨 없음"),)]
        out.append(f"    {label} {row['total']:>4}프레임: " + " · ".join(cells))
    return out


def block_table(ds, block, role, keep):
    """블록마다 {session, block, startSec, role, frames, kept, kinds{종류: 쓴 프레임 수}}. 시각은 그 세션의 첫 프레임부터의 s다."""
    kind = dataset.kind_of(ds["class_id"], ds["labelled"])
    rows = []
    for s in np.unique(ds["session"]):
        for b in np.unique(block[ds["session"] == s]):
            m = (ds["session"] == s) & (block == b)
            rows.append({"session": int(s), "block": int(b), "startSec": float(b * dataset.BLOCK_SEC), "role": dataset.ROLES[int(role[m][0])],
                         "frames": int(m.sum()), "kept": int((m & keep).sum()),
                         "kinds": {key: int((m & keep & (kind == k)).sum()) for k, key in enumerate(dataset.KIND_KEYS)}})
    return rows


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
    smooth = joints.smooth_within(p, segments(ds["frame_ns"], ds["session"], test))[test]  # 시험 덩어리 안에서만 떨림을 걷은 예측
    finger = np.array([next(k for k, (_, idx) in enumerate(joints.FINGERS) if j + 1 in idx) for j in joints.MOVING])

    def summary(v):
        return {"mean": float(v.mean()), "min": float(v.min()), "byFinger": {joints.FINGERS[k][0]: float(v[finger == k].mean()) for k in range(5)}}

    slope, corr = joints.regression(guess, true)
    out["collapse"] = {"slope": summary(slope), "corr": summary(corr), "sigmaRatioSmoothed": summary(joints.collapse(smooth, true)),
                       "sigmaRatioRaw": summary(joints.collapse(guess, true)), "smoothFrames": joints.SMOOTH_FRAMES}

    def segment(q):
        return dict(zip(joints.SEGMENTS, joints.by_segment(joints.bone_shortfall(q, true)).tolist()))

    out["boneShortfallPercent"] = {"modelSmoothed": segment(smooth), "modelRaw": segment(guess), "meanPose": segment(mean[None])}
    labelled, cls = ds["labelled"][test], ds["class_id"][test]
    e_model = joints.frame_errors(guess, true) * scale
    e_mean = joints.frame_errors(mean[None], true) * scale
    names = dict(zip(ds["label_ids"].tolist(), ds["label_names"].tolist()))
    out["perLabel"] = [{"classId": int(c), "name": names.get(int(c), "?"), "frames": int(m.sum()),
                        "mpjpeMm": float(e_model[m].mean()), "meanPoseMm": float(e_mean[m].mean())}
                       for c in np.unique(cls[labelled]) for m in [labelled & (cls == c)]]
    kinds = dataset.kind_of(cls, labelled)  # 동작 종류별(손가락·ASL·손목·자유) MPJPE: 시험 프레임 중 그 종류가 없으면 frames 0과 null
    out["perKind"] = {key: ({"frames": int((kinds == k).sum()), "mpjpeMm": float(e_model[kinds == k].mean()),
                             "meanPoseMm": float(e_mean[kinds == k].mean()), "ratioToMeanPose": float(e_model[kinds == k].mean() / e_mean[kinds == k].mean())}
                            if (kinds == k).any() else {"frames": 0, "mpjpeMm": None, "meanPoseMm": None, "ratioToMeanPose": None})
                      for k, (key, _) in enumerate(dataset.KINDS)}
    five = ds["labelled"][train] & (ds["class_id"][train] == ASL5)
    other = labelled & (cls != ASL5)
    if five.any() and other.any():
        template = y[train][five].mean(axis=0)
        k = other
        out["asl5Bias"] = {"modelMm": float(np.mean(joints.toward(guess[k], true[k], mean, template) * scale[k])),
                           "meanPoseMm": float(np.mean(joints.toward(mean[None], true[k], mean, template) * scale[k])),
                           "meanToTemplateMm": float(np.linalg.norm(template - mean) * np.mean(scale[k])), "frames": int(k.sum())}
        out["asl5CloserReference"] = {"model": joints.closer_to(guess[k], true[k], template),
                                      "meanPose": joints.closer_to(np.broadcast_to(mean, true[k].shape), true[k], template), "frames": int(k.sum())}
    hold = ds["labelled"] & (ds["class_id"] >= 0)  # 2 s 동작(자유 동작 −1·일시정지 −2 아님)의 경계 0.5 s 뒤 유지 구간
    in_test = np.zeros(len(y), dtype=bool)
    in_test[test] = True
    for source, mask in (("test", hold & in_test), ("all", hold)):  # 시험에 유지 구간이 없으면 전체 정답의 흔들림(참고)
        groups = ds["session"][mask].astype(np.int64) * 1_000_000 + ds["prompt"][mask]
        floor = joints.hold_floor(y[mask], groups, FLOOR_MIN_FRAMES) * mm[mask]
        if np.isfinite(floor).any():
            break
    out["noiseFloorMm"] = float(np.nanmean(floor)) if np.isfinite(floor).any() else None
    out["noiseFloorFrames"] = int(np.isfinite(floor).sum())
    out["noiseFloorSource"] = source if np.isfinite(floor).any() else None
    return out


def broken(t, v, gap_sec=CHUNK_GAP_SEC):
    """시각 간격이 gap_sec를 넘는 곳에 nan을 끼워 선이 끊기게 한다."""
    k = np.flatnonzero(np.diff(t) > gap_sec) + 1
    return np.insert(t.astype(np.float64), k, np.nan), np.insert(v.astype(np.float64), k, np.nan)


def runs(t, v, gap=0.5, pad=0.07):
    """시각 순서 t(s)와 값 v에서 값이 같고 이웃 간격이 gap(s) 안인 연속 구간 [(시작, 길이, 값)]. 길이는 pad(s)만큼 늘려 한 프레임짜리도 보이게 한다."""
    t = np.asarray(t, dtype=np.float64)
    v = np.asarray(v)
    if len(t) == 0:
        return []
    cut = np.flatnonzero((np.diff(v) != 0) | (np.diff(t) > gap)) + 1
    return [(float(t[a]), float(t[b - 1] - t[a]) + pad, int(v[a])) for a, b in zip(np.r_[0, cut], np.r_[cut, len(t)])]


def timeline_strip(ax, ds, parts, session):
    """세션 시간표 띠 둘(그 세션 첫 프레임부터의 s): 위는 동작 종류(모든 프레임. 쉬는 때는 빈 회색), 아래는 나누기 역할(학습·검증·시험).
    역할 띠의 빈 곳은 쓰지 않은 프레임(역할이 바뀐 경계 뒤 비움·정답 거르기)이다. 겹치면 시험이 이긴다(과적합 점검은 모두 시험)."""
    from matplotlib.patches import Patch
    idx = np.flatnonzero(ds["session"] == session)
    idx = idx[np.argsort(ds["frame_ns"][idx], kind="stable")]
    t = (ds["frame_ns"][idx] - ds["frame_ns"][idx[0]]) / 1e9
    has = (ds["class_id"] != pairs.NO_CLASS) & (ds["class_id"] != pairs.PAUSE_CLASS)  # 동작 안에 든 프레임(경계 0.5 s도 그 동작 종류로 친다)
    role = np.full(len(ds["y"]), -1)
    for r, part in enumerate(parts):
        role[part] = r
    for y0, values, colors in ((1.2, dataset.kind_of(ds["class_id"], has)[idx], KIND_COLORS), (0.0, role[idx], ROLE_COLORS)):
        for k, color in enumerate(colors):
            ax.broken_barh([(a, w) for a, w, v in runs(t, values) if v == k], (y0, 1.0), facecolors=color, edgecolor=SURFACE, linewidth=0.8)
    for a, w, v in runs(t, dataset.kind_of(ds["class_id"], has)[idx]):  # 대비가 3:1 아래인 칸이 있어 종류 이름을 띠에 직접 쓴다
        if v < 4 and w >= 8.0:
            ax.text(a + w / 2, 1.7, dataset.KINDS[v][1], ha="center", va="center", fontsize=8.5, color=INK)
    ax.set_xlim(0, t[-1] + 0.5)
    ax.set_ylim(-0.1, 2.3)
    ax.set_yticks([0.5, 1.7])
    ax.set_yticklabels(["나누기", "동작 종류"])
    ax.set_xticks(np.arange(0, t[-1] + 1, 10))
    ax.set_xlabel(f"세션 {ds['sessions'][session]} 시각(s, 첫 프레임부터. 나누기 띠의 빈 곳 = 경계 뒤 비운 2 s·거른 프레임)")
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    handles = [Patch(facecolor=c, label=n) for c, n in zip(KIND_COLORS, [k for _, k in dataset.KINDS] + ["라벨 없음·쉬는 때"])]
    handles += [Patch(facecolor=c, label=n) for c, n in zip(ROLE_COLORS, ROLE_NAMES)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.55), ncol=8, fontsize=8.5, frameon=False, handlelength=1.2, columnspacing=1.2)


def figure(path, ds, pred, parts, m, title, pieces, labels):
    """report.png 한 장. 왼쪽 위 세 줄은 시험 프레임 덩어리(pieces)의 손끝 시계열(덩어리가 여럿이면 이어 붙임), 오른쪽은 MPJPE 막대와 계기 숫자,
    맨 아래는 세션 시간표 띠(동작 종류와 나누기 역할). 영상 프레임이나 사람 모습은 넣지 않는다."""
    train, val, test = parts
    plt = hs_io.pyplot()
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": BASE,
                         "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 9, "text.color": INK})
    y = ds["y"].reshape(-1, 20, 3)
    p = np.asarray(pred).reshape(-1, 20, 3)
    mean = joints.mean_pose(y[train])
    session = int(ds["session"][pieces[0][0]])
    scale = float(ds["length_mm"][session])
    xs, spans, x0 = [], [], 0.0  # 덩어리마다 시작 시각을 0으로 두고 GAP_DRAW만큼 띄워 이어 붙인 가로 자리
    for piece in pieces:
        tt = (ds["frame_ns"][piece] - ds["frame_ns"][piece[0]]) / 1e9
        xs.append(tt + x0)
        spans.append((x0, x0 + float(tt[-1])))
        x0 += float(tt[-1]) + GAP_DRAW
    many = len(pieces) > 1
    fig = plt.figure(figsize=(13.5, 8.9), constrained_layout=True)
    grid = fig.add_gridspec(4, 2, width_ratios=[2.2, 1.0], height_ratios=[1, 1, 1, 0.62])
    names = {tip: name for name, idx in joints.FINGERS for tip in idx[-1:]}
    first = None
    for row, tip in enumerate(FIG_TIPS):
        ax = fig.add_subplot(grid[row, 0], sharex=first)
        first = first or ax
        ax.axhline(float(np.linalg.norm(mean[tip - 1])) * scale, color=MUTED, lw=1.0, ls="--", label="평균 자세(학습)")
        for k, (piece, x) in enumerate(zip(pieces, xs)):
            for q, color, label in ((y, INK, "정답(MediaPipe)"), (p, BLUE, "FastViT 예측")):
                ax.plot(*broken(x, np.linalg.norm(q[piece, tip - 1], axis=1) * scale), color=color, lw=1.3, label=label if k == 0 else None)
        for (_, a), (b, _) in zip(spans[:-1], spans[1:]):  # 덩어리 사이 구분선
            ax.axvline((a + b) / 2, color=BASE, lw=1.0, ls=":")
        ax.set_ylabel(f"{names[tip]} 끝–손목(mm)")
        ax.grid(axis="y", color=GRID, lw=0.7)
        ax.set_xlim(-0.3, spans[-1][1] + 0.3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if row == 0:
            ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, fontsize=8.5, frameon=False)  # 자료를 가리지 않게 그림 위쪽 바깥에 둔다
            head = (f"시험 블록 {len(pieces)}개 이어 붙임(단위 손 × L {scale:.1f} mm)" if many
                    else f"시험 구간 처음 {spans[0][1]:.0f} s(단위 손 × L {scale:.1f} mm)")
            ax.set_title(head, loc="left", fontsize=10, color=INK)
        if row < len(FIG_TIPS) - 1:
            ax.tick_params(labelbottom=False)
    if many:
        ax.set_xticks([(a + b) / 2 for a, b in spans])
        ax.set_xticklabels(labels, fontsize=8.5, color=INK2)
    else:
        ax.set_xlabel("시험 구간 시작부터의 시간(s)")
    ax = fig.add_subplot(grid[0, 1])
    bar_names = ["평균 자세", "메도이드", "FastViT"]
    values = [m["baselines"]["meanPoseMm"], m["baselines"]["medoidMm"], m["mpjpeMm"]]
    bars = ax.bar(bar_names, values, color=[BASE, BASE, BLUE], width=0.6)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9, color=INK)
    if m["noiseFloorSource"] == "test":
        ax.axhline(m["noiseFloorMm"], color=INK2, lw=1.0, ls=":")
        ax.set_title(f"점선 = 시험 정답 잡음 바닥 {m['noiseFloorMm']:.1f} mm", loc="left", fontsize=9, color=INK2)
    elif m["noiseFloorSource"] == "all":  # 시험과 다른 자료라 바닥 선으로 긋지 않는다
        ax.set_title(f"시험에 2 s 동작이 없어 바닥 선 없음(전체 정답 흔들림 {m['noiseFloorMm']:.1f} mm, 참고)", loc="left", fontsize=8.5, color=INK2)
    ax.set_ylabel("시험 MPJPE(mm)")
    ax.set_ylim(0, max(values) * 1.18)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax = fig.add_subplot(grid[1:3, 1])
    ax.axis("off")
    c, s = m["collapse"], m["boneShortfallPercent"]
    text = [f"평균 자세 대비 비  {m['ratioToMeanPose']:.2f}",
            f"MPJPE {m['mpjpeMm']:.1f} mm, 움직이는 점 {m['movingMpjpeMm']:.1f} mm(평균 자세 {m['baselines']['meanPoseMovingMm']:.1f})",
            f"붕괴(움직이는 점): 회귀 기울기 {c['slope']['mean']:.2f} · 상관 {c['corr']['mean']:.2f}",
            f"  σ비 평활({c['smoothFrames']}프레임) {c['sigmaRatioSmoothed']['mean']:.2f} · 평활 전(참고) {c['sigmaRatioRaw']['mean']:.2f}",
            "  기울기 " + " · ".join(f"{k} {v:.2f}" for k, v in c["slope"]["byFinger"].items()),
            "뼈 길이 부족(%) 손바닥 · 첫 · 가운데 · 끝 마디",
            "  FastViT 평활 " + " · ".join(f"{v:.1f}" for v in s["modelSmoothed"].values()),
            "  평활 전(참고) " + " · ".join(f"{v:.1f}" for v in s["modelRaw"].values()),
            "  평균 자세 " + " · ".join(f"{v:.1f}" for v in s["meanPose"].values())]
    if "asl5Bias" in m:
        a = m["asl5Bias"]
        text += ["편 손(ASL-5) 쪽 치우침(평균 자세 → ASL-5 평균 방향)",
                 f"  FastViT {a['modelMm']:+.1f} mm · 평균 자세 {a['meanPoseMm']:+.1f} mm({a['frames']}프레임)"]
    cells = [f"{name} {m['perKind'][key]['mpjpeMm']:.1f}({m['perKind'][key]['meanPoseMm']:.1f})" if m["perKind"][key]["frames"] else f"{name} 없음"
             for key, name in dataset.KINDS]
    text += ["동작 종류별 시험 MPJPE(mm), FastViT(평균 자세)", "  " + " · ".join(cells[:2]), "  " + " · ".join(cells[2:])]
    text.append(f"학습 {m['frames']['train']} · 검증 {m['frames']['val']} · 시험 {m['frames']['test']}프레임, epoch {m['bestEpoch']}/{m['epochs']}")
    step = min(0.085, 0.97 / len(text))
    for k, line in enumerate(text):
        ax.text(0.0, 1.0 - k * step, line, transform=ax.transAxes, va="top", fontsize=11 if k == 0 else 9,
                fontweight="bold" if k == 0 else "normal", color=INK if k == 0 else INK2)
    timeline_strip(fig.add_subplot(grid[3, :]), ds, parts, session)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def figure_pieces(ds, test, block=None):
    """그림에 그릴 시험 프레임 덩어리 [(프레임 번호, 시각 순서)]와 이름. block (N,)이 없으면 시험 구간 첫 세션의 처음 EXCERPT_SEC s 하나,
    있으면(블록 나누기) 시험 블록마다 하나다(그 블록의 쓴 프레임 전부)."""
    test = np.asarray(test)
    order = test[np.lexsort((ds["frame_ns"][test], ds["session"][test]))]
    order = order[ds["session"][order] == ds["session"][order[0]]]  # 시험 구간의 첫 세션
    if block is None:
        t = (ds["frame_ns"][order] - ds["frame_ns"][order[0]]) / 1e9
        return [order[t <= EXCERPT_SEC]], [None]
    kind = dataset.kind_of(ds["class_id"], ds["labelled"])
    pieces, labels = [], []
    for b in np.unique(block[order]):
        piece = order[block[order] == b]
        count = np.bincount(kind[piece], minlength=5)[:4]
        main = [dataset.KINDS[k][1] for k in np.argsort(-count, kind="stable") if count[k] > 0 and count[k] >= 0.2 * count.sum()]
        labels.append(f"블록 {b} · {b * dataset.BLOCK_SEC:.0f}–{(b + 1) * dataset.BLOCK_SEC:.0f} s\n" + (" + ".join(main) or "라벨 없음"))
        pieces.append(piece)
    return pieces, labels


def wrist_metrics(ds, pred, train, test):
    """시험 프레임의 손목 각 오차(도). pred (N, 3)는 모델의 손목 출력이고 정답이 있는 프레임(wrist_ok)만 잰다. 기준선은 학습 프레임 정답의
    평균 방향(평균 자세에 해당). angle = 3D 방향 각, imagePlane = 카메라 깊이 축에 수직인 면(그림 평면)에 투영한 방향의 각(믿을 만한 성분),
    extension·deviation = 손등 쪽 젖힘·새끼 쪽 꺾임 각의 절대 오차, slope = 예측 각을 정답 각에 회귀한 기울기(1 따라감, 0 평균에 붙음)."""
    ok = ds["wrist_ok"]
    tr, te = train[ok[train]], test[ok[test]]
    out = {"frames": {"train": int(len(tr)), "test": int(len(te))}}
    if len(tr) == 0 or len(te) == 0:
        return out

    def unit(v):
        return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None)

    def angle(a, b):
        return np.degrees(np.arccos(np.clip((a * b).sum(axis=1), -1.0, 1.0)))

    def plane(v, d):
        return unit(v - (v * d).sum(axis=1, keepdims=True) * d)

    def stats(v):
        return {"mean": float(np.mean(v)), "median": float(np.median(v))}

    def slope(p, t):
        var = float(np.var(t))
        return float(np.mean((p - p.mean()) * (t - t.mean())) / var) if var > 0 else None

    u = ds["wrist"].astype(np.float64)
    d = ds["wrist_depth"].astype(np.float64)[te]
    t = u[te]
    p = unit(pred[te].astype(np.float64))
    base = np.repeat(unit(u[tr].mean(axis=0, keepdims=True)), len(te), axis=0)
    (ext_t, dev_t), (ext_p, dev_p), (ext_b, dev_b) = (wrist.extension_deviation(v) for v in (t, p, base))
    out.update({
        "angleDeg": {"model": stats(angle(p, t)), "meanDirection": stats(angle(base, t))},
        "imagePlaneAngleDeg": {"model": stats(angle(plane(p, d), plane(t, d))), "meanDirection": stats(angle(plane(base, d), plane(t, d)))},
        "extensionAbsErrDeg": {"model": stats(np.abs(ext_p - ext_t)), "meanDirection": stats(np.abs(ext_b - ext_t))},
        "deviationAbsErrDeg": {"model": stats(np.abs(dev_p - dev_t)), "meanDirection": stats(np.abs(dev_b - dev_t))},
        "slope": {"extension": slope(ext_p, ext_t), "deviation": slope(dev_p, dev_t)},
        "trueSpreadDeg": {"extension": float(np.std(ext_t)), "deviation": float(np.std(dev_t))},
    })
    return out


def report(m):
    """출력할 요약 줄."""
    b, c, s = m["baselines"], m["collapse"], m["boneShortfallPercent"]
    lines = [f"평균 자세 대비 비 {m['ratioToMeanPose']:.3f}: FastViT {m['mpjpeMm']:.2f} mm / 평균 자세 {b['meanPoseMm']:.2f} mm(메도이드 {b['medoidMm']:.2f} mm)",
             f"  움직이는 점 MPJPE {m['movingMpjpeMm']:.2f} mm(평균 자세 {b['meanPoseMovingMm']:.2f}, 메도이드 {b['medoidMovingMm']:.2f})",
             f"  붕괴(움직이는 점 16 × 3축): 회귀 기울기 평균 {c['slope']['mean']:.3f}(최소 {c['slope']['min']:.3f}), 상관 평균 {c['corr']['mean']:.3f},"
             " 손가락별 기울기 " + ", ".join(f"{k} {v:.2f}" for k, v in c["slope"]["byFinger"].items()),
             f"    σ비 평활({c['smoothFrames']}프레임) 평균 {c['sigmaRatioSmoothed']['mean']:.3f}(최소 {c['sigmaRatioSmoothed']['min']:.3f}),"
             f" 평활 전(참고, 떨림이 부풀림) {c['sigmaRatioRaw']['mean']:.3f}",
             "  뼈 길이 부족 % FastViT 평활 " + ", ".join(f"{k} {v:.1f}" for k, v in s["modelSmoothed"].items()),
             "                 평활 전(참고) " + ", ".join(f"{k} {v:.1f}" for k, v in s["modelRaw"].items()),
             "                 평균 자세 " + ", ".join(f"{k} {v:.1f}" for k, v in s["meanPose"].items())]
    if "asl5Bias" in m:
        a, r = m["asl5Bias"], m["asl5CloserReference"]
        lines += [f"  편 손(ASL-5) 쪽 치우침(ASL-5가 아닌 시험 라벨 {a['frames']}프레임, 평균 자세 → ASL-5 평균 방향 투영): FastViT {a['modelMm']:+.2f} mm,"
                  f" 평균 자세 {a['meanPoseMm']:+.2f} mm(두 평균 사이 {a['meanToTemplateMm']:.1f} mm)",
                  f"    참고(옛 계기): 정답보다 ASL-5 평균에 가까운 예측 {r['model'] * 100:.1f}%(평균 자세 {r['meanPose'] * 100:.1f}%)"]
    if m["noiseFloorSource"] == "test":
        lines.append(f"  정답 잡음 바닥 {m['noiseFloorMm']:.2f} mm(시험 유지 구간 {m['noiseFloorFrames']}프레임)")
    elif m["noiseFloorSource"] == "all":
        lines.append(f"  전체 정답 유지 구간 흔들림(참고) {m['noiseFloorMm']:.2f} mm({m['noiseFloorFrames']}프레임) — 시험에 2 s 동작이 없어 바닥으로 쓰지 않는다")
    lines.append("  동작별 MPJPE(mm, 평균 자세): " + ", ".join(f"{r['name']} {r['mpjpeMm']:.1f}({r['meanPoseMm']:.1f}, {r['frames']})" for r in m["perLabel"]))
    lines.append("  동작 종류별 MPJPE(mm, 평균 자세, 비, 프레임): " + ", ".join(
        f"{text} {k['mpjpeMm']:.1f}({k['meanPoseMm']:.1f}, {k['ratioToMeanPose']:.2f}, {k['frames']})" if k["frames"] else f"{text} 시험에 없음"
        for key, text in dataset.KINDS for k in [m["perKind"][key]]))
    lines.append(f"  나누기별 MPJPE: " + ", ".join(f"{k} {v:.2f} mm" for k, v in m["splitMpjpeMm"].items()))
    w = m.get("wrist")
    if w and "angleDeg" in w:
        a, g, e, v, k = w["angleDeg"], w["imagePlaneAngleDeg"], w["extensionAbsErrDeg"], w["deviationAbsErrDeg"], w["slope"]

        def fmt(x):
            return "—" if x is None else f"{x:.2f}"

        lines += [f"  손목(무게 {w['weight']:g}, 시험 {w['frames']['test']}프레임): 방향 각 오차 중앙 {a['model']['median']:.1f}°"
                  f"(평균 방향 {a['meanDirection']['median']:.1f}°), 그림 평면 {g['model']['median']:.1f}°({g['meanDirection']['median']:.1f}°)",
                  f"    젖힘 오차 {e['model']['median']:.1f}°({e['meanDirection']['median']:.1f}°, 기울기 {fmt(k['extension'])}) ·"
                  f" 꺾임 오차 {v['model']['median']:.1f}°({v['meanDirection']['median']:.1f}°, 기울기 {fmt(k['deviation'])})"
                  f" · 정답 흩어짐 젖힘 {w['trueSpreadDeg']['extension']:.1f}° 꺾임 {w['trueSpreadDeg']['deviation']:.1f}°"]
    elif w:
        lines.append(f"  손목: 정답이 있는 프레임이 없음(학습 {w['frames']['train']} · 시험 {w['frames']['test']})")
    return lines


def split_frames(ds, mode, args):
    """나누기: ((학습, 검증, 시험) 프레임 번호, 블록 번호 (N,) 또는 None, metrics.json split, 그림 제목의 설명, 출력 줄). 나눌 수 없으면 ValueError."""
    n = len(ds["y"])
    block, split, lines = None, {}, []
    if mode == "overfit":
        return (np.arange(n),) * 3, None, split, "과적합 점검(모든 프레임으로 학습·평가)", lines
    if mode == "time":
        if len(ds["sessions"]) > 1:
            lines.append("세션이 여럿인데 time: 세션마다 70/15/15라 시험이 모두 시간표 끝(자유 동작)이다 — 세션 단위로 견주려면 --mode session")
        split = {"fractions": list(dataset.SPLIT), "gapSec": dataset.GAP_SEC}
        return (dataset.time_split(ds["frame_ns"], ds["session"]), None, split,
                "세션 안 시간 나누기(낙관적 상한: 앞 70% 학습 · 15% 검증 · 끝 15% 시험, 경계 2 s 비움)", lines)
    if mode == "blocks":
        block, role, keep = dataset.block_assign(ds["frame_ns"], ds["session"], offset=args.offset)
        train, val, test = dataset.block_split(ds["frame_ns"], ds["session"], offset=args.offset)
        table = block_table(ds, block, role, keep)
        dropped = {name: int(((role == r) & ~keep).sum()) for r, name in enumerate(dataset.ROLES)}
        used = len(train) + len(val) + len(test)
        lines += [f"블록 나누기: {dataset.BLOCK_SEC:g} s 블록 {len(table)}개, 무늬 offset {args.offset}(블록 번호 mod {dataset.BLOCK_PERIOD},"
                  f" 시험 자리 {list(dataset.BLOCK_TEST)} · 검증 자리 {list(dataset.BLOCK_VAL)}), 역할이 바뀐 경계 뒤 {dataset.GAP_SEC:g} s 비움",
                  "  블록별 역할(학=학습 검=검증 시=시험): " + " ".join(f"{r['block']}{ROLE_NAMES[dataset.ROLES.index(r['role'])][0]}" for r in table),
                  f"  쓴 프레임 학습 {len(train)} · 검증 {len(val)} · 시험 {len(test)}(전체 {n}, 비운 프레임 학습 {dropped['train']} · 검증 {dropped['val']}"
                  f" · 시험 {dropped['test']}), 쓴 프레임 중 비율 {len(train) / used:.0%}/{len(val) / used:.0%}/{len(test) / used:.0%}",
                  "  시험 블록: " + ", ".join(f"{r['block']}번({r['startSec']:.0f}–{r['startSec'] + dataset.BLOCK_SEC:.0f} s)" for r in table if r["role"] == "test")
                  + " / 검증 블록: " + ", ".join(f"{r['block']}번({r['startSec']:.0f}–{r['startSec'] + dataset.BLOCK_SEC:.0f} s)" for r in table if r["role"] == "val")]
        split = {"blockSec": dataset.BLOCK_SEC, "period": dataset.BLOCK_PERIOD, "offset": args.offset, "testSlots": list(dataset.BLOCK_TEST),
                 "valSlots": list(dataset.BLOCK_VAL), "gapSec": dataset.GAP_SEC, "droppedFrames": dropped, "blocks": table}
        what = f"블록 나누기(같은 착용 안 낙관적 상한, 10 s 블록 학습 14 · 검증 3 · 시험 3자리, offset {args.offset}, 경계 뒤 2 s 비움)"
        return (train, val, test), block, split, what, lines
    names = ds["sessions"].tolist()

    def pick(given):
        if not given:
            return None
        missing = [g for g in given if g not in names]
        if missing:
            raise ValueError(f"자료에 없는 세션 {', '.join(missing)}(쓴 세션: {', '.join(names)})")
        return [names.index(g) for g in given]

    parts, notes, sets = dataset.session_split(ds["frame_ns"], ds["session"], ds["wall_ms"], ds["rewear"], test=pick(args.test), val=pick(args.val),
                                               wear=ds.get("wear"))
    by_name = [[names[s] for s in group] for group in sets]
    lines += ["세션 나누기: " + " · ".join(f"{label} {', '.join(group) or '없음'}" for label, group in zip(ROLE_NAMES, by_name))]
    lines += ["  " + note for note in notes]
    split = {"train": by_name[0], "val": by_name[1], "test": by_name[2], "valFromTrainTail": not sets[1], "notes": notes}
    what = f"세션 나누기(학습 {len(sets[0])} · 검증 {len(sets[1]) if sets[1] else '학습 세션 끝 15%'} · 시험 {len(sets[2])}세션)"
    return parts, None, split, what, lines


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="FastViT-T12로 에코 창에서 손 관절 20점을 학습·평가한다")
    parser.add_argument("sessions", nargs="*", help="폰 세션 이름(없으면 쓸 수 있는 세션 모두)")
    parser.add_argument("--mode", choices=MODES,
                        help="session: 세션 단위, blocks: 10 s 블록을 흩어 나누기, time: 세션 안 시간 나누기, overfit: 모든 프레임으로 학습·평가"
                             "(없으면 세션이 둘 이상이면 session, 하나면 blocks)")
    parser.add_argument("--run", help="실행 이름(없으면 <날짜-시각>-<mode>)")
    parser.add_argument("--data", default=hs_io.DEFAULT_DATA)
    parser.add_argument("--out", help="학습 결과를 둘 폴더(없으면 <data>/processed/training)")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="최대 epoch")
    parser.add_argument("--max-minutes", type=float, default=MAX_MINUTES, help="학습 벽시계 상한(분). 넘으면 그 epoch 뒤 멈춘다")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0, help="blocks 모드: 고정 무늬(블록 번호 mod 20)를 블록 단위로 돌리는 칸 수")
    parser.add_argument("--test", action="append", metavar="세션", help="session 모드: 시험 세션(여러 번 줄 수 있음. 없으면 마지막 착용 묶음 전체 — '워치 다시 참' 표시 세션부터 다음 표시 전까지)")
    parser.add_argument("--val", action="append", metavar="세션", help="session 모드: 검증 세션(여러 번 줄 수 있음. 없으면 시험이 아닌 가장 늦은 세션)")
    parser.add_argument("--shift", type=int, default=0, metavar="칸", help="학습 증강: 거리 축 ±칸 무작위 이동(논문 5칸 ≈ 1.8 cm, 0이면 끔)")
    parser.add_argument("--mask", action="store_true", help="학습 증강: 시간 5–15줄·거리 5–10칸 구간 가리기(논문)")
    parser.add_argument("--wrist", type=float, default=0.0, metavar="무게",
                        help="손목 상태 다중 과제: 워치 기준 손 방향 3값을 함께 학습하고 손실에 이 무게로 더한다(0이면 끔, 예 0.1)")
    parser.add_argument("--save-x", action="store_true", help="dataset.npz에 창 x도 쓴다(세션당 약 170 MB. 기본은 row·목표·라벨·입력 지문만)")
    args = parser.parse_args()
    if not 0 <= args.shift <= dataset.BINS.start:
        print(f"--shift는 0…{dataset.BINS.start}칸")
        return 1
    if args.wrist < 0:
        print("--wrist는 0 이상")
        return 1
    wide = slice(dataset.BINS.start - args.shift, dataset.BINS.stop + args.shift)
    ds, lines = dataset.build(args.data, args.sessions, bins=wide, standardized=args.shift == 0)
    print("\n".join(lines))
    if ds is None:
        print("쓸 세션이 없음")
        return 1
    n = len(ds["y"])
    mode = args.mode or ("session" if len(ds["sessions"]) >= 2 else "blocks")
    if not args.mode:
        print(f"나누기 {mode}(--mode 없음: 세션 {len(ds['sessions'])}개 — 둘 이상이면 session, 하나면 blocks)")
    run = args.run or f"{time.strftime('%Y%m%d-%H%M%S')}-{mode}"
    folder = os.path.join(args.out or os.path.join(args.data, "processed", "training"), run)
    try:
        parts, block, split, what, lines = split_frames(ds, mode, args)
    except ValueError as e:
        print(f"나누지 못함: {e}")
        return 1
    print("\n".join(lines))
    train, val, test = parts
    if min(len(train), len(val), len(test)) == 0:
        print(f"나눈 자료가 비었음(학습 {len(train)} · 검증 {len(val)} · 시험 {len(test)})")
        return 1
    mix = kind_mix(ds, parts)
    print("\n".join(mix_lines(mix)))
    if args.wrist > 0:
        has = {name: int(ds["wrist_ok"][idx].sum()) for name, idx in zip(dataset.ROLES, parts)}
        methods = ", ".join(f"{n} {w or '없음'}" for n, w in zip(ds["sessions"], ds["wrist_method"]))
        print(f"손목 정답이 있는 프레임: 학습 {has['train']} · 검증 {has['val']} · 시험 {has['test']}(세션별 방위 맞춤: {methods})")
        if has["train"] == 0:
            print("학습 프레임에 손목 정답이 없어 --wrist를 쓸 수 없음(워치·폰 센서 기록이 있는 세션이 필요)")
            return 1
    os.makedirs(folder, exist_ok=True)
    dataset.save(os.path.join(folder, "dataset.npz"), ds, with_x=args.save_x)
    seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{run}: {mode}, 학습 {len(train)} · 검증 {len(val)} · 시험 {len(test)}프레임, 장치 {device}"
          f"{'(' + torch.cuda.get_device_name(0) + ')' if device == 'cuda' else ' — GPU가 없어 아주 느리다'}")
    x = torch.as_tensor(ds["x"], device=device)
    augment = None
    if args.shift or args.mask:
        augment = Augment(x, args.shift, args.mask)
        x = augment.center()
        hide = f", 그중 {MASK_PROB:.0%}에 시간 {MASK_ROWS[0]}–{MASK_ROWS[1]}줄·거리 {MASK_BINS[0]}–{MASK_BINS[1]}칸 가리기" if args.mask else ""
        print(f"증강: 묶음의 {AUG_FRACTION:.0%}에 거리 ±{args.shift}칸 이동{hide}")
    target = ds["y"]
    if args.wrist > 0:  # 손가락 60 · u 3 · 깊이 축 3 · ok 1(loss_fn)
        target = np.concatenate([ds["y"], ds["wrist"], ds["wrist_depth"], ds["wrist_ok"][:, None].astype(np.float32)], axis=1)
    y = torch.as_tensor(target, device=device)
    start = time.time()
    try:
        model, history, best_epoch, stopped = fit(x, y, segments(ds["frame_ns"], ds["session"], train), segments(ds["frame_ns"], ds["session"], val),
                                                  args.epochs, args.seed, max_minutes=args.max_minutes, augment=augment, wrist_weight=args.wrist)
    except FloatingPointError as e:
        print(f"학습을 멈춤: {e}")
        return 1
    seconds = time.time() - start
    out = predict(model, x, np.arange(n))
    pred = out[:, :FINGER_OUT]
    m = instruments(ds, pred, train, test)
    yy = ds["y"].reshape(-1, 20, 3)
    mm = ds["length_mm"][ds["session"]]
    m["splitMpjpeMm"] = {k: float(np.mean(joints.frame_errors(pred.reshape(-1, 20, 3)[v], yy[v]) * mm[v]))
                         for k, v in (("train", train), ("val", val), ("test", test))}
    m.update({"format": 2, "run": run, "mode": mode, "sessions": ds["sessions"].tolist(), "lengthMm": ds["length_mm"].tolist(),
              "frames": {"train": int(len(train)), "val": int(len(val)), "test": int(len(test))}, "seed": args.seed,
              "epochs": len(history), "maxEpochs": args.epochs, "maxMinutes": args.max_minutes, "stoppedBy": stopped, "bestEpoch": best_epoch,
              "trainSeconds": seconds,
              "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu", "history": history, "mix": mix, "split": split,
              "augment": {"shift": args.shift, "mask": args.mask, "fraction": AUG_FRACTION, "maskProb": MASK_PROB, "maskRows": list(MASK_ROWS),
                          "maskBins": list(MASK_BINS)} if augment else None})
    if args.wrist > 0:
        m["wrist"] = {"weight": args.wrist, "depthWeight": DEPTH_WEIGHT, **wrist_metrics(ds, out[:, FINGER_OUT:FINGER_OUT + WRIST_OUT], train, test)}
    torch.save({"arch": ARCH, "in_chans": 2, "num_classes": int(model.num_classes), "bins": [dataset.BINS.start, dataset.BINS.stop],
                "window": dataset.WINDOW, "mode": mode, "seed": args.seed, "epoch": best_epoch, "sessions": ds["sessions"].tolist(), "wrist": args.wrist,
                "state_dict": model.state_dict()},
               os.path.join(folder, "model.pt"))
    hs_io.write_json(os.path.join(folder, "metrics.json"), m)
    names = m["sessions"]
    which = ", ".join(names) if len(names) <= 3 else f"{len(names)}개({names[0]} … {names[-1]})"
    pieces, piece_labels = figure_pieces(ds, test, block)
    figure(os.path.join(folder, "report.png"), ds, pred, parts, m, f"{run}: {what} — 세션 {which}", pieces, piece_labels)
    why = {"patience": f"검증 손실이 {PATIENCE} epoch 동안 나아지지 않음", "epochs": f"최대 {args.epochs} epoch", "time": f"벽시계 상한 {args.max_minutes:g}분"}[stopped]
    print("\n".join([f"{run}: {len(history)} epoch({why}), {seconds / 60:.1f}분, 최고 epoch {best_epoch}"] + report(m)))
    print(f"결과: {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
