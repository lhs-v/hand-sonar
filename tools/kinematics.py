"""손가락 관절 각도 표현(2026-11-01) — 손바닥 점과 뼈 길이를 고정하고, 뼈마다 굽힘·벌림 두 각으로 점 20개를 만든다(순기구학).

- 좌표는 목표 배열(joints.targets: 단위 손 손바닥 틀, 손목 원점, 점 1–20이 자리 0–19)이다.
- 고정(fit_template, 학습 프레임에서): 손바닥 점 다섯(PALM: 엄지 CMC 점 1, 검지·중지·약지·소지 MCP 점 5·9·13·17)은 학습 평균,
  뼈 15개(BONES: 손가락마다 뿌리 → 다음 관절 → 다음 관절 → 끝)의 길이는 학습 중앙값이다. HaMeR(MANO) 정답에서는 손바닥 점이
  0.4 mm 이하로만 움직이고 뼈 길이 변동이 0.1–0.3%라 거의 그대로 맞는다. MediaPipe 정답은 손바닥 2–3 mm, 뼈 2–10%가 표현 밖에 남는다.
- 각도(ik·fk): 뼈마다 부모 방향(첫 뼈는 손목 → 뿌리)에 대한 굽힘 f와 벌림 a 두 값, 모두 30값(라디안)이다. 굽힘은 손바닥 쪽(−z)이 +,
  벌림은 소지 쪽(+y)이 +다. 방향 d = cos a·(cos f·e1 − sin f·e3) + sin a·e2. 첫 뼈의 틀은 e1 = 부모 방향, e2 = z × e1의 단위벡터
  (z = 손바닥 법선, 손등 쪽), e3 = e1 × e2다. 다음 뼈로 넘어갈 때 e1 = d로 두고, e2는 d에 수직인 성분만 남겨 다시 단위로 만들고(평행 이동),
  e3 = e1 × e2로 둔다. 그래서 한 평면 안에서만 굽는 손가락은 둘째·셋째 뼈의 벌림이 0이다. 뼈 비틀림은 점 위치를 바꾸지 않아 두지
  않는다. 뼈 길이가 템플릿과 같으면 fk(ik(y)) = y다.
- 칸(fit_template bins > 0): 각도마다 학습 0.5–99.5% 범위의 양쪽에 그 폭의 10%와 2°를 더한 구간을 bins칸 중심으로 나눈다.
  soft_argmax는 칸 확률의 기댓값이다. soft_labels는 정답 각도(구간 안으로 자름)에 칸 폭의 SIGMA_BINS배 가우스를 씌운 분포이고,
  bin_loss는 그 분포와의 교차 엔트로피다(SimCC, ECCV 2022의 1차원 분류).
- Head: 몸통(timm 모델) 출력의 앞부분을 각도로 바꿔 순기구학한 점 60값을 내고, 뒤 extra값(손목 3값 등)은 그대로 붙인다. 회귀(bins 0)는
  각도 = 학습 평균 + 학습 표준편차 × 출력이고, 칸은 각도마다 bins개 로짓이다(마지막 로짓을 logits에 남긴다). 템플릿은 버퍼라 state_dict에 들어간다.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import hs_io  # noqa: E402,F401  사용자 설치 numpy·torch 경로도 여기서 붙는다
import numpy as np  # noqa: E402
import torch  # noqa: E402

PALM = (0, 4, 8, 12, 16)  # 목표 자리: 엄지 CMC(점 1), 검지·중지·약지·소지 MCP(점 5·9·13·17)
BONES = tuple((root + k, root + k + 1) for root in PALM for k in range(3))  # 15개: 손가락마다 뿌리 → … → 끝
ANGLES = 2 * len(BONES)
SIGMA_BINS = 1.5  # 정답 칸 분포의 가우스 폭(칸)
MARGIN = 0.1  # 칸 구간을 학습 범위 폭의 이만큼 양쪽에 넓힌다
MARGIN_RAD = float(np.radians(2.0))  # 그리고 2°를 더한다


def _unit(v):
    return v / torch.linalg.norm(v, dim=-1, keepdim=True).clamp(min=1e-12)


def _first_frame(e1):
    """첫 뼈의 틀 (e1, e2, e3): e2 = z × e1의 단위벡터, e3 = e1 × e2."""
    z = torch.zeros_like(e1)
    z[..., 2] = 1.0
    e2 = _unit(torch.linalg.cross(z, e1))
    return e1, e2, torch.linalg.cross(e1, e2)


def _next_frame(e2, d):
    """뼈 d로 넘어간 틀: e1 = d, e2는 d에 수직인 성분만 남겨 단위로, e3 = e1 × e2."""
    e2 = _unit(e2 - (e2 * d).sum(-1, keepdim=True) * d)
    return d, e2, torch.linalg.cross(d, e2)


def fk_t(angles, palm, lengths):
    """각도 (B, 15, 2)[굽힘, 벌림] · 손바닥 점 (5, 3) · 뼈 길이 (15,) → 점 (B, 20, 3)(torch, 미분 가능)."""
    count = angles.shape[0]
    points = [None] * 20
    for f, root in enumerate(PALM):
        p = palm[f].expand(count, 3)
        points[root] = p
        e1, e2, e3 = _first_frame(_unit(p))
        for k in range(3):
            j = 3 * f + k
            bend, spread = angles[:, j, 0:1], angles[:, j, 1:2]
            d = torch.cos(spread) * (torch.cos(bend) * e1 - torch.sin(bend) * e3) + torch.sin(spread) * e2
            p = p + lengths[j] * d
            points[root + k + 1] = p
            e1, e2, e3 = _next_frame(e2, d)
    return torch.stack(points, dim=1)


def ik_t(points):
    """점 (B, 20, 3) → 각도 (B, 15, 2)[굽힘, 벌림](torch). 뼈 방향만 보고 길이는 보지 않는다."""
    out = []
    for root in PALM:
        p = points[:, root]
        e1, e2, e3 = _first_frame(_unit(p))
        for k in range(3):
            q = points[:, root + k + 1]
            d = _unit(q - p)
            spread = torch.asin((d * e2).sum(-1).clamp(-1.0, 1.0))
            bend = torch.atan2(-(d * e3).sum(-1), (d * e1).sum(-1))
            out.append(torch.stack([bend, spread], dim=-1))
            e1, e2, e3 = _next_frame(e2, d)
            p = q
    return torch.stack(out, dim=1)


def fk(angles, palm, lengths):
    """fk_t의 numpy 판(float64)."""
    with torch.no_grad():
        return fk_t(*(torch.as_tensor(np.asarray(v, dtype=np.float64)) for v in (angles, palm, lengths))).numpy()


def ik(points):
    """ik_t의 numpy 판(float64)."""
    with torch.no_grad():
        return ik_t(torch.as_tensor(np.asarray(points, dtype=np.float64))).numpy()


def fit_template(y, bins=0):
    """학습 목표 (N, 20, 3) → {"palm" (5, 3) 평균, "lengths" (15,) 중앙값, 각도 "mean"·"std" (30,), bins > 0이면 칸 중심 "centers" (30, bins)}."""
    y = np.asarray(y, dtype=np.float64)
    angles = ik(y).reshape(len(y), ANGLES)
    t = {"palm": y[:, list(PALM)].mean(axis=0),
         "lengths": np.array([np.median(np.linalg.norm(y[:, c] - y[:, p], axis=1)) for p, c in BONES]),
         "mean": angles.mean(axis=0), "std": np.maximum(angles.std(axis=0), np.radians(1.0))}
    if bins:
        lo, hi = np.percentile(angles, 0.5, axis=0), np.percentile(angles, 99.5, axis=0)
        pad = MARGIN * (hi - lo) + MARGIN_RAD
        t["centers"] = np.linspace(lo - pad, hi + pad, int(bins), axis=1)
    return t


def empty_template(bins=0):
    """fit_template와 모양이 같은 빈 템플릿(저장한 state_dict를 불러올 자리)."""
    t = {"palm": np.zeros((len(PALM), 3)), "lengths": np.zeros(len(BONES)), "mean": np.zeros(ANGLES), "std": np.ones(ANGLES)}
    if bins:
        t["centers"] = np.zeros((ANGLES, int(bins)))
    return t


def soft_argmax(logits, centers):
    """칸 로짓 (B, 30, K) → 각도 (B, 30): 칸 확률의 기댓값."""
    return (torch.softmax(logits, dim=-1) * centers).sum(-1)


def soft_labels(angles, centers):
    """정답 각도 (B, 30) → 칸 분포 (B, 30, K): 구간 안으로 자른 각도에 칸 폭의 SIGMA_BINS배 가우스."""
    width = centers[:, 1:2] - centers[:, 0:1]
    a = torch.maximum(torch.minimum(angles, centers[:, -1]), centers[:, 0])
    w = torch.exp(-0.5 * ((centers - a[..., None]) / (SIGMA_BINS * width)) ** 2)
    return w / w.sum(-1, keepdim=True)


def bin_loss(logits, angles, centers):
    """칸 교차 엔트로피 −Σ q·log p(각도·프레임 평균). logits (B, 30, K), 정답 각도 (B, 30)."""
    return -(soft_labels(angles, centers) * torch.log_softmax(logits, dim=-1)).sum(-1).mean()


def angle_errors(pred, true):
    """예측·정답 점 (N, 20, 3)의 뼈별 각 차(도, ±180°로 감음)의 절댓값 평균: 굽힘·벌림 전체와 뼈 15개마다."""
    d = ik(pred) - ik(true)
    d = np.degrees(np.abs((d + np.pi) % (2 * np.pi) - np.pi))
    return {"flexionDeg": float(d[..., 0].mean()), "abductionDeg": float(d[..., 1].mean()),
            "byBoneFlexionDeg": d[..., 0].mean(axis=0).tolist(), "byBoneAbductionDeg": d[..., 1].mean(axis=0).tolist()}


class Head(torch.nn.Module):
    """몸통 출력의 앞 30값(회귀) 또는 30 × K값(칸 로짓)을 각도로 바꿔 순기구학한 점 60값 + 뒤 extra값."""

    def __init__(self, backbone, template, extra=0):
        super().__init__()
        self.backbone = backbone
        self.extra = int(extra)
        centers = template.get("centers")
        self.bins = 0 if centers is None else int(np.shape(centers)[1])
        for name in ("palm", "lengths", "mean", "std") + (("centers",) if self.bins else ()):
            self.register_buffer(name, torch.as_tensor(np.asarray(template[name]), dtype=torch.float32))
        self.num_classes = 3 * 20 + self.extra
        self.logits = None

    def forward(self, x):
        out = self.backbone(x)
        n = ANGLES * max(self.bins, 1)
        raw, rest = out[:, :n], out[:, n:]
        if self.bins:
            self.logits = raw.reshape(-1, ANGLES, self.bins)
            angles = soft_argmax(self.logits, self.centers)
        else:
            angles = self.mean + self.std * raw
        points = fk_t(angles.reshape(-1, len(BONES), 2), self.palm, self.lengths)
        return torch.cat([points.reshape(-1, 3 * 20), rest], dim=1)
