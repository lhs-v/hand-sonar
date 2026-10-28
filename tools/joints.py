"""손 관절 정답 전처리, 뼈, 손바닥 틀, 계기 계산(손 소나 계획 4, 학습 설계 §3·§5, 조사 보고서 §5 표).

정답 전처리(설계 §3):
1. 거르기(keep_frames): 판정 점수(landmarks.csv score, 추적하는 손의 MediaPipe 손잡이 점수)가 0.5 아래인 프레임, 손이 둘 보인 줄에서
   손목이 튄 프레임(hs_io.read_landmarks valid = pairs.csv hand), 손목–새끼 MCP 길이가 세션 중앙값 ±3σ 밖인 프레임을 뺀다.
   σ = 1.4826 × MAD(중앙 절대 편차를 정규 분포의 σ에 맞추는 배율, robust z 3). 배율 없는 3 MAD(≈ 2σ)는 20261027-235412에서
   5.9%(230프레임)를 빼며 손목 굽히기 28%·손목 젖히기 19%처럼 동작에 치우쳤다(1.4826배면 77프레임). unit_hand가 프레임마다 자기
   |p17 − p0|로 나누므로 MediaPipe의 프레임별 크기 오차는 목표에 해가 없어, 크게 벗어난 프레임만 뺀다. 보간하지 않는다.
   판정 점수 하한 0.5는 구조상 아무것도 거르지 않는다: 폰 앱(HandTracker)은 MediaPipe 손잡이 분류의 첫(최고) 범주가 추적하는 손인
   손만 고르고 그 점수를 적는데, 두 범주 분류의 최고 점수라 늘 0.5 이상이다(이 세션 최소 0.831). 손 품질은 MediaPipe 안의 손 찾기·
   있음·추적 문턱 0.5가 이미 지킨다. 하한은 설계 §3-1 값 그대로 두고(손이 없는 프레임의 nan을 거르는 자리), 품질 문턱이 필요해지면
   손잡이 점수 0.9처럼 뜻 있는 값을 쓴다(이 세션 0.9 아래 14프레임).
2. 거울 되돌리기(to_left): world 좌표는 거울 그림에서 나왔으므로 wx → −wx. 오른손 세션은 한 번 더 뒤집어 왼손으로 접는다(한 손 틀).
   확인(thumb_palmar): 실제 왼손이면 손바닥 틀에서 엄지 관절 2·3·4가 법선 반대쪽(손바닥 쪽, z < 0)에 있다. 거울 그대로면 법선 쪽이다
   (실데이터 20261027-235412, 쓴 3,802프레임: 되돌린 뒤 손바닥 쪽 99.66%, 되돌리기 전에는 법선 쪽 99.66%. 공개 영상 확인은 99.7%).
3. 손목 원점: 21점에서 p0을 뺀다. 목표는 1–20번 20점(60값).
4. 손바닥 틀(palm_frame): x = (p5 − p0) 단위벡터, z = (p5 − p0) × (p17 − p0) 단위벡터(실제 왼손에서는 손등 쪽), y = z × x
   (새끼 쪽이 +y). 회전의 행이 (x, y, z)이고 행렬식은 늘 +1이다. 공개 영상 관절 비교(joint_check.py)와 같은 틀이다.
5. 손 크기(unit_hand): |p17 − p0|를 1로 맞춘 단위 손으로 학습한다. 보고는 세션의 L(mm, 거른 프레임의 |p17 − p0| 중앙값)을 곱한다.
6. 시간 평활은 첫 판에 하지 않는다.
7. 손목은 목표에서 뺀다. 거울을 되돌린 카메라 좌표의 v9 = p9 − p0(m)은 따로 저장만 한다(targets).

계기(설계 §5): 아래 함수는 단위 손 좌표의 (프레임, 20, 3) 배열을 받는다. mm는 부르는 쪽이 프레임마다 L을 곱한다.
- frame_errors: 프레임마다 관절 평균 거리(MPJPE의 프레임 값). joints=MOVING이면 손바닥 점 5·9·13·17을 뺀 움직이는 점 16개.
- mean_pose·medoid: 학습 평균 자세(상수)와 학습 프레임 중 평균에 가장 가까운 실제 프레임.
- collapse: 관절·축마다 예측 표준편차 ÷ 정답 표준편차(움직이는 점만. 점 5의 y·z와 점 17의 z는 틀 정의상 늘 0이다). 정답과 무관한
  프레임 떨림도 σ_pred를 키워, 아무것도 따라가지 못하는 모델이 "덜 쏠림"으로 읽힌다(검토: 시간 나누기 모델 σ비 0.674, 기울기 −0.015).
- regression: 관절·축마다 예측을 정답에 회귀한 기울기 cov(p, t) ÷ var(t)와 상관. 기울기는 정답과 무관한 떨림에 부풀지 않는다.
- smooth_within: 연속 덩어리 안에서만 SMOOTH_FRAMES 가운데 이동 평균한 예측. 떨림을 걷은 예측으로 σ비와 뼈 길이 부족을 다시 잰다.
- bone_shortfall·by_segment: 뼈 20개마다 (1 − 예측 길이 중앙값 ÷ 정답 길이 중앙값) × 100, 마디(손바닥·첫·가운데·끝) 평균.
- toward: (예측 − 정답)을 start → end(학습 평균 자세 → 학습 ASL-5 평균) 단위 방향에 투영한 값. 편 손 쪽 치우침을 단위 손 길이로 잰다.
- closer_to: 예측이 자기 정답보다 template(ASL-5 평균)에 더 가까운 프레임 비율(참고. 평균 쏠림과 ASL-5 쏠림을 가르지 못한다).
- hold_floor: 정답 잡음 바닥. 유지 구간마다 그 구간 중앙값 자세까지의 관절 평균 거리.
"""
import site
import sys

sys.path.append(site.getusersitepackages())  # python -I에서도 사용자 설치 numpy를 찾게
import numpy as np  # noqa: E402

FINGERS = (("엄지", (1, 2, 3, 4)), ("검지", (5, 6, 7, 8)), ("중지", (9, 10, 11, 12)), ("약지", (13, 14, 15, 16)), ("소지", (17, 18, 19, 20)))
BONES = tuple(((0,) + idx)[s: s + 2] for _, idx in FINGERS for s in range(4))  # 뼈 20개: 손가락마다 손목→첫 관절, 마디 3개
SEGMENTS = ("손바닥", "첫 마디", "가운데 마디", "끝 마디")  # BONES[4k + s]의 s
PALM = (5, 9, 13, 17)  # 손바닥 점(한 사람 안에서 거의 상수)
MOVING = tuple(j - 1 for j in range(1, 21) if j not in PALM)  # 목표 배열(점 1–20 → 자리 0–19) 안 움직이는 점 16개의 자리
TIPS = (4, 8, 12, 16, 20)
MIN_SCORE = 0.5  # 판정 점수 하한(설계 §3-1). 손잡이 최고 범주 점수라 늘 0.5 이상이어서 구조상 거르지 않는다(머리말 1)
MAD_LIMIT = 3.0  # 손목–새끼 MCP 길이가 세션 중앙값에서 이 σ배 넘게 벗어나면 뺀다
MAD_SCALE = 1.4826  # σ = MAD × 이 값(정규 분포에서 MAD ≈ 0.6745σ). 배율 없이 3 MAD면 ±2σ라 손목 동작이 치우쳐 빠졌다(머리말 1)
MIRROR = np.array([-1.0, 1.0, 1.0])  # wx → −wx
SMOOTH_FRAMES = 9  # 떨림을 걷는 평활 폭: 가운데 ± 4프레임(30 fps에서 0.3 s, 2 s 동작보다 훨씬 짧음). 검토가 같은 폭으로 σ비 0.674 → 0.349를 보였다


def to_left(world, hand):
    """파일의 world 좌표(거울 그림 기준, (M, 21, 3)) → 실제 손 모양. 오른손 세션은 다시 뒤집어 왼손으로 접는다(거울 되돌리기 + 접기 = 그대로)."""
    out = np.asarray(world, dtype=np.float64) * MIRROR
    return out * MIRROR if hand == "right" else out


def palm_frame(points):
    """(M, 21, 3) → (손목 원점 손바닥 틀 좌표 (M, 21, 3), 회전 (M, 3, 3)). 회전의 행이 틀의 x·y·z 축(원래 좌표로 적은 단위벡터)이다."""
    p = np.asarray(points, dtype=np.float64)
    a = p[:, 5] - p[:, 0]
    b = p[:, 17] - p[:, 0]
    ex = a / np.linalg.norm(a, axis=1, keepdims=True)
    n = np.cross(a, b)
    ez = n / np.linalg.norm(n, axis=1, keepdims=True)
    ey = np.cross(ez, ex)
    rot = np.stack([ex, ey, ez], axis=1)
    return np.einsum("mac,mjc->mja", rot, p - p[:, :1]), rot


def unit_hand(coords):
    """손바닥 틀 좌표를 |p17|(손목–새끼 MCP)이 1이 되게 나눈다. (단위 손 좌표, 나눈 길이 (M,))."""
    length = np.linalg.norm(coords[:, 17], axis=1)
    return coords / length[:, None, None], length


def thumb_palmar(coords):
    """손바닥 틀 좌표에서 엄지 관절 2·3·4의 평균 z가 0보다 작은(손바닥 쪽) 프레임 비율. 실제 왼손 틀이면 1에 가깝다."""
    return float((coords[:, [2, 3, 4], 2].mean(axis=1) < 0).mean())


def scored(world, score, valid):
    """(점수·손목 조건을 지난 프레임 (M,) bool, 손목–새끼 MCP 길이 (M,) m). 길이 거르기의 중앙값·σ는 이 프레임에서 잰다."""
    length = np.linalg.norm(world[:, 17] - world[:, 0], axis=1)
    return np.asarray(valid, dtype=bool) & np.isfinite(length) & (np.nan_to_num(score, nan=-1.0) >= MIN_SCORE), length


def length_limits(length, base):
    """길이 거르기의 (중앙값, σ = MAD_SCALE × MAD) m. base 프레임에서 잰다."""
    med = float(np.median(length[base]))
    return med, MAD_SCALE * float(np.median(np.abs(length[base] - med)))


def keep_frames(world, score, valid):
    """설계 §3-1 거르기. (남길 프레임 (M,) bool, 손목–새끼 MCP 길이 (M,) m). 중앙값·σ는 점수·손목 조건을 지난 프레임에서 잰다."""
    base, length = scored(world, score, valid)
    if not base.any():
        return base, length
    med, sigma = length_limits(length, base)
    return base & (np.abs(np.nan_to_num(length, nan=np.inf) - med) <= MAD_LIMIT * sigma), length


def targets(world, hand):
    """거른 프레임의 world (M, 21, 3) → (목표 (M, 20, 3) 단위 손, v9 (M, 3) m, 엄지가 손바닥 쪽인 프레임 비율)."""
    real = to_left(world, hand)
    coords, _ = palm_frame(real)
    unit, _ = unit_hand(coords)
    return unit[:, 1:], real[:, 9] - real[:, 0], thumb_palmar(coords)


def with_wrist(points):
    """목표 (N, 20, 3)에 손목(원점)을 붙여 (N, 21, 3)."""
    p = np.asarray(points, dtype=np.float64)
    return np.concatenate([np.zeros((len(p), 1, 3)), p], axis=1)


def bone_lengths(points):
    """목표 (N, 20, 3)의 뼈 20개 길이 (N, 20)."""
    p = with_wrist(points)
    return np.stack([np.linalg.norm(p[:, j] - p[:, i], axis=1) for i, j in BONES], axis=1)


def frame_errors(pred, true, joints=None):
    """프레임마다 관절 평균 거리 (N,). joints는 목표 배열 안 자리(없으면 20점 모두). pred가 (1, 20, 3)이면 모든 프레임에 같은 자세다."""
    d = np.linalg.norm(np.asarray(pred, dtype=np.float64) - np.asarray(true, dtype=np.float64), axis=-1)
    return (d if joints is None else d[:, list(joints)]).mean(axis=1)


def mean_pose(train):
    """학습 평균 자세 (20, 3)."""
    return np.asarray(train, dtype=np.float64).mean(axis=0)


def medoid(train):
    """학습 프레임 중 평균 자세까지 관절 평균 거리가 가장 짧은 실제 프레임 (20, 3)."""
    train = np.asarray(train, dtype=np.float64)
    return train[int(np.argmin(frame_errors(train, mean_pose(train)[None])))]


def collapse(pred, true, joints=MOVING):
    """관절·축마다 예측 표준편차 ÷ 정답 표준편차 (len(joints), 3). 1보다 한참 작으면 평균으로 쏠린 것이다."""
    j = list(joints)
    return np.asarray(pred)[:, j].std(axis=0) / np.asarray(true)[:, j].std(axis=0)


def regression(pred, true, joints=MOVING):
    """관절·축마다 (기울기 cov(p, t) ÷ var(t), 상관) 각 (len(joints), 3). 기울기 1 = 정답만큼 움직임, 0 = 정답과 무관(평균에 붙었거나 떨림만).
    정답과 무관한 떨림은 σ비를 키우지만 기울기는 키우지 않는다(상관은 낮춘다). 예측이 움직이지 않는 관절·축의 상관은 0으로 둔다."""
    j = list(joints)
    p = np.asarray(pred, dtype=np.float64)[:, j]
    t = np.asarray(true, dtype=np.float64)[:, j]
    dp, dt = p - p.mean(axis=0), t - t.mean(axis=0)
    cov = (dp * dt).mean(axis=0)
    spread = np.sqrt((dp ** 2).mean(axis=0) * (dt ** 2).mean(axis=0))
    return cov / (dt ** 2).mean(axis=0), np.divide(cov, spread, out=np.zeros_like(cov), where=spread > 0)


def smooth_within(points, pieces, width=SMOOTH_FRAMES):
    """덩어리(프레임 번호 배열 목록, 각각 시각 순서) 안에서만 가운데 width 이동 평균한 사본. 덩어리 끝은 창을 줄이고(덩어리 밖 프레임을
    섞지 않는다), 덩어리에 들지 않은 프레임은 그대로 둔다."""
    src = np.asarray(points, dtype=np.float64)
    out = src.copy()
    half = width // 2
    for piece in pieces:
        piece = np.asarray(piece, dtype=np.int64)
        c = np.concatenate([np.zeros((1,) + src.shape[1:]), np.cumsum(src[piece], axis=0)])
        k = np.arange(len(piece))
        lo, hi = np.maximum(k - half, 0), np.minimum(k + half + 1, len(piece))
        out[piece] = (c[hi] - c[lo]) / (hi - lo).reshape((-1,) + (1,) * (src.ndim - 1))
    return out


def toward(pred, true, start, end):
    """프레임마다 (pred − true)를 start → end 단위 방향(20점 × xyz = 60차원)에 투영한 값 (N,). +면 end 쪽으로 치우쳤다.
    pred가 한 자세 (1, 20, 3)이어도 된다(평균 자세 예측기)."""
    u = (np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)).ravel()
    d = np.asarray(pred, dtype=np.float64) - np.asarray(true, dtype=np.float64)
    return d.reshape(len(d), -1) @ (u / np.linalg.norm(u))


def bone_shortfall(pred, true):
    """뼈마다 (1 − 예측 길이 중앙값 ÷ 정답 길이 중앙값) × 100 (20,). +면 예측 뼈가 짧다. pred가 한 자세(1, 20, 3)여도 된다."""
    return (1.0 - np.median(bone_lengths(pred), axis=0) / np.median(bone_lengths(true), axis=0)) * 100.0


def by_segment(per_bone):
    """뼈 20개 값 → 마디 4개(SEGMENTS 순서, 다섯 손가락 평균)."""
    return np.asarray(per_bone, dtype=np.float64).reshape(5, 4).mean(axis=0)


def closer_to(pred, true, template):
    """예측이 자기 정답보다 template (20, 3)에 더 가까운 프레임 비율."""
    return float((frame_errors(pred, np.asarray(template)[None]) < frame_errors(pred, true)).mean())


def hold_floor(true, groups, min_frames=5):
    """정답 잡음 바닥: 유지 구간(groups 값이 같은 프레임)마다 구간 중앙값 자세(좌표별 중앙값)까지의 관절 평균 거리 (N,).
    프레임이 min_frames보다 적은 구간은 nan."""
    true = np.asarray(true, dtype=np.float64)
    groups = np.asarray(groups)
    out = np.full(len(true), np.nan)
    for g in np.unique(groups):
        k = np.flatnonzero(groups == g)
        if len(k) >= min_frames:
            out[k] = frame_errors(true[k], np.median(true[k], axis=0)[None])
    return out
