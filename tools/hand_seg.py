"""손만 보이기(render_video.py --seg): 영상 랜드마크 21점의 사전 마스크 + GrabCut 다듬기. numpy와 OpenCV만 쓴다.

좌표는 모두 '자르기 좌표'(손 패널과 같은 정사각 자르기, 한 변 SEG 픽셀)다.
1. 사전 마스크: 손바닥 다각형(점 0·1·2·5·9·13·17의 볼록 껍질) + 뼈 20개마다 둥근 끝의 굵은 선(캡슐). 반지름 = K × 손 크기(px), 손 크기 =
   max(|p0 − p9|, |p5 − p17| ÷ 0.7, 바닥값)(손이 카메라 쪽으로 기울어 |p0 − p9|가 줄어도 반지름이 무너지지 않게). 엄지는 K_THUMB배.
2. GrabCut(GC_INIT_WITH_MASK): 확실한 앞 = 뼈 중심선 캡슐(반지름 × CORE, 손목에서 시작하는 뼈는 길이의 CORE_TRIM만큼 안에서 시작 — 시계줄이
   들지 않게) + 줄인 손바닥, 앞일 듯 = 캡슐(반지름 × MID) + 손바닥, 뒤일 듯 = 사전 마스크를 RING px 부풀린 띠, 확실한 뒤 = 나머지. ITERS번 반복.
   손가락 사이 좁은 틈은 GrabCut의 매끄러움 항이 앞으로 메우므로, 코어 픽셀 색(피부)과 바깥 띠 색(배경)의 작은 가우시안 혼합으로 잰 로그
   우도비가 배경 쪽으로 SEED_T 넘는 픽셀(코어 밖)은 미리 '확실한 뒤'로 못박는다. 최종 앞 = GrabCut 앞 ∩ 부풀린 사전 마스크 — 그 밖은 보이지 않는다.
3. GrabCut 앞이 사전 마스크 넓이의 MIN_KEEP 아래거나 GrabCut이 못 돌면 사전 마스크(조금 깎음)를 쓴다.
상태: 0 = GrabCut, 1 = 사전 마스크로 대체, 2 = 손 없음(랜드마크 없음). 값은 첫 학습 세션(20261027-235412)의 확인 프레임(얼굴·몸 앞의 손, 주먹,
편 손, 손목 꺾기, 빠른 동작, 영상 밖으로 잘린 손)을 보며 맞췄다.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # python -I는 3.11부터 -P를 겸해 스크립트 폴더를 경로에 넣지 않는다
import hs_io  # noqa: E402,F401  사용자 설치 numpy 경로도 여기서 붙는다
import cv2  # noqa: E402
import numpy as np  # noqa: E402

SEG = 320  # 분할 해상도(= 손 패널 크기)
K = 0.12  # 캡슐 반지름 / 손 크기
K_THUMB = 1.30  # 엄지는 이 배수만큼 더 굵게
CORE = 0.45  # 확실한 앞(뼈 중심선) 반지름 / 캡슐 반지름
CORE_TRIM = 0.30  # 코어에서 손목 쪽 뼈(0–x)의 시작을 뼈 길이의 이만큼 안으로
PALM_ERODE = 0.50  # 손바닥 다각형을 안쪽으로 줄이는 양 / 반지름
MID = 0.6  # '앞일 듯' 캡슐 반지름 / 사전 마스크 반지름
RING = 6  # 사전 마스크 밖 '뒤일 듯' 띠 두께(px)
ITERS = 3  # GrabCut 반복
SIZE_FLOOR = 0.60  # 손 크기 바닥 = SIZE_FLOOR × 세션 중앙값 손 크기
MIN_KEEP = 0.45  # GrabCut 앞 넓이 / 사전 마스크 넓이가 이보다 작으면 망가진 것으로 보고 사전 마스크로
ROI_PAD = 14  # GrabCut에 넣는 영역 = 부풀린 사전 마스크 상자 + 이 여백
SEED_T = 2.0  # 로그 우도비(배경 − 피부)가 이보다 크면 '확실한 뒤'
BAND = 24  # 배경 색 모델을 만드는 띠: 부풀린 사전 마스크 밖 이 두께(px)
K_FG, K_BG = 3, 4  # 색 모델 성분 수
LAB_L = 0.6  # Lab의 L(밝기) 배율 — 1보다 작으면 색을 더 중시
SKIN_PCT, SKIN_MARGIN = 2.0, 2.0  # 띠 픽셀 중 피부 모델 우도가 코어의 SKIN_PCT 백분위 − SKIN_MARGIN 밑인 것만 배경 표본으로
SMOOTH = 1.2  # 최종 앞 마스크의 톱니 없애기: 가우시안(시그마) 뒤 0.5 문턱
FEATHER = 1.0  # 가장자리 부드럽게(가우시안 시그마, px)

BONES = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20))  # 손바닥을 손목에서 부채꼴로 채우는 뼈(그림의 뼈대와 다름)
THUMB_BONES = BONES[:4]
PALM_PTS = (0, 1, 2, 5, 9, 13, 17)
SH = 4  # 부분 픽셀 그리기 비트


def hand_size(pts, floor=0.0):
    """손 크기(px): 손목–가운데 MCP, 손바닥 폭을 손목–MCP로 환산한 값, 바닥값 중 가장 큰 것."""
    s09 = float(np.hypot(*(pts[0] - pts[9])))
    s517 = float(np.hypot(*(pts[5] - pts[17]))) / 0.7
    return max(s09, s517, floor)


def _fx(v):
    return int(round(float(v) * (1 << SH)))


def _capsule(img, a, b, r, val=255):
    th = max(1, int(round(2 * r)))
    cv2.line(img, (_fx(a[0]), _fx(a[1])), (_fx(b[0]), _fx(b[1])), val, th, cv2.LINE_8, SH)
    for p in (a, b):
        cv2.circle(img, (_fx(p[0]), _fx(p[1])), _fx(r), val, -1, cv2.LINE_8, SH)


def draw_capsules(shape, pts, r, r_thumb, scale=1.0, trim=0.0):
    """뼈마다 둥근 끝 굵은 선. trim > 0이면 손목에서 시작하는 뼈의 시작을 뼈 길이의 trim만큼 안으로 당긴다(코어용)."""
    m = np.zeros(shape, np.uint8)
    for bone in BONES:
        a, b = pts[bone[0]], pts[bone[1]]
        if trim > 0 and bone[0] == 0:
            a = a + trim * (b - a)
        _capsule(m, a, b, (r_thumb if bone in THUMB_BONES else r) * scale)
    return m


def palm_polygon(shape, pts):
    m = np.zeros(shape, np.uint8)
    hull = cv2.convexHull(np.round(pts[list(PALM_PTS)] * (1 << SH)).astype(np.int32))
    cv2.fillConvexPoly(m, hull, 255, cv2.LINE_8, SH)
    return m


def ellipse(rad):
    d = max(1, int(round(rad)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))


def build_masks(pts, size, valid=None, shape=(SEG, SEG)):
    """(사전 마스크, 코어, 부풀린 사전 마스크, 앞일 듯) 불리언 마스크. valid(영상 안 = 참)가 있으면 그 안으로 자른다."""
    r = K * size
    r_th = K_THUMB * r
    palm = palm_polygon(shape, pts)
    prior = (draw_capsules(shape, pts, r, r_th) > 0) | (palm > 0)
    core = (draw_capsules(shape, pts, r, r_th, scale=CORE, trim=CORE_TRIM) > 0) | (cv2.erode(palm, ellipse(PALM_ERODE * r)) > 0)
    likely = (draw_capsules(shape, pts, r, r_th, scale=MID) > 0) | (palm > 0)
    dil = cv2.dilate(prior.astype(np.uint8), ellipse(RING)) > 0
    if valid is not None:
        prior, core, dil, likely = prior & valid, core & valid, dil & valid, likely & valid
    core &= prior
    return prior, core, dil, likely | core


def _fit_gmm(X, k, rng):
    """kmeans로 나눠 성분마다 평균·공분산을 구한 작은 가우시안 혼합. X (N, 3)."""
    X = X.astype(np.float32)
    if len(X) > 3000:
        X = X[rng.choice(len(X), 3000, replace=False)]
    k = int(max(1, min(k, len(X) // 12)))
    cv2.setRNGSeed(7)
    _, lab, _ = cv2.kmeans(X, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 1, cv2.KMEANS_PP_CENTERS)
    lab = lab.ravel()
    comps = []
    for c in range(k):
        Xc = X[lab == c].astype(np.float64)
        if len(Xc) < 6:
            continue
        cov = np.cov(Xc.T) + np.eye(3) * 4.0
        comps.append((len(Xc) / len(X), Xc.mean(0), np.linalg.inv(cov), np.linalg.slogdet(cov)[1]))
    return comps


def _log_gmm(comps, X):
    out = []
    for w, mu, icov, ld in comps:
        d = X - mu
        out.append(np.log(w) - 0.5 * (np.einsum("ni,ij,nj->n", d, icov, d) + ld + 3 * np.log(2 * np.pi)))
    out = np.stack(out, 1)
    mx = out.max(1, keepdims=True)
    return mx[:, 0] + np.log(np.exp(out - mx).sum(1))


def color_llr(crop_bgr, core, dil, seed=0):
    """부풀린 사전 마스크 안 픽셀마다 log p(배경) − log p(피부)(float32 (H, W), 그 밖은 0). 피부 모델 = 코어 픽셀. 배경 모델 = 부풀린 마스크 밖
    BAND px 띠 중 '피부 모델 밖 색'만(얼굴처럼 피부색인 배경을 넣으면 손가락 가장자리의 밝은 피부까지 배경으로 몰리므로 뺀다)."""
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2Lab).astype(np.float64)
    lab[..., 0] *= LAB_L
    outside = ~cv2.dilate(dil.astype(np.uint8), ellipse(2)).astype(bool)
    band = cv2.dilate(dil.astype(np.uint8), ellipse(BAND)).astype(bool) & outside
    out = np.zeros(core.shape, np.float32)
    if core.sum() < 30 or band.sum() < 30:
        return out
    rng = np.random.RandomState(seed)
    fg = _fit_gmm(lab[core], K_FG, rng)
    if not fg:
        return out
    cut = np.percentile(_log_gmm(fg, lab[core]), SKIN_PCT) - SKIN_MARGIN
    Xb = lab[band]
    notskin = _log_gmm(fg, Xb) < cut
    if notskin.sum() < 150:
        return out
    bg = _fit_gmm(Xb[notskin], K_BG, rng)
    if not bg:
        return out
    ys, xs = np.nonzero(dil)
    X = lab[ys, xs]
    out[ys, xs] = (_log_gmm(bg, X) - _log_gmm(fg, X)).astype(np.float32)
    return out


def grabcut_mask(img_bgr, prior, core, dil, likely, bg_seed, seed=0):
    """GrabCut(GC_INIT_WITH_MASK)의 앞 불리언 마스크(전체 크기). 실패하면 cv2.error."""
    h, w = prior.shape
    gc = np.full((h, w), cv2.GC_BGD, np.uint8)
    gc[dil] = cv2.GC_PR_BGD
    gc[likely] = cv2.GC_PR_FGD
    if bg_seed is not None:
        gc[bg_seed & ~core] = cv2.GC_BGD
    gc[core] = cv2.GC_FGD
    ys, xs = np.nonzero(dil)
    y0, y1 = max(0, ys.min() - ROI_PAD), min(h, ys.max() + 1 + ROI_PAD)
    x0, x1 = max(0, xs.min() - ROI_PAD), min(w, xs.max() + 1 + ROI_PAD)
    sub_gc = np.ascontiguousarray(gc[y0:y1, x0:x1])
    cv2.setRNGSeed(int(seed) + 11)  # kmeans 초기화를 프레임마다 고정(다시 돌려도 같은 결과)
    cv2.grabCut(np.ascontiguousarray(img_bgr[y0:y1, x0:x1]), sub_gc, None, np.zeros((1, 65)), np.zeros((1, 65)), ITERS, cv2.GC_INIT_WITH_MASK)
    out = np.zeros((h, w), bool)
    out[y0:y1, x0:x1] = (sub_gc == cv2.GC_FGD) | (sub_gc == cv2.GC_PR_FGD)
    return out


def clean_mask(fg, core, dil, hole_area=80):
    """앞 마스크 정리: 부풀린 사전 마스크 안으로 자르기 → 코어와 이어진 덩어리만 → 가장자리에 닿지 않는 작은 구멍(그림자 얼룩)만 메우기."""
    fg = (fg & dil) | core
    n, lab = cv2.connectedComponents(fg.astype(np.uint8), connectivity=8)
    if n > 2:
        keep = np.unique(lab[core & fg])
        fg = np.isin(lab, keep[keep > 0])
    n, lab, stats, _ = cv2.connectedComponentsWithStats((~fg).astype(np.uint8), connectivity=4)
    h, w = fg.shape
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if area <= hole_area and x > 0 and y > 0 and x + ww < w and y + hh < h:
            fg[lab == i] = True
    return fg


def segment(crop_bgr, pts, size, valid=None, frame_id=0):
    """한 프레임 분할. crop_bgr (SEG, SEG, 3), pts (21, 2) 자르기 좌표, size 손 크기(px). (알파 uint8 (SEG, SEG), 상태, 정보 dict)."""
    shape = crop_bgr.shape[:2]
    if not np.isfinite(pts).all():
        return np.zeros(shape, np.uint8), 2, {}
    prior, core, dil, likely = build_masks(pts, size, valid, shape)
    info = {"prior_area": int(prior.sum())}
    status = 0
    if prior.sum() < 30 or core.sum() < 5:
        status = 1
    else:
        try:
            bg_seed = (color_llr(crop_bgr, core, dil, seed=frame_id) > SEED_T) & dil & ~core
            fg = clean_mask(grabcut_mask(crop_bgr, prior, core, dil, likely, bg_seed, seed=frame_id), core, dil)
            info["ratio"] = int(fg.sum()) / max(1, info["prior_area"])
            if info["ratio"] < MIN_KEEP:
                status = 1
        except cv2.error as e:  # GrabCut이 못 돌았다
            info["error"] = str(e).splitlines()[-1][:80]
            status = 1
    if status == 1:
        fg = ((cv2.erode(prior.astype(np.uint8), ellipse(2)) > 0) | core) & dil
    else:
        fg = ((cv2.GaussianBlur(fg.astype(np.float32), (0, 0), SMOOTH) > 0.5) | core) & dil
    alpha = cv2.GaussianBlur(fg.astype(np.float32), (0, 0), FEATHER) * dil.astype(np.float32)  # 부풀린 사전 마스크 밖은 반드시 0
    if valid is not None:
        alpha *= valid.astype(np.float32)
    return np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8), status, info


def valid_mask(M, frame_shape):
    """자르기 좌표에서 영상 안인 픽셀(불리언 (SEG, SEG)). M = 영상 → 자르기 2×3 행렬."""
    h, w = frame_shape[:2]
    ones = np.full((h, w), 255, np.uint8)
    return cv2.warpAffine(ones, M, (SEG, SEG), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
