"""결과 영상 한 장 그리기(render_video.py가 부른다): 손 영상 위 겹쳐 보기 + 손바닥 틀 3D 두 시점 + 정보 띠 + 오차 시간 띠.

그림 한 장은 960 × 680 RGB다. 글은 PIL, 선은 OpenCV(부분 픽셀, 안티앨리어싱)로 그린다. 손 영상 패널은 2배로 그린 뒤 줄여 선을 부드럽게 한다.
글꼴은 한글이 되는 것을 FONTS에서 찾는다(윈도 맑은 고딕, 리눅스 나눔고딕·Noto Sans CJK, 맥 Apple SD 산돌고딕 Neo). 없으면 set_fonts로 준다.
S(render_video.prepare)에서 쓰는 칸: img_px (N, 21, 2) 영상 픽셀, world (N, 21, 3) 파일 그대로의 world, real·rot·Lm(joints.palm_frame·unit_hand의
입력·회전·길이), unit (N, 21, 3) 단위 손, L_mm 세션 L, n 프레임 수, crop_c·crop_side 자르기, ymax 시간 띠 세로 끝(mm).
"""
import os
import shutil
import subprocess
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # python -I는 3.11부터 -P를 겸해 스크립트 폴더를 경로에 넣지 않는다
import hs_io  # noqa: E402,F401  사용자 설치 numpy 경로도 여기서 붙는다
import joints  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

CW, CH = 960, 680  # 그림 크기(짝수: yuv420p)
SURFACE, INK, INK2, MUTED, GRID, BASE, BLUE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#2a78d6"
GREEN, ORANGE, PURPLE, GRAY, CARD = "#2a9d8f", "#e3a33b", "#6b4fb0", "#898781", "#f1f0ea"
ROLE_COLORS = (GREEN, ORANGE, PURPLE)  # 학습 · 검증 · 시험(dataset.ROLES 순서) — 꼬리표와 시간 띠 아래 역할 띠
HAND_BG = (20, 20, 20)  # 손 패널의 빈 바탕(BGR): 영상 없음·손만 보이는 판의 배경
FONTS = (  # (보통, 굵게)
    (r"C:\Windows\Fonts\malgun.ttf", r"C:\Windows\Fonts\malgunbd.ttf"),
    ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
    ("/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc", "/usr/share/fonts/noto-cjk/NotoSansCJK-Bold.ttc"),
    ("/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc", "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Bold.ttc"),
    ("/System/Library/Fonts/AppleSDGothicNeo.ttc", "/System/Library/Fonts/AppleSDGothicNeo.ttc"),
)


def rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i: i + 2], 16) for i in (0, 2, 4))


def bgr(c):
    return tuple(reversed(rgb(c))) if isinstance(c, str) else tuple(reversed(c))


# ---- 글꼴 -----------------------------------------------------------------------------------------------
_FONT_PATHS = None
_FONTS = {}


def find_fonts():
    """FONTS에서 있는 (보통, 굵게) 경로. 굵은 글꼴이 없으면 보통으로. 하나도 없으면 None."""
    for regular, bold in FONTS:
        if os.path.exists(regular):
            return regular, bold if os.path.exists(bold) else regular
    return None


def set_fonts(regular=None, bold=None):
    """글꼴 파일을 정한다(주지 않으면 find_fonts). 못 찾으면 FileNotFoundError."""
    global _FONT_PATHS
    paths = (regular, bold or regular) if regular else find_fonts()
    if paths is None:
        raise FileNotFoundError("한글 글꼴을 찾지 못함 — --font로 .ttf·.ttc 경로를 준다(예: 우분투 fonts-nanum의 NanumGothic.ttf)")
    _FONT_PATHS = paths
    _FONTS.clear()
    return paths


def font(size, bold=False):
    if _FONT_PATHS is None:
        set_fonts()
    key = (size, bold)
    if key not in _FONTS:
        _FONTS[key] = ImageFont.truetype(_FONT_PATHS[1 if bold else 0], size)
    return _FONTS[key]


def text(draw, xy, s, size=14, color=INK, bold=False, anchor="lm"):
    draw.text(xy, s, font=font(size, bold), fill=rgb(color) if isinstance(color, str) else color, anchor=anchor)


def tlen(draw, s, size=14, bold=False):
    return draw.textlength(s, font=font(size, bold))


def wrap(draw, s, size, width, bold=False):
    """띄어쓰기 사이에서 폭에 맞춰 줄바꿈."""
    lines, cur = [], ""
    for word in s.split(" "):
        trial = (cur + " " + word).strip()
        if tlen(draw, trial, size, bold) <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


# ---- OpenCV 부분 픽셀 그리기 ---------------------------------------------------------------------------
SH = 4


def _p(pt):
    return int(round(pt[0] * (1 << SH))), int(round(pt[1] * (1 << SH)))


def line(img, a, b, color, th):
    cv2.line(img, _p(a), _p(b), color, th, cv2.LINE_AA, SH)


def dot(img, c, r, color):
    cv2.circle(img, _p(c), int(round(r * (1 << SH))), color, -1, cv2.LINE_AA, SH)


EDGES = tuple(joints.BONES) + ((5, 9), (9, 13), (13, 17))  # 뼈 20개 + 손바닥 가로줄


def draw_simple(img, pts, color, th, r):
    for i, j in EDGES:
        line(img, pts[i], pts[j], color, th)
    for k in range(21):
        dot(img, pts[k], r, color)


# ---- 기하 -------------------------------------------------------------------------------------------------
def fit_similarity(src, dst):
    """회전 포함 닮음 변환(반사 없음) dst ≈ A (src − sm) + dm 의 최소제곱. (A 2×2, sm, dm). src·dst (K, 2)."""
    sm, dm = src.mean(0), dst.mean(0)
    za = (src - sm) @ np.array([1.0, 1.0j])
    zb = (dst - dm) @ np.array([1.0, 1.0j])
    c = (np.conj(za) * zb).sum() / (np.abs(za) ** 2).sum()
    a, b = c.real, c.imag
    return np.array([[a, -b], [b, a]]), sm, dm


def pred_world(S, n, unit_pred):
    """예측 (20, 3) 단위 손(손바닥 틀) → 파일 그대로의 world 좌표 (20, 3) m. 그 프레임 정답의 손바닥 회전·손목·|p17 − p0|(프레임별 L)로
    되돌린다(joints.targets의 정확한 역변환. 오른손 세션의 접기도 to_left가 거울과 함께 되돌린다)."""
    coords = unit_pred * S["Lm"][n]
    real = S["rot"][n].T @ coords.T  # (3, 20)
    return joints.to_left(real.T + S["real"][n, 0], S["hand"])


def pred_to_image(S, n, unit_pred):
    """예측을 영상 픽셀 (21, 2)로: 정답 world(x, y) → 영상 닮음 맞춤의 선형 부분 A를 예측 − 정답 world 차이에 적용한다. 곧 예측 영상점 =
    정답 영상점 + A (예측 world − 정답 world)(x, y). 정답 맞춤의 점별 잔차(첫 세션 평균 약 3.8 px)는 모델 탓이 아니라 더하지 않는다.
    예측이 정답과 같으면 정확히 정답 영상점이 된다."""
    gw = S["world"][n]
    A, _, _ = fit_similarity(gw[:, :2], S["img_px"][n])
    pw = pred_world(S, n, unit_pred)
    out = S["img_px"][n].copy()
    out[1:] += (pw[:, :2] - gw[1:, :2]) @ A.T
    return out


def crop_params(px, size, margin=0.25, min_side=110.0, half=2):
    """프레임마다 정사각 자르기 (중심 (N, 2), 한 변 (N,)): 영상 랜드마크 바깥 상자에 margin(한쪽)을 더한다. 손이 없는 프레임은 앞뒤 프레임에서
    선형으로 채우고(하나도 없으면 영상 가운데), 흔들림을 줄이려고 앞뒤 half개 프레임을 삼각 가중 평균한다. size = 영상 (폭, 높이)."""
    ok = np.isfinite(px).all(axis=(1, 2))
    n = len(px)
    if not ok.any():
        return np.tile(np.asarray(size, dtype=np.float64) / 2, (n, 1)), np.full(n, float(min(size)))
    mn, mx = px[ok].min(1), px[ok].max(1)
    c = np.full((n, 2), np.nan)
    side = np.full(n, np.nan)
    c[ok] = (mn + mx) / 2
    side[ok] = np.maximum((mx - mn).max(1) * (1 + 2 * margin), min_side)
    have, idx = np.flatnonzero(ok), np.arange(n)
    c = np.stack([np.interp(idx, have, c[have, d]) for d in range(2)], axis=1)
    side = np.interp(idx, have, side[have])
    w = np.concatenate([np.arange(1, half + 2), np.arange(half, 0, -1)]).astype(float)
    w /= w.sum()

    def smooth(a):
        pad = np.pad(a, ((half, half),) + ((0, 0),) * (a.ndim - 1), mode="edge")
        return sum(w[k] * pad[k: k + len(a)] for k in range(len(w)))

    return smooth(c), smooth(side)


def crop_matrix(c, side, out):
    k = out / side
    return np.array([[k, 0, -k * (c[0] - side / 2)], [0, k, -k * (c[1] - side / 2)]], dtype=np.float64)


def crop_frame(frame_bgr, c, side, out):
    M = crop_matrix(c, side, out)
    img = cv2.warpAffine(frame_bgr, M, (out, out), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(52, 52, 52))
    return img, M


# ---- 장면 구성 --------------------------------------------------------------------------------------------
PA = (40, 84, 320, 320)  # 손 영상 겹쳐 보기(왼쪽 위 x, y, 폭, 높이)
PB = (376, 84, 292, 320)  # 앞쪽(손등 쪽에서 본 모습)
PC = (684, 84, 236, 320)  # 옆쪽(엄지 쪽에서 본 모습)
SS = 2  # 손 영상 패널 초과 표본 배수
K3 = 1.3  # 3D 패널 확대(px/mm), 두 패널 같다
V0 = -14.0  # 패널 아래 끝의 세로 값(mm) — 손바닥 틀 x(손목 → 검지 방향)
H0_B = -125.0  # 앞쪽 패널 왼쪽 끝의 가로 값 h = −y
H0_C = -70.0  # 옆쪽 패널 왼쪽 끝의 가로 값 h = −z
PB_M = (376, 84, 276, 320)  # MANO 판: 정답 메시(카메라에서 본 방향)
PC_M = (668, 84, 276, 320)  # MANO 판: 예측 메시
MESH_SIDE = 274  # 메시 패널 안 정사각 그림(세로 가운데, 손 영상 패널과 같은 자르기)
GT_MESH, PRED_MESH = "#bdb6a8", "#6ea0eb"  # 정답·예측 메시 색(예측은 BLUE를 밝게 — 그늘이 보이게)
STRIP = (16, 462, 928, 104)  # 정보 띠
TL = (16, 576, 928, 92)  # 오차 시간 띠
TP = (TL[0] + 70, TL[1] + 30, TL[2] - 70 - 20, 38)  # 시간 띠의 그림 칸(x, y, 폭, 높이)
BAND = 5  # 그림 칸 아래 역할 띠 두께
TL_GAP_SEC = 0.5  # 이웃 점 간격이 이보다 길면 선을 끊는다


def to_panel(pt_h, pt_v, panel, h0):
    x0, y0, w, h = panel
    return (x0 + (pt_h - h0) * K3, y0 + h - (pt_v - V0) * K3)


def view_points(unit_mm, view):
    """unit_mm (K, 3) 손바닥 틀 mm → (h, v). 앞쪽: h = −y, v = x. 옆쪽: h = −z, v = x."""
    return (-unit_mm[:, 1] if view == "front" else -unit_mm[:, 2]), unit_mm[:, 0]


def panel_points(mm, panel, h0, view):
    hh, vv = view_points(mm, view)
    return np.stack([to_panel(a, b, panel, h0) for a, b in zip(hh, vv)]) - np.array((panel[0] + 1, panel[1] + 1))


def static_background(title, subtitle, notes, ymax, video=True, mean_mm=None, mano=False, gt_name="MediaPipe"):
    """영상 내내 같은 배경(제목·범례·패널 틀·격자·평균 자세·설명·띠 바탕)을 한 번 그린다. RGB numpy. mano면 3D 뼈대 대신 메시 패널 둘.
    gt_name은 범례·메시 제목에 적을 정답 3D 이름(dataset.GT_NAMES)."""
    if mano:
        mean_mm = None  # 메시 판에는 3D 뼈대 패널이 없다
    img = Image.new("RGB", (CW, CH), rgb(SURFACE))
    d = ImageDraw.Draw(img)
    text(d, (16, 24), title, 22, INK, bold=True)
    text(d, (16, 50), subtitle, 13, INK2)
    entries = ((f"{gt_name} 정답", "gt"), ("FastViT 예측", "pred")) + ((("학습 평균 자세", "mean"),) if mean_mm is not None else ())
    widths = [tlen(d, label, 13) + 28 + 10 for label, _ in entries]
    lx = CW - 16 - sum(widths) - 20 * (len(entries) - 1)
    for (label, kind), wd in zip(entries, widths):
        if kind == "gt":
            d.line([(lx, 24), (lx + 28, 24)], fill=rgb(INK), width=4)
            d.line([(lx + 1, 24), (lx + 14, 24)], fill=(255, 255, 255), width=2)
        else:
            d.line([(lx, 24), (lx + 28, 24)], fill=rgb(BLUE if kind == "pred" else BASE), width=4 if kind == "pred" else 2)
        text(d, (lx + 38, 24), label, 13, INK2)
        lx += wd + 20
    text(d, (PA[0], 72), "손 영상 위 겹쳐 보기(근사)" if video else "화면 좌표 뼈대(영상 없음, 근사)", 14, INK, bold=True)
    if mano:
        text(d, (PB_M[0], 72), f"정답 MANO — {gt_name} 관절에 맞춤", 14, INK, bold=True)
        text(d, (PC_M[0], 72), "예측 MANO — FastViT 관절에 맞춤", 14, INK, bold=True)
        for x0, y0, w, h in (PB_M, PC_M):
            d.rectangle([x0, y0, x0 + w, y0 + h], fill=(255, 255, 255), outline=rgb(BASE))
    for panel, h0, view in () if mano else ((PB, H0_B, "front"), (PC, H0_C, "side")):
        x0, y0, w, h = panel
        d.rectangle([x0, y0, x0 + w, y0 + h], fill=(255, 255, 255), outline=rgb(BASE))
        for v in range(0, 251, 50):  # 격자 50 mm
            _, py = to_panel(0, v, panel, h0)
            if y0 < py < y0 + h:
                d.line([(x0, py), (x0 + w, py)], fill=rgb(GRID), width=1)
                text(d, (x0 + 4, py - 8), f"{v}", 10, MUTED)
        for hv in range(-150, 151, 50):
            px, _ = to_panel(hv, 0, panel, h0)
            if x0 < px < x0 + w:
                d.line([(px, y0), (px, y0 + h)], fill=rgb(GRID) if hv != 0 else rgb(BASE), width=1)
        text(d, (x0 + w - 4, y0 + h - 9), "mm", 10, MUTED, anchor="rm")
        text(d, (x0 + 22, y0 + 10), "← 새끼 쪽" if view == "front" else "← 손등 쪽", 11, MUTED)
        text(d, (x0 + w - 6, y0 + 10), "엄지 쪽 →" if view == "front" else "손바닥 쪽 →", 11, MUTED, anchor="rm")
    for k, s in enumerate(wrap(d, notes[0], 11, PA[2])):
        text(d, (PA[0], PA[1] + PA[3] + 12 + 14 * k), s, 11, MUTED)
    for k, s in enumerate(wrap(d, notes[1], 11, PC[0] + PC[2] - PB[0])):
        text(d, (PB[0], PB[1] + PB[3] + 12 + 14 * k), s, 11, MUTED)
    for x, y, w, h in (STRIP, TL):
        d.rounded_rectangle([x, y, x + w, y + h], radius=8, fill=rgb(CARD))
    x, y, w, h = TL  # 시간 띠: 세로 눈금·범례
    text(d, (x + 16, y + 14), "프레임 오차(mm)", 12, MUTED)
    for v in (0.0, ymax / 2, ymax):
        py = TP[1] + TP[3] - v / ymax * TP[3]
        d.line([(TP[0], py), (TP[0] + TP[2], py)], fill=rgb(GRID), width=1)
        text(d, (TP[0] - 6, py), f"{v:g}", 10, MUTED, anchor="rm")
    lx = x + w - 16
    for label, color, kind in (("시험", PURPLE, "band"), ("검증", ORANGE, "band"), ("학습", GREEN, "band"),
                               ("평균 자세 상수", MUTED, "line"), ("FastViT", BLUE, "line")):
        lx -= tlen(d, label, 11)
        text(d, (lx, y + 14), label, 11, INK2)
        if kind == "line":
            d.line([(lx - 26, y + 14), (lx - 6, y + 14)], fill=rgb(color), width=2)
            lx -= 26 + 16
        else:
            d.rectangle([lx - 14, y + 11, lx - 4, y + 17], fill=rgb(color))
            lx -= 14 + 10
    canvas = np.asarray(img).copy()
    if mean_mm is not None:  # 학습 평균 자세(모든 프레임이 같다)
        for panel, h0, view in ((PB, H0_B, "front"), (PC, H0_C, "side")):
            x0, y0, w, h = panel
            draw_simple(canvas[y0 + 1: y0 + h, x0 + 1: x0 + w], panel_points(mean_mm, panel, h0, view), rgb(BASE), 1, 1.8)
    return canvas


def draw_overlay(crop2x, gt_pts, pred_pts):
    """2배 크기 손 영상(BGR) 위에: 예측은 굵은 파랑 띠를 먼저, 정답은 검은 윤곽의 가는 흰 선을 그 위에 얹는다. 둘이 겹치면 파랑 띠 한가운데
    흰 선이 보이고, 어긋나면 흰 선이 파랑 띠 밖으로 나온다."""
    if pred_pts is not None:
        draw_simple(crop2x, pred_pts, bgr(BLUE), 8, 4.8)
    for i, j in EDGES:
        line(crop2x, gt_pts[i], gt_pts[j], (14, 14, 14), 6)
    for k in range(21):
        dot(crop2x, gt_pts[k], 4.8, (14, 14, 14))
    for i, j in EDGES:
        line(crop2x, gt_pts[i], gt_pts[j], (255, 255, 255), 3)
    for k in range(21):
        dot(crop2x, gt_pts[k], 3.0, (255, 255, 255))


def composite_hand(crop2_bgr, alpha_u8):
    """2배 손 패널 위에 분할 알파(분할 해상도 uint8)를 입혀 손만 남기고 나머지는 어두운 바탕으로 바꾼다. 알파는 자르기 좌표 그대로 배수만큼
    키운다(자르기와 같은 좌표 규칙이라 어긋남이 없다)."""
    h, w = crop2_bgr.shape[:2]
    s = h / alpha_u8.shape[0]
    a = cv2.warpAffine(alpha_u8, np.array([[s, 0, 0], [0, s, 0]], dtype=np.float64), (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    a = a.astype(np.float32)[..., None] / 255.0
    out = crop2_bgr.astype(np.float32) * a + np.array(HAND_BG, np.float32) * (1.0 - a)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def runs(mask):
    """참인 구간 [(시작, 끝 포함)]."""
    edges = np.diff(np.r_[0, np.asarray(mask, dtype=np.int8), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))


def draw_timeline(canvas, tl, ymax):
    """시간 띠의 움직이는 부분(RGB 캔버스): 창 [a, b] s 안의 평균 자세·모델 오차 선, 역할 띠, 지금 프레임 커서. 돌려주는 값 (a, b)로 글자를 단다."""
    t, a, b = tl["t"], tl["a"], tl["b"]
    span = max(b - a, 1e-9)
    sel = np.flatnonzero((t >= a) & (t <= b))
    xs = TP[0] + (t[sel] - a) / span * TP[2]
    for values, color, th in ((tl["mean"], rgb(MUTED), 1), (tl["err"], rgb(BLUE), 2)):
        v = values[sel]
        ok = np.isfinite(v)
        breaks = np.r_[False, (np.diff(t[sel]) > TL_GAP_SEC)]
        for s, e in runs(ok):
            for piece in np.split(np.arange(s, e + 1), np.flatnonzero(breaks[s + 1: e + 1]) + 1):
                if len(piece) < 2:
                    continue
                ys = TP[1] + TP[3] - np.minimum(v[piece], ymax) / ymax * TP[3]
                pts = np.round(np.stack([xs[piece], ys], axis=1) * (1 << SH)).astype(np.int32)
                cv2.polylines(canvas, [pts], False, color, th, cv2.LINE_AA, SH)
    role = tl["role"][sel]
    by = TP[1] + TP[3] + 4
    for r, color in enumerate(ROLE_COLORS):
        for s, e in runs(role == r):
            x0 = int(round(xs[s]))
            x1 = int(round(xs[e + 1])) if e + 1 < len(xs) else int(round(xs[e])) + 1
            cv2.rectangle(canvas, (x0, by), (max(x0, x1 - 1), by + BAND - 1), rgb(color), -1)
    cx = TP[0] + (tl["now"] - a) / span * TP[2]
    line(canvas, (cx, TP[1] - 4), (cx, by + BAND + 1), rgb(INK2), 1)
    return a, b


def mesh_image(S, n, verts_mm):
    """손바닥 틀 메시 (V, 3) mm → (영상 픽셀 (V, 2), 깊이 (V,) 영상 px 단위). 손 방향·위치·크기는 그 프레임 정답으로 되돌리고(pred_world와 같은 식),
    world → 영상은 정답 21점의 닮음 맞춤을 그대로 쓴다. 깊이는 정답 world z(클수록 카메라에서 멂)."""
    unit = verts_mm / S["L_mm"]
    w = joints.to_left((S["rot"][n].T @ (unit * S["Lm"][n]).T).T + S["real"][n, 0], S["hand"])
    A, sm, dm = fit_similarity(S["world"][n][:, :2], S["img_px"][n])
    return (w[:, :2] - sm) @ A.T + dm, w[:, 2] * np.sqrt(abs(np.linalg.det(A)))


def draw_mesh(img, pts, depth, faces, color):
    """메시를 칠한다(화가 알고리즘: 먼 면부터). 밝기는 면 법선과 시선의 각(양면). pts (V, 2)·depth (V,)는 같은 px 단위, color는 img와 같은 순서."""
    p3 = np.column_stack([pts, depth])
    a, b, c = p3[faces[:, 0]], p3[faces[:, 1]], p3[faces[:, 2]]
    nrm = np.cross(b - a, c - a)
    shade = 0.30 + 0.70 * np.abs(nrm[:, 2]) / (np.linalg.norm(nrm, axis=1) + 1e-12)
    tri = np.round(pts[faces] * (1 << SH)).astype(np.int32)
    col = np.asarray(color, dtype=np.float64)
    for f in np.argsort(-(a[:, 2] + b[:, 2] + c[:, 2])):
        cv2.fillConvexPoly(img, tri[f], (col * shade[f]).tolist(), cv2.LINE_8, SH)


def mesh_panel(canvas, panel, S, n, verts_mm, faces, color):
    """메시 패널 하나: 손 영상 패널과 같은 자르기로 2배 그린 뒤 줄여 붙인다(verts_mm이 None이면 빈 패널)."""
    x0, y0, w, h = panel
    side = MESH_SIDE * SS
    buf = np.full((side, side, 3), 255, np.uint8)
    if verts_mm is not None:
        px, depth = mesh_image(S, n, verts_mm)
        if np.isfinite(px).all() and np.isfinite(depth).all():
            M = crop_matrix(S["crop_c"][n], S["crop_side"][n], side)
            draw_mesh(buf, px @ M[:, :2].T + M[:, 2], depth * M[0, 0], faces, rgb(color))
    oy = y0 + (h - MESH_SIDE) // 2
    canvas[oy: oy + MESH_SIDE, x0 + 1: x0 + 1 + MESH_SIDE] = cv2.resize(buf, (MESH_SIDE, MESH_SIDE), interpolation=cv2.INTER_AREA)


def render_frame(S, bg, n, frame_bgr, info, seg=None, tl=None, mano=None):
    """한 장(RGB). frame_bgr = 영상 프레임(None이면 영상 없는 판: 어두운 바탕에 화면 좌표 뼈대).
    info: dict(pred=(20, 3)|None, err, prompt, kind, edge, time, tag, tag_color, tag_note, seg_text, meanpose).
    seg: None이면 손 영상 전체. dict(alpha=uint8(분할 해상도)|None)이면 손만 보이는 판 — alpha가 None이면(랜드마크 없음) 원본 프레임을 쓰지 않고
    어두운 빈 패널. tl: 시간 띠(draw_timeline의 dict) 또는 None. mano: None이면 3D 뼈대 두 시점, dict(faces, gt=(정점 mm, 잔차)|None,
    pred=(정점 mm, 잔차)|None)이면 정답·예측 MANO 메시(static_background(mano=True)와 함께)."""
    canvas = bg.copy()
    has_hand = bool(np.isfinite(S["img_px"][n]).all())
    hidden = seg is not None and seg.get("alpha") is None
    size = PA[2] * SS
    if frame_bgr is None or hidden:
        crop2 = np.empty((size, size, 3), np.uint8)
        crop2[:] = HAND_BG
        M = crop_matrix(S["crop_c"][n], S["crop_side"][n], size)
    else:
        crop2, M = crop_frame(frame_bgr, S["crop_c"][n], S["crop_side"][n], size)
        crop2 = (crop2.astype(np.float32) * 0.92).astype(np.uint8)  # 흰 선이 더 보이게 살짝 어둡게
        if seg is not None:
            crop2 = composite_hand(crop2, seg["alpha"])
    if has_hand and not hidden:
        gt_img = S["img_px"][n] @ M[:, :2].T + M[:, 2]
        pred_img = None if info["pred"] is None else pred_to_image(S, n, info["pred"]) @ M[:, :2].T + M[:, 2]
        draw_overlay(crop2, gt_img, pred_img)
    crop = cv2.resize(crop2, (PA[2], PA[3]), interpolation=cv2.INTER_AREA)
    canvas[PA[1]: PA[1] + PA[3], PA[0]: PA[0] + PA[2]] = crop[:, :, ::-1]
    cv2.rectangle(canvas, (PA[0], PA[1]), (PA[0] + PA[2] - 1, PA[1] + PA[3] - 1), rgb(BASE), 1)
    L = S["L_mm"]
    gt_mm = S["unit"][n] * L  # (21, 3) 손목(원점) 포함
    gt_ok = bool(np.isfinite(gt_mm).all())
    pr_mm = None if info["pred"] is None else np.vstack([np.zeros((1, 3)), info["pred"] * L])
    if mano is not None:
        for panel, key, color in ((PB_M, "gt", GT_MESH), (PC_M, "pred", PRED_MESH)):
            mesh_panel(canvas, panel, S, n, None if mano[key] is None else mano[key][0], mano["faces"], color)
    for panel, h0, view in () if mano is not None else ((PB, H0_B, "front"), (PC, H0_C, "side")):
        x0, y0, w, h = panel
        sub = canvas[y0 + 1: y0 + h, x0 + 1: x0 + w]  # 패널 안쪽만 칠한다(바깥은 잘림)
        pg = panel_points(gt_mm, panel, h0, view) if gt_ok else None
        if pr_mm is not None:
            pp = panel_points(pr_mm, panel, h0, view)
            if pg is not None:
                for k in range(1, 21):  # 관절별 오차 선(연한 파랑)
                    line(sub, pg[k], pp[k], (170, 196, 232), 1)
            draw_simple(sub, pp, rgb(BLUE), 2, 2.6)
        if pg is not None:
            draw_simple(sub, pg, (11, 11, 11), 2, 2.8)
    span = draw_timeline(canvas, tl, S["ymax"]) if tl is not None else None
    img = Image.fromarray(canvas)
    d = ImageDraw.Draw(img)
    if not has_hand or hidden:
        text(d, (PA[0] + PA[2] / 2, PA[1] + PA[3] / 2), "손 없음", 24, "#a3a29b", bold=True, anchor="mm")
    if mano is not None:
        for (x0, y0, w, h), key, empty in ((PB_M, "gt", "정답 없음"), (PC_M, "pred", "예측 없음")):
            if mano[key] is None:
                text(d, (x0 + w / 2, y0 + h / 2), empty, 18, "#a3a29b", bold=True, anchor="mm")
            else:
                text(d, (x0 + 8, y0 + h - 11), f"맞춤 잔차 {mano[key][1]:.1f} mm", 11, MUTED)
    x, y, w, h = STRIP
    text(d, (x + 16, y + 20), "동작", 12, MUTED)
    name = info["prompt"]
    text(d, (x + 56, y + 20), name, 20, INK, bold=True)
    nx = x + 56 + tlen(d, name, 20, True) + 12
    if info["kind"]:
        kw = tlen(d, info["kind"], 12) + 16
        d.rounded_rectangle([nx, y + 9, nx + kw, y + 31], radius=11, outline=rgb(BASE), fill=(255, 255, 255))
        text(d, (nx + 8, y + 20), info["kind"], 12, INK2)
        nx += kw + 10
    if info.get("edge"):
        text(d, (nx, y + 20), "동작 시작 직후(0.5 s 안)", 12, MUTED)
    text(d, (x + 16, y + 52), "시간", 12, MUTED)
    text(d, (x + 56, y + 52), f"{info['time']:.2f} s", 17, INK, bold=True)
    text(d, (x + 152, y + 52), f"프레임 {n} / {S['n']}", 12, INK2)
    if info.get("seg_text"):
        text(d, (x + 16, y + 82), info["seg_text"], 13, INK2, bold=True)
    mx = x + 400
    text(d, (mx, y + 52), "MPJPE", 12, MUTED)
    if np.isfinite(info["err"]):
        text(d, (mx + 52, y + 52), f"{info['err']:.1f} mm", 26, INK, bold=True)
    else:
        text(d, (mx + 52, y + 52), "— (예측 없음, 정답만 표시)", 14, MUTED)
    if info.get("meanpose") is not None and np.isfinite(info["meanpose"]):
        text(d, (mx + 52, y + 82), f"참고: 학습 평균 자세 상수로 찍으면 {info['meanpose']:.1f} mm", 11, MUTED)
    tag, col = info["tag"], info["tag_color"]
    tw = tlen(d, tag, 14, True) + 24
    tx = x + w - 16 - tw
    d.rounded_rectangle([tx, y + 14, tx + tw, y + 44], radius=15, fill=rgb(col))
    text(d, (tx + 12, y + 29), tag, 14, "#ffffff", bold=True)
    if info.get("tag_note"):
        text(d, (x + w - 16, y + 62), info["tag_note"], 11, MUTED, anchor="rm")
    if span is not None:
        by = TP[1] + TP[3] + 4 + BAND + 9
        text(d, (TP[0], by), f"{span[0]:.0f} s", 10, MUTED)
        text(d, (TP[0] + TP[2], by), f"{span[1]:.0f} s", 10, MUTED, anchor="rm")
    return np.asarray(img)


def card_image(title, subtitle, big, sub, lines, footer):
    """구간 제목 카드(RGB). lines: 최대 셋(보통 글 · 강조한 숫자 · 참고)."""
    img = Image.new("RGB", (CW, CH), rgb(SURFACE))
    d = ImageDraw.Draw(img)
    text(d, (16, 24), title, 22, INK, bold=True)
    text(d, (16, 50), subtitle, 13, INK2)
    text(d, (CW / 2, 210), big, 52, INK, bold=True, anchor="mm")
    text(d, (CW / 2, 270), sub, 28, INK2, anchor="mm")
    for s, yy, (size, color, bold) in zip(lines, (350, 392, 426), ((18, INK, False), (24, PURPLE, True), (15, MUTED, False))):
        text(d, (CW / 2, yy), s, size, color, bold=bold, anchor="mm")
    text(d, (CW / 2, CH - 70), footer, 16, MUTED, anchor="mm")
    return np.asarray(img)


# ---- 영상 쓰기·읽기 ----------------------------------------------------------------------------------------
def ffmpeg_exe():
    """imageio-ffmpeg의 ffmpeg, 없으면 PATH의 ffmpeg."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        exe = shutil.which("ffmpeg")
        if exe is None:
            raise FileNotFoundError("ffmpeg를 찾지 못함 — pip install imageio-ffmpeg 또는 ffmpeg를 설치")
        return exe


class Mp4Writer:
    """RGB 프레임을 ffmpeg 표준 입력으로 보내 H.264(CRF 18, yuv420p, faststart) mp4를 쓴다."""

    def __init__(self, path, fps, w=CW, h=CH):
        self.path, self.size = path, (h, w, 3)
        cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps:.3f}",
               "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-g", "30", "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.count = 0

    def write(self, rgb_frame):
        assert rgb_frame.shape == self.size and rgb_frame.dtype == np.uint8
        self.p.stdin.write(np.ascontiguousarray(rgb_frame).tobytes())
        self.count += 1

    def close(self):
        self.p.stdin.close()
        rc = self.p.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg 종료 코드 {rc}")


def video_frames(path):
    """영상을 처음부터 차례로 디코드해 (번호, BGR 프레임)을 낸다(가변 프레임률이라 시각으로 찾지 않는다)."""
    cap = cv2.VideoCapture(path)
    n = 0
    try:
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            yield n, fr
            n += 1
    finally:
        cap.release()
