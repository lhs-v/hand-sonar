"""손목 상태 정답: 카메라 프레임마다 워치 기준 손 방향(손 소나, 2026-10-30 손목 상태 학습).

    python -I tools/wrist.py [--data 데이터 폴더] [세션 이름 …]   (보고만 한다. 학습 자료는 dataset.session_data가 이 모듈을 부른다)

목표 u(프레임마다 3값): 손 방향(손목 → 중지 MCP, MediaPipe world를 거울 되돌린 것)을 워치 기기 축으로 적은 단위 벡터. 워치 축은 왼손목에
버튼이 손 쪽일 때 x = 손 쪽(팔뚝 방향), y = 12시 = 새끼 쪽, z = 화면 밖 = 손등 쪽이다. 손을 팔과 일자로 펴면 u ≈ (1, 0, 0).
손등 쪽으로 젖히면 u_z > 0, 새끼 쪽으로 꺾으면 u_y > 0(extension_deviation).
- 워치 자세 R(t)(워치 → 세계, 세계 z = 위): 워치 자이로를 적분하고 중력 센서로 기울기를 바로잡는다(Mahony, KP·KI, 지자기는 쓰지 않음).
  실데이터 8세션(2026-10-28)에서 방위 맞춤이 세션 1/3마다 3–6° 안에서만 흔들렸다(드리프트가 작음).
- 카메라 → 세계 C: 폰은 세워 두고 움직이지 않으므로 상수다(그 세션 폰 자이로 99% 값을 보고에 적는다). 기울기 2자유도는 폰 가속도
  중앙값(세계 위쪽)으로, 방위 1자유도는 맞춤으로 정한다(fit_yaw).
  - 기준 자세(prompts.csv classId −3, flag 없음, 평행 자세 유지 구간)가 있으면 그 프레임에서 손 축 셋(손 방향·새끼 쪽·손등 쪽)을 워치 축
    x·y·z에 맞춘다("calibration").
  - 없으면(이 기능 전 세션) 손가락·ASL 동작(classId 0–14, 동작 시작 0.5 s 뒤) 프레임에서 손의 새끼 쪽 축만 워치 y축에 맞춘다("ulnar").
    손목을 손등·손바닥 쪽으로 꺾어도 그 축은 돌지 않아, 평소 수집 자세의 젖힘(실데이터 약 25°)에 흔들리지 않는다. 옆 꺾임·비틀림은 평균 0으로 본다.
  방위가 틀려도 u에는 cos(손의 앙각)만큼 줄어 들어간다(손을 세운 평소 자세면 앙각 약 70°라 방위 10° 오차가 u에 약 3.4°).
- 카메라 축 K: MediaPipe world는 거울 그림 기준이라 x를 뒤집은 것(joints.to_left)이 K다 — x 카메라가 본 그림 오른쪽, y 아래, z 카메라에서
  멀어지는 쪽. 폰 기기 축과는 화면 회전(meta camera.displayRotation, 없으면 0)으로 잇는다(폰 앱 CalibrationGeometry와 같은 약속).
- 깊이: MediaPipe 3D는 그림 투영에 맞춘 값이라 K의 z(카메라 쪽) 성분을 믿기 어렵다(실데이터: 손목 굽히기에서 깊이 기울기의 부호가 세션마다
  갈림, 그림 평면 성분은 그림 속 길이와 맞음). 그래서 프레임마다 카메라 깊이 축을 워치 축으로 적은 d(3값)도 내고, 학습 손실이 그 성분을
  가볍게 본다(train.py --wrist).
- 쓸 수 있는 프레임(ok): 추적하는 손이 있고, 워치 자이로 표본이 그 시각 ±MAX_GAP_MS 안에 있다. 왼손 세션만 계산한다(오른손은 워치 축이 다르다).
session_wrist(data, name) → ({"u": (M, 3), "depth": (M, 3), "ok": (M,), "method": str}, 보고 줄) 또는 (None, 보고 줄). M = landmarks.csv 프레임 수.
"""
import argparse
import math
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # python -I는 3.11부터 -P를 겸해 스크립트 폴더를 경로에 넣지 않는다
import align  # noqa: E402
import hs_io  # noqa: E402  사용자 설치 numpy 경로도 여기서 붙는다
import joints  # noqa: E402
import numpy as np  # noqa: E402

KP = 1.0  # Mahony 비례 이득(rad/s): 중력으로 기울기를 바로잡는 빠르기
KI = 0.02  # 적분 이득: 자이로 치우침 추정
MAX_GAP_MS = 20.0  # 카메라 프레임에서 가장 가까운 자이로 표본이 이보다 멀면 그 프레임은 쓰지 않는다(표본 간격 10 ms)
EDGE_SEC = 0.5  # 동작 시작 뒤 이 안은 맞춤에 쓰지 않는다(pairs.EDGE_SEC와 같음)
NEUTRAL = -3  # prompts.csv 기준 자세(shared Labels.NEUTRAL_ID)
REF_CLASSES = range(0, 15)  # 기준 자세가 없을 때 방위를 맞출 동작: 손가락 굽히기·ASL 숫자(손목은 그대로 두는 동작)
PHONE_STILL = 0.1  # 폰 자이로 99% 값(rad/s)이 이보다 크면 폰이 움직인 세션이라 알린다(실데이터 0.005–0.017)
MIN_REF = 30  # 방위 맞춤에 쓸 프레임 하한
SCREEN = {0: ((1, 0), (0, 1)), 1: ((0, -1), (1, 0)), 2: ((-1, 0), (0, -1)), 3: ((0, 1), (-1, 0))}  # 화면 회전 → (화면 오른쪽, 화면 위)의 기기 x·y


def mahony(t_ns, gyro, up):
    """워치 자세 (n, 3, 3)(워치 → 세계, 세계 z = 위)와 끝의 자이로 치우침 추정 (3,). gyro (n, 3) rad/s, up (n, 3)은 같은 시각의 중력 센서 값(위쪽).
    첫 자세는 첫 중력을 세계 z에 맞춘 것(방위는 임의)이다. 표본 k의 자이로 값으로 (k − 1, k] 구간을 돌린다."""
    n = len(t_ns)
    out = np.zeros((n, 3, 3))
    if n == 0:
        return out, np.zeros(3)
    g0 = up[0] / np.linalg.norm(up[0])
    axis = np.cross(g0, [0.0, 0.0, 1.0])
    s = float(np.linalg.norm(axis))
    half = math.atan2(s, float(g0[2])) / 2
    w, x, y, z = (1.0, 0.0, 0.0, 0.0) if s < 1e-12 else (math.cos(half), *(axis / s * math.sin(half)))
    bx = by = bz = 0.0
    t = np.asarray(t_ns, dtype=np.float64) * 1e-9
    gyro = np.asarray(gyro, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    for k in range(n):
        if k:
            dt = t[k] - t[k - 1]
            gx, gy, gz = up[k - 1]  # 지금 자세(k − 1 시각)와 같은 시각의 중력으로 견준다. 다음 표본과 견주면 그사이 돈 만큼이 거짓 오차가 된다
            g = math.sqrt(gx * gx + gy * gy + gz * gz)
            if g > 1e-6:
                gx, gy, gz = gx / g, gy / g, gz / g
            # 지금 자세가 예측한 위쪽(워치 축) = R^T (0, 0, 1)
            vx, vy, vz = 2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z
            ex, ey, ez = (gy * vz - gz * vy, gz * vx - gx * vz, gx * vy - gy * vx) if g > 1e-6 else (0.0, 0.0, 0.0)
            bx, by, bz = bx - KI * ex * dt, by - KI * ey * dt, bz - KI * ez * dt
            ox, oy, oz = gyro[k, 0] - bx + KP * ex, gyro[k, 1] - by + KP * ey, gyro[k, 2] - bz + KP * ez
            rate = math.sqrt(ox * ox + oy * oy + oz * oz)
            if rate * dt > 1e-12 and dt > 0:
                h = rate * dt / 2
                c, sn = math.cos(h), math.sin(h) / rate
                qx, qy, qz = ox * sn, oy * sn, oz * sn
                w, x, y, z = (w * c - x * qx - y * qy - z * qz, w * qx + x * c + y * qz - z * qy,
                              w * qy - x * qz + y * c + z * qx, w * qz + x * qy - y * qx + z * c)
                norm = math.sqrt(w * w + x * x + y * y + z * z)
                w, x, y, z = w / norm, x / norm, y / norm, z / norm
        out[k] = ((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
                  (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
                  (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)))
    return out, np.array([bx, by, bz])


def rotation_to_z(v):
    """단위 벡터 v를 (0, 0, 1)로 보내는 가장 작은 회전 (3, 3)."""
    v = np.asarray(v, dtype=np.float64) / np.linalg.norm(v)
    axis = np.cross(v, [0.0, 0.0, 1.0])
    s = float(np.linalg.norm(axis))
    if s < 1e-12:
        return np.eye(3) if v[2] > 0 else np.diag([1.0, -1.0, -1.0])
    return rot_axis(axis / s, math.atan2(s, float(v[2])))


def rot_axis(axis, angle):
    """단위 축 axis 둘레로 angle(rad) 도는 회전 (3, 3)(로드리게스)."""
    a = np.asarray(axis, dtype=np.float64)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def up_camera(accel, display_rotation=0):
    """폰 가속도(기기 축, 가만히 있으면 위쪽)를 카메라 축 K의 단위 위쪽으로."""
    right, up = SCREEN[int(display_rotation) % 4]
    a = np.asarray(accel, dtype=np.float64)
    v = np.array([-(a[0] * right[0] + a[1] * right[1]), -(a[0] * up[0] + a[1] * up[1]), a[2]])
    return v / np.linalg.norm(v)


def hand_axes(points):
    """(M, 21, 3) 실제 왼손 K 좌표 → 손 축 (M, 3, 3). 행 0 = 손 방향(p9 − p0), 행 1 = 새끼 쪽, 행 2 = 손등 쪽(손바닥 법선을 행 0에 직교화).
    joints.palm_frame과 같은 손잡이 약속이고 x만 중지 MCP 쪽이다."""
    p = np.asarray(points, dtype=np.float64)
    x = p[:, 9] - p[:, 0]
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    n = np.cross(p[:, 5] - p[:, 0], p[:, 17] - p[:, 0])
    n -= (n * x).sum(axis=1, keepdims=True) * x
    z = n / np.linalg.norm(n, axis=1, keepdims=True)
    return np.stack([x, np.cross(z, x), z], axis=1)


def fit_yaw(a, b):
    """세계 z 둘레 회전 Rz(ψ)로 Rz(ψ) a_i ≈ b_i(최소제곱)인 ψ(rad). a·b (k, 3). 수평 성분이 거의 없으면 nan."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    s = float((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum())
    c = float((a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]).sum())
    return math.atan2(s, c) if math.hypot(s, c) > 1e-9 * max(len(a), 1) else float("nan")


def yaw_pairs(c0, axes, rot, method):
    """방위 맞춤 짝 (a, b): a = C0 · 손 축(K), b = R · 워치 축. calibration은 세 축 모두, ulnar는 새끼 쪽 축만."""
    which = (0, 1, 2) if method == "calibration" else (1,)
    a = np.concatenate([axes[:, k] @ c0.T for k in which])
    b = np.concatenate([rot[:, :, k] for k in which])
    return a, b


def extension_deviation(u):
    """워치 축 손 방향 u (M, 3) → (손등 쪽 젖힘°, 새끼 쪽 꺾임°). 둘 다 손 방향 x 성분과 이루는 각이다."""
    u = np.asarray(u, dtype=np.float64)
    return np.degrees(np.arctan2(u[:, 2], u[:, 0])), np.degrees(np.arctan2(u[:, 1], u[:, 0]))


def class_of_frames(prompts, t_ns, edge_sec=EDGE_SEC):
    """프레임마다 (그 시각을 담은 동작 classId(없으면 −99), 그 줄 flag, 동작 시작 edge_sec 뒤인가)."""
    cls = np.full(len(t_ns), -99, dtype=np.int64)
    flag = np.full(len(t_ns), "", dtype=object)
    settled = np.zeros(len(t_ns), dtype=bool)
    for p in prompts:
        m = (t_ns >= p["startNs"]) & (t_ns < p["endNs"])
        cls[m] = p["classId"]
        flag[m] = p["flag"]
        settled[m] = t_ns[m] >= p["startNs"] + edge_sec * 1e9
    return cls, flag, settled


def session_wrist(data, name):
    """세션 하나의 손목 정답(머리말). 계산할 수 없으면 (None, 까닭 줄)."""
    folder = hs_io.session_folder(data, name)
    phone = hs_io.read_session(folder)
    if phone.hand != "left":
        return None, [f"{name}: 손목 정답 없음(왼손 세션만 계산 — 오른손은 워치 축이 다름)"]
    info, al = align.load(data, name)
    if al is None:
        return None, [f"{name}: 손목 정답 없음(워치와 맞추지 못한 세션)"]
    watch = hs_io.read_sensors(al.recording.folder)
    if watch is None or not {"gyro", "grav"} <= set(watch.tracks):
        return None, [f"{name}: 손목 정답 없음(워치 녹음 {info['recording']}에 자이로·중력 기록이 없음)"]
    phone_sensors = hs_io.read_sensors(folder, hs_io.PHONE_SENSORS)
    if phone_sensors is None or "acc" not in phone_sensors.tracks or not len(phone_sensors.tracks["acc"].t_ns):
        return None, [f"{name}: 손목 정답 없음(폰 가속도 기록이 없어 카메라 기울기를 모름)"]
    marks = hs_io.read_landmarks(folder)
    prompts = hs_io.read_prompts(folder)
    rotation = (phone.meta.get("camera") or {}).get("displayRotation")
    accel = np.median(phone_sensors.tracks["acc"].values, axis=0)
    still = float(np.percentile(np.linalg.norm(phone_sensors.tracks["gyro"].values, axis=1), 99)) if "gyro" in phone_sensors.tracks else float("nan")
    # 워치 시각 → 워치 녹음 프레임 → 폰 시각
    t_gyro = al.phone_ns(al.recording.clock.frames(watch.ns("gyro")))
    t_grav = al.phone_ns(al.recording.clock.frames(watch.ns("grav")))
    grav = watch.tracks["grav"].values
    up = np.stack([np.interp(t_gyro, t_grav, grav[:, k]) for k in range(3)], axis=1)
    rot, bias = mahony(t_gyro, watch.tracks["gyro"].values, up)
    j = np.clip(np.searchsorted(t_gyro, marks.frame_ns), 1, len(t_gyro) - 1)
    j = np.where(np.abs(t_gyro[j - 1] - marks.frame_ns) <= np.abs(t_gyro[j] - marks.frame_ns), j - 1, j)
    near = np.abs(t_gyro[j] - marks.frame_ns) <= MAX_GAP_MS * 1e6
    ok = marks.tracked & near
    m = len(marks.frame_ns)
    axes = np.full((m, 3, 3), np.nan)
    if ok.any():
        axes[ok] = hand_axes(joints.to_left(marks.world[ok], "left"))
    rot_f = rot[j]
    up_k = up_camera(accel, rotation or 0)
    c0 = rotation_to_z(up_k)
    cls, flag, settled = class_of_frames(prompts, marks.frame_ns)
    calib = ok & (cls == NEUTRAL) & (flag == "")
    method = "calibration" if calib.sum() >= MIN_REF else "ulnar"
    ref = calib if method == "calibration" else ok & settled & np.isin(cls, list(REF_CLASSES))
    if ref.sum() < MIN_REF:
        return None, [f"{name}: 손목 정답 없음(방위 맞춤에 쓸 프레임 {int(ref.sum())}개 — {MIN_REF}개 아래)"]
    psi = fit_yaw(*yaw_pairs(c0, axes[ref], rot_f[ref], method))
    if not math.isfinite(psi):
        return None, [f"{name}: 손목 정답 없음(방위 맞춤 축이 거의 수직이라 방위를 못 정함)"]
    cam = rot_axis([0.0, 0.0, 1.0], psi) @ c0  # 카메라 → 세계
    to_watch = np.einsum("mba,bc->mac", rot_f, cam)  # 카메라 → 워치 (M, 3, 3)
    u = np.einsum("mac,mc->ma", to_watch, np.nan_to_num(axes[:, 0]))
    depth = to_watch[:, :, 2]
    u[~ok] = 0.0
    # 보고: 방위가 세션 동안 흔들렸는지(맞춤 프레임 셋으로 나눠), 기준 자세에서 손이 팔과 얼마나 맞았는지, 동작 종류별 젖힘·꺾임
    idx = np.flatnonzero(ref)
    thirds = [math.degrees(fit_yaw(*yaw_pairs(c0, axes[s], rot_f[s], method))) for s in np.array_split(idx, 3) if len(s)]
    ext, dev = extension_deviation(u)
    lines = [f"{name}: 손목 정답 {int(ok.sum())}/{m}프레임, 방위 맞춤 {method}({int(ref.sum())}프레임) {math.degrees(psi):.1f}°"
             f"(1/3씩 {', '.join(f'{v:.1f}' for v in thirds)}), 폰 뒤로 기욺 {math.degrees(math.acos(min(1.0, -up_k[1]))):.1f}°"
             f"·자이로 99% {still:.3f} rad/s, 워치 자이로 치우침 {np.round(bias, 4).tolist()} rad/s"]
    if still > PHONE_STILL:
        lines.append(f"  폰이 움직였다(자이로 99% {still:.3f} > {PHONE_STILL} rad/s) — 카메라 기울기를 상수로 본 가정이 틀릴 수 있다")
    if calib.any():
        hand_elev = np.degrees(np.arcsin(np.clip(axes[calib, 0] @ cam.T @ np.array([0, 0, 1.0]), -1, 1)))
        arm_elev = np.degrees(np.arcsin(np.clip(np.abs(rot_f[calib][:, 2, 0]), -1, 1)))
        lines.append(f"  기준 자세 {int(calib.sum())}프레임: 손 앙각 {np.median(hand_elev):.1f}°(카메라·폰 중력) · 팔뚝 앙각 {np.median(arm_elev):.1f}°(워치),"
                     f" 젖힘 {np.median(ext[calib]):.1f}° · 꺾임 {np.median(dev[calib]):.1f}°")
    for c, label in ((10, "숫자 5"), (15, "손목 젖히기"), (16, "손목 굽히기"), (17, "새끼 쪽 꺾기"), (-1, "자유 동작")):
        sel = ok & settled & (cls == c)
        if sel.sum() >= 10:
            lines.append(f"  {label}: 젖힘 {np.median(ext[sel]):.1f}° [{np.percentile(ext[sel], 10):.1f}, {np.percentile(ext[sel], 90):.1f}]"
                         f" · 꺾임 {np.median(dev[sel]):.1f}° [{np.percentile(dev[sel], 10):.1f}, {np.percentile(dev[sel], 90):.1f}]")
    return {"u": u.astype(np.float32), "depth": depth.astype(np.float32), "ok": ok, "method": method}, lines


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="세션마다 손목 상태 정답(워치 기준 손 방향)을 계산해 보고한다")
    parser.add_argument("sessions", nargs="*", help="폰 세션 이름(없으면 쓸 수 있는 세션 모두)")
    parser.add_argument("--data", default=hs_io.DEFAULT_DATA)
    args = parser.parse_args()
    found = False
    for name, why in hs_io.chosen(args.data, args.sessions):
        if why:
            print(f"{name}: 건너뜀({why})")
            continue
        try:
            got, lines = session_wrist(args.data, name)
        except Exception as e:  # noqa: BLE001  한 세션이 넘어져도 다른 세션은 본다
            got, lines = None, [f"{name}: 처리 중 오류({type(e).__name__}: {e})"]
        print("\n".join(lines))
        found |= got is not None
    return 0 if found else 1


if __name__ == "__main__":
    sys.exit(main())
