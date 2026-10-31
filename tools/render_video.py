"""학습한 모델의 결과를 영상으로 본다(손 소나): 세션 하나의 프레임마다 정답 손 뼈대와 FastViT 예측을 함께 그린다.

    python -I tools/render_video.py <학습 결과 폴더> --out <출력 폴더> [--data 데이터 폴더] [--session 세션] [--frames test|all]
                                    [--video 영상 파일 | --no-video] [--seg] [--mano MANO_LEFT.pkl] [--check N] [--limit N] [--fps F]
                                    [--font 글꼴 [--font-bold 굵은 글꼴]]

- 학습 결과 폴더: train.py가 쓴 model.pt·dataset.npz·metrics.json. 나누기는 metrics.json의 mode·split(블록 offset, 세션 이름)으로
  train.split_frames를 다시 불러 만들고, 프레임 수가 metrics.json frames와 다르면 멈춘다. 예측은 그 세션의 창을 지금 처리 폴더에서 다시 만들어
  (dataset.session_data) model.pt로 낸다. 다시 만든 프레임·목표가 dataset.npz와 다르면(학습 뒤 처리를 다시 돌림) 멈춘다. 정답은 그 실행의 것이다
  (dataset.npz gt — train.py --gt. hamer면 3D 패널·손 틀이 처리 폴더 hamer.npz를 쓰고, 영상 위 흰 뼈대는 늘 MediaPipe 화면 좌표다).
- --session: 없으면 시험 프레임이 가장 많은 세션.
- --frames test(기본): 그 실행의 시험 프레임. blocks는 시험 블록마다(블록 앞에 제목 카드. 블록 첫 2 s의 비운 프레임은 정답만),
  session·time은 그 세션의 첫 시험 프레임부터 마지막까지, overfit은 세션 전체. all: 세션 전체(꼬리표가 학습·검증·시험 역할을 알린다).
  예측은 test면 시험 프레임에만, all이면 학습 자료에 든 모든 프레임에 그린다.
- 화면(render_draw): 왼쪽 = 손 영상 위 겹쳐 보기(정답 = 검은 윤곽의 흰 선, 예측 = 파란 띠. 예측은 그 프레임 정답의 손 방향·크기·위치에
  손가락 모양만 예측으로 바꾼 근사를 닮음 변환으로 영상 픽셀에 옮긴 것), 가운데·오른쪽 = 손바닥 틀 3D(앞쪽 = 손등 쪽, 옆쪽 = 엄지 쪽에서 본
  모습. 정답 검정 · 예측 파랑 · 학습 평균 자세 회색, 단위 손 × 세션 L mm, 축은 모든 프레임이 같다), 정보 띠 = 동작·시간·MPJPE·꼬리표,
  맨 아래 = 프레임 오차의 시간 띠(FastViT · 학습 평균 자세 상수, 아래 역할 띠, 커서 = 지금 프레임. 구간이 30 s보다 길면 30 s 창이 따라간다).
- 영상: 기본 <data>/phone/sessions/<세션>/video.mp4. landmarks.csv 줄 n = 영상 프레임 n이어야 한다: meta.json에 video가 있으면 ok이고 프레임 수가
  줄 수와 같고 버린 줄이 없을 때, 없으면(공개 데이터 변환 — --video로 원본 mp4를 준다) 줄 n을 영상의 n번째 프레임으로 본다. 기본 영상이 없거나
  맞지 않으면 영상 없이 그린다(--no-video와 같은 판: 왼쪽 패널은 어두운 바탕에 화면 좌표 뼈대). 영상은 가변 프레임률이라 처음부터 차례로 디코드한다.
- --seg: 왼쪽 패널에서 손만 남긴다(hand_seg: 랜드마크 사전 마스크 + GrabCut). 랜드마크가 없는 프레임은 원본을 쓰지 않고 어두운 빈 패널.
- --mano: 3D 뼈대 두 시점 대신 정답·예측 MANO 메시를 카메라에서 본 방향으로 나란히 그린다(mano.py — 모델 파일은 https://mano.is.tue.mpg.de
  에 등록하고 받는다, 저장소에 넣지 않는다). 모양·크기는 세션 정답으로 하나, 손 방향·위치는 두 메시 모두 정답 그대로라 손가락 모양만
  다르다. 프레임마다 따로 맞추고 평활하지 않는다(맞춤 잔차를 패널에 적는다).
- 출력: <out>/<실행>_<세션>_<frames>[_seg][_mano].mp4(H.264 CRF 18, 초당 장 수 = 카메라 프레임 간격 중앙값). --check N이면 영상 대신 고르게 뽑은
  N장과 첫 제목 카드를 <out>/<같은 이름>-check/*.png로. 손·얼굴이 찍힌 영상에서 만들므로 --out이 저장소 안이면 멈춘다.
- 필요한 것: torch·timm(예측), opencv-python, Pillow, imageio-ffmpeg(없으면 PATH의 ffmpeg), 한글 글꼴(render_draw.FONTS 또는 --font).
"""
import argparse
import os
import sys
import time
import types

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # python -I는 3.11부터 -P를 겸해 스크립트 폴더를 경로에 넣지 않는다
import hs_io  # noqa: E402  사용자 설치 numpy 경로도 여기서 붙는다
import dataset  # noqa: E402
import hand_seg  # noqa: E402
import joints  # noqa: E402
import pairs  # noqa: E402
import render_draw as D  # noqa: E402
import wrist  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402

FRAMES = ("test", "all")
CARD_SEC = 1.5  # 구간 제목 카드 길이
TL_WINDOW_SEC = 30.0  # 시간 띠: 구간이 이보다 길면 이 길이의 창이 지금 프레임을 따라간다
NO_VIDEO_SIZE = (360, 480)  # 영상도 meta video도 없을 때 화면 좌표에 곱하는 (폭, 높이): 폰 앱의 세로 3:4
LOG_EVERY = 500
ROLE_TAGS = (("학습 프레임", D.GREEN), ("검증 프레임(멈춤 판단에만 씀)", D.ORANGE), ("시험 프레임(학습에 안 씀)", D.PURPLE))
TITLES = {"blocks": "처음 보는 10 s 덩어리 — 블록 나누기의 시험 블록", "session": "처음 보는 세션 — 세션 나누기의 시험 세션",
          "time": "세션 끝 15% — 시간 나누기의 시험 구간", "overfit": "과적합 점검 — 학습한 프레임 그대로"}
NOTE_VIDEO = ("파란 뼈대는 근사입니다. 손 방향·크기·위치는 정답({gt})을 그대로 쓰고 손가락 모양만 예측으로 바꿨습니다. "
              "예측과 정답의 차이를 닮음 변환으로 영상 픽셀에 옮겼습니다.")
NOTE_NO_VIDEO = ("영상 없이 MediaPipe 화면 좌표로 그린 뼈대입니다. 파란 뼈대는 근사입니다: 손 방향·크기·위치는 정답 그대로이고 "
                 "손가락 모양만 예측입니다.")
NOTE_MANO = ("MANO 왼손을 프레임마다 관절 21점에 맞춘 메시입니다. 모양·크기는 세션 정답으로 하나, 손 방향·위치는 두 메시 모두 정답 그대로라 "
             "손가락 모양만 다릅니다. 평활 없음. 잔차 = 맞춘 MANO 관절과 목표 관절의 평균 거리.")


def note_3d(L):
    return (f"손바닥 틀: 손목이 원점, 세로 = 손목→검지 MCP 방향. 단위 손 × 세션 L {L:.1f} mm. 검정 = 정답, 파랑 = 예측, 회색 = 학습 평균 자세, "
            "연한 파란 선 = 관절별 오차. 축은 모든 프레임이 같습니다.")


def inside(path, root):
    """path가 root 안(root 자신 포함)인가."""
    path, root = os.path.normcase(os.path.abspath(path)), os.path.normcase(os.path.abspath(root))
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:  # 윈도에서 드라이브가 다름
        return False


def run_split(saved, m):
    """metrics.json의 나누기를 train.split_frames로 다시 만든다: (학습, 검증, 시험) 프레임 번호. 프레임 수가 metrics.json과 다르면 ValueError."""
    import train  # torch를 여기서야 불러온다
    split = m.get("split") or {}
    args = types.SimpleNamespace(offset=int(split.get("offset", 0)), test=split.get("test") or None, val=split.get("val") or None)
    parts = train.split_frames(saved, m["mode"], args)[0]
    got = [int(len(p)) for p in parts]
    want = [int(m["frames"][k]) for k in ("train", "val", "test")]
    if got != want:
        raise ValueError(f"다시 만든 나누기(학습·검증·시험 {got}프레임)가 metrics.json frames {want}와 다름 — 그 실행의 dataset.npz·metrics.json인지 볼 것")
    return parts


def pick_session(saved, test, name=None):
    """(세션 번호, 세션마다 시험 프레임 수). name이 없으면 시험 프레임이 가장 많은 세션."""
    names = saved["sessions"].tolist()
    counts = np.bincount(saved["session"][test], minlength=len(names))
    if name is None:
        return int(counts.argmax()), counts
    if name not in names:
        raise ValueError(f"그 실행의 자료에 없는 세션 {name}(쓴 세션: {', '.join(names)})")
    return names.index(name), counts


def run_gt(saved):
    """그 실행 dataset.npz의 손가락 정답 3D(dataset.GT_SOURCES). 이 칸이 없는 옛 실행은 mediapipe다."""
    return str(saved["gt"]) if "gt" in saved else "mediapipe"


def run_row_norm(saved):
    """그 실행 dataset.npz가 줄 단위 정규화로 만든 것인가. 이 칸이 없는 옛 실행은 거짓이다."""
    return bool(saved["row_norm"]) if "row_norm" in saved else False


def run_channels(saved):
    """그 실행 dataset.npz의 입력 채널 묶음(dataset.CHANNEL_SETS). 이 칸이 없는 옛 실행은 base다."""
    return str(saved["channels"]) if "channels" in saved else "base"


def predict_session(data, run, saved, k):
    """(세션 k 학습 자료 프레임의 예측 (M, 20, 3), 목표 (M, 20, 3), 장치). 창은 지금 처리 폴더에서 다시 만든다."""
    import torch
    import train
    name = str(saved["sessions"][k])
    part, lines = dataset.session_data(data, name, gt=run_gt(saved), row_norm=run_row_norm(saved), channels=run_channels(saved))
    if part is None:
        raise ValueError(f"{name}: 학습 자료를 다시 만들지 못함 — " + " / ".join(lines))
    mine = saved["session"] == k
    if not np.array_equal(part["frame_ns"], saved["frame_ns"][mine]) or not np.allclose(part["y"], saved["y"][mine], atol=1e-5):
        raise ValueError(f"{name}: 처리 폴더에서 다시 만든 프레임·목표가 dataset.npz와 다름 — 학습 뒤 처리(profiles·pairs)를 다시 돌렸으면 다시 학습한다")
    ck = torch.load(os.path.join(run, "model.pt"), map_location="cpu", weights_only=True)
    if "bins" in ck and (list(ck["bins"]) != [dataset.BINS.start, dataset.BINS.stop] or int(ck["window"]) != dataset.WINDOW):
        raise ValueError(f"model.pt 입력(칸 {ck['bins']}, 창 {ck['window']}줄)이 지금 dataset(칸 {dataset.BINS.start}–{dataset.BINS.stop}, {dataset.WINDOW}줄)과 다름")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train.make_model(int(ck.get("num_classes", train.FINGER_OUT)), int(ck.get("in_chans", 2)))  # --wrist면 손목 3값이 뒤에, --channels면 입력 채널이 다르다
    model.load_state_dict(ck["state_dict"])
    model.to(device)
    pred = train.predict(model, torch.as_tensor(part["x"], device=device), np.arange(len(part["y"])))[:, :train.FINGER_OUT]
    return pred.reshape(-1, 20, 3).astype(np.float64), part["y"].reshape(-1, 20, 3).astype(np.float64), device


def video_source(args, data, name, n_rows, meta, dropped):
    """(영상 경로 또는 None, 화면 좌표에 곱할 (폭, 높이), 알릴 줄). --video로 준 영상이 맞지 않으면 ValueError."""
    v = meta.get("video") or {}
    size = (int(v["width"]), int(v["height"])) if v.get("width") and v.get("height") else NO_VIDEO_SIZE
    if args.no_video:
        return None, size, "영상 없이(--no-video)"
    path = args.video or os.path.join(hs_io.session_folder(data, name), "video.mp4")
    why, note = None, ""
    if not os.path.exists(path):
        why = f"영상 없음({path})"
    elif meta.get("video") is not None and not (v.get("ok") and v.get("frames") == n_rows and dropped == 0):
        why = f"영상 프레임과 landmarks.csv 줄이 1:1이 아님(meta video ok {v.get('ok')}·프레임 {v.get('frames')}, 줄 {n_rows}, 버린 줄 {dropped})"
    else:
        cap = cv2.VideoCapture(path)
        count, size = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
        if meta.get("video") is None:
            note = f", meta에 video가 없어 줄 n = 영상의 n번째 프레임으로 봄(영상 머리의 프레임 수 {count}, 줄 {n_rows})"
    if why:
        if args.video:
            raise ValueError(why)
        return None, size, why + " — 영상 없이 그린다"
    return path, size, f"영상 {path}({size[0]}×{size[1]}{note})"


def prepare(data, run, m, saved, parts, k, frames, size):
    """세션 프레임(0…N−1) 기준 자료 S: render_draw 머리말의 칸 + 역할·예측·오차·동작·자르기. 돌려주는 두 번째 값은 장치 이름."""
    name = str(saved["sessions"][k])
    folder = hs_io.session_folder(data, name)
    pred_ds, y_ds, device = predict_session(data, run, saved, k)
    marks = hs_io.read_landmarks(folder)
    hand = hs_io.read_session(folder).hand
    n = len(marks.frame_ns)
    S = {"name": name, "hand": hand, "n": n, "frame_ns": marks.frame_ns, "world": marks.world, "tracked": marks.tracked, "valid": marks.valid,
         "score": marks.score, "img_px": marks.image[:, :, :2] * np.asarray(size, dtype=np.float64), "L_mm": float(saved["length_mm"][k]),
         "gt": run_gt(saved)}
    S["good"], _ = joints.keep_frames(marks.world, marks.score, marks.valid)
    if S["gt"] == "hamer":  # 정답 뼈대와 예측을 놓을 손 틀도 그 실행의 정답 3D로(거르기는 dataset처럼 MediaPipe로 하고 HaMeR 관절이 없는 프레임을 더 뺀다)
        S["world"], has = wrist.hamer_points(os.path.join(data, "processed", name, wrist.HAMER_FILE), marks.frame_ns)
        S["good"] &= has
    with np.errstate(invalid="ignore", divide="ignore"):  # 손이 없는 줄(nan → 0)은 nan이 된다
        real = joints.to_left(np.nan_to_num(S["world"], nan=0.0), hand)
        coords, rot = joints.palm_frame(real)
        unit, Lm = joints.unit_hand(coords)
    S.update(real=real, rot=rot, unit=unit, Lm=Lm)
    mine = np.flatnonzero(saved["session"] == k)
    row_of = {int(t): i for i, t in enumerate(marks.frame_ns)}
    pos = np.array([row_of[int(t)] for t in saved["frame_ns"][mine]], dtype=np.int64)  # 학습 자료 프레임 → 세션 프레임
    role_ds = np.full(len(saved["y"]), -1, np.int8)
    for r, idx in enumerate(parts):
        role_ds[idx] = r
    S["role"] = np.full(n, -2, np.int8)  # −2 학습 자료 밖, −1 자료 안이지만 안 씀(경계 뒤 비움), 0 학습, 1 검증, 2 시험
    S["role"][pos] = role_ds[mine]
    show = (role_ds[mine] == 2) if frames == "test" else np.ones(len(mine), dtype=bool)
    L = S["L_mm"]
    mean = joints.mean_pose(saved["y"][parts[0]].reshape(-1, 20, 3))
    S["mean_mm"] = np.vstack([np.zeros((1, 3)), mean * L])
    S["pred"], S["err"], S["mean_err"] = np.full((n, 20, 3), np.nan), np.full(n, np.nan), np.full(n, np.nan)
    S["pred"][pos[show]] = pred_ds[show]
    S["err"][pos[show]] = joints.frame_errors(pred_ds[show], y_ds[show]) * L
    S["mean_err"][pos[show]] = joints.frame_errors(mean[None], y_ds[show]) * L
    S["t"] = (marks.frame_ns - int(saved["frame_ns"][mine].min())) / 1e9  # 학습 자료 첫 프레임부터(블록 나누기와 같은 기준)
    if m["mode"] == "blocks":
        block = dataset.block_assign(saved["frame_ns"], saved["session"], offset=int(m["split"].get("offset", 0)))[0][mine]
        S["block"] = np.where(S["t"] >= 0, np.floor(S["t"] / dataset.BLOCK_SEC), -1).astype(np.int64)
        if not np.array_equal(S["block"][pos], block):
            raise ValueError("세션 프레임의 블록 번호가 dataset.block_assign과 다름")
        S["test_blocks"] = sorted({int(b) for b in block[role_ds[mine] == 2]})
    p = dataset.read_pairs(os.path.join(data, "processed", name, "pairs.csv"))  # 줄마다 landmarks.csv와 같은 프레임(session_data가 확인)
    S["prompt"], S["class_id"], S["edge"] = p["prompt"], p["classId"], p["edge"] == 1
    S["labelled"] = (p["prompt"] >= 0) & (p["edge"] == 0) & (p["classId"] != pairs.PAUSE_CLASS) & (p["flag"] != "wrong")
    S["prompts"] = {q["index"]: q for q in hs_io.read_prompts(folder)}
    S["crop_c"], S["crop_side"] = D.crop_params(S["img_px"], size)
    shown = np.r_[S["err"][np.isfinite(S["err"])], S["mean_err"][np.isfinite(S["mean_err"])]]
    S["ymax"] = float(max(40.0, np.ceil(np.percentile(shown, 99) / 10) * 10)) if len(shown) else 40.0
    return S, device


def plan(S, mode, frames):
    """렌더할 구간 [dict(title, frames(세션 프레임 번호, 오름차순), range(시작·끝 s))] — 구간끼리 겹치지 않고 시각 순서다."""
    t = S["t"]
    if frames == "all" or mode == "overfit":
        return [{"title": "세션 전체", "frames": np.arange(S["n"]), "range": (float(t[0]), float(t[-1]))}]
    test = np.flatnonzero(S["role"] == 2)
    if not len(test):
        raise ValueError(f"세션 {S['name']}에는 시험 프레임이 없음 — 다른 --session을 고르거나 --frames all")
    if mode == "blocks":
        return [{"title": f"시험 블록 {b}", "frames": np.flatnonzero(S["block"] == b), "range": (b * dataset.BLOCK_SEC, (b + 1) * dataset.BLOCK_SEC)}
                for b in S["test_blocks"]]
    return [{"title": "시험 구간", "frames": np.arange(test.min(), test.max() + 1), "range": (float(t[test.min()]), float(t[test.max()]))}]


def kind_name(class_id):
    if class_id == pairs.PAUSE_CLASS:
        return "일시정지"
    if class_id == pairs.NO_CLASS:
        return ""
    return dataset.KINDS[int(dataset.kind_of(np.array([class_id]), np.array([True]))[0])][1]


def prompt_text(S, n):
    """(동작 이름, 종류, 동작 시작 0.5 s 안)."""
    if not S["prompts"]:
        return "동작 라벨 없음", "", False
    k, cid = int(S["prompt"][n]), int(S["class_id"][n])
    if k < 0 or k not in S["prompts"]:
        return "동작 밖(준비·쉬는 때)", "", False
    if cid == pairs.PAUSE_CLASS:
        return "일시정지", "", False
    name, kind = S["prompts"][k]["name"], kind_name(cid)
    return name, "" if kind == name else kind, bool(S["edge"][n])


def excluded_reason(S, n):
    if not S["tracked"][n]:
        return "손 없음"
    if not np.isfinite(S["score"][n]) or S["score"][n] < joints.MIN_SCORE:
        return f"판정 점수 {joints.MIN_SCORE} 아래"
    if not S["valid"][n]:
        return "손이 둘인 줄에서 손목이 튐(다른 손)"
    if not S["good"][n]:
        return f"정답 거르기: 손목–새끼 MCP 길이가 ±{joints.MAD_LIMIT:g}σ 밖"
    return "에코 짝 줄 없음 또는 창의 앞 줄 모자람"


def tag_of(S, n, mode):
    """(꼬리표, 색, 설명)."""
    r = int(S["role"][n])
    if r == -2:
        return "제외(거르기)", D.GRAY, excluded_reason(S, n)
    if r == -1:
        return "제외(경계 뒤 비움)", D.GRAY, f"나누기 경계 뒤 {dataset.GAP_SEC:g} s는 학습·평가에 쓰지 않음(창이 다른 쪽 자료와 겹치지 않게)"
    if mode == "overfit":
        return "학습에 쓴 프레임(과적합 점검)", D.GREEN, ""
    return ROLE_TAGS[r] + ("",)


def main_kinds(S, frames):
    """frames 중 라벨 붙은 프레임의 20% 이상인 동작 종류 이름(많은 순)."""
    lab = frames[S["labelled"][frames]]
    if not len(lab):
        return []
    counts = np.bincount(dataset.kind_of(S["class_id"][lab], np.ones(len(lab), dtype=bool)), minlength=4)[:4]
    return [dataset.KINDS[j][1] for j in np.argsort(-counts, kind="stable") if counts[j] and counts[j] >= 0.2 * len(lab)]


def frame_info(S, n, mode, seg, i, total):
    name, kind, edge = prompt_text(S, n)
    tag, color, note = tag_of(S, n, mode)
    has = bool(np.isfinite(S["err"][n]))
    a, b = seg["range"]
    return {"pred": S["pred"][n] if has else None, "err": S["err"][n], "prompt": name, "kind": kind, "edge": edge, "time": float(S["t"][n]), "tag": tag,
            "tag_color": color, "tag_note": note, "seg_text": f"{seg['title']}  ({round(a):d}–{round(b):d} s)  {i + 1}/{total}",
            "meanpose": S["mean_err"][n] if has else np.nan}


def timeline(S, seg, n):
    """시간 띠 dict(render_draw.draw_timeline): 구간이 TL_WINDOW_SEC 안이면 구간 전체, 길면 지금 프레임을 따라가는 창."""
    fr = seg["frames"]
    t0, t1, now = float(S["t"][fr[0]]), float(S["t"][fr[-1]]), float(S["t"][n])
    if t1 - t0 <= TL_WINDOW_SEC:
        a, b = t0, max(t1, t0 + 1e-3)
    else:
        a = min(max(now - TL_WINDOW_SEC / 2, t0), t1 - TL_WINDOW_SEC)
        b = a + TL_WINDOW_SEC
    return {"t": S["t"][fr], "err": S["err"][fr], "mean": S["mean_err"][fr], "role": S["role"][fr], "a": a, "b": b, "now": now}


def card(S, title, subtitle, seg, i, total):
    fr = seg["frames"]
    has = fr[np.isfinite(S["err"][fr])]
    kinds = main_kinds(S, has)
    a, b = seg["range"]
    first = f"예측을 그린 프레임 {len(has)}개"
    if kinds:
        first += f"  ·  주요 동작: {', '.join(kinds)}"
    elif S["prompts"]:
        first += "  ·  라벨 붙은 동작 없음"
    lines = [first] + ([f"이 구간 평균 MPJPE  {np.mean(S['err'][has]):.1f} mm", f"참고: 학습 평균 자세 상수로 찍으면  {np.mean(S['mean_err'][has]):.1f} mm"]
                       if len(has) else ["이 구간에는 예측을 그린 프레임이 없음"])
    return D.card_image(title, subtitle, seg["title"], f"세션 {round(a):d}–{round(b):d} s", lines, f"{i + 1} / {total}")


def mano_fit(S, path, frames, log=print):
    """그릴 프레임의 MANO 메시 {"faces", "gt": {프레임: (정점 (V, 3) mm, 잔차 mm)}, "pred": {…}}. 모양·크기는 그 세션의 정답(거르기를 지난
    프레임)으로 하나 맞추고, 정답·예측 관절에 프레임마다 손목 회전·자세를 맞춘다(mano 머리말). 정점은 목표와 같은 손바닥 틀(손목 원점)이다."""
    import mano
    import torch
    layer = mano.Layer(mano.load(path), "cuda" if torch.cuda.is_available() else "cpu")
    start = time.time()
    gt = S["unit"] * S["L_mm"]  # (N, 21, 3) 손목(원점) 포함
    ok = np.isfinite(gt).all(axis=(1, 2))
    good = np.flatnonzero(ok & S["good"])
    if not len(good):
        raise ValueError("모양을 맞출 정답 프레임이 없음")
    beta, log_s, res_shape = mano.fit_shape(layer, gt[good])
    frames = np.asarray(frames, dtype=np.int64)
    out = {"faces": layer.faces, "gt": {}, "pred": {}}
    picks = {"gt": frames[ok[frames]], "pred": frames[np.isfinite(S["err"][frames])]}
    targets = {"gt": gt[picks["gt"]], "pred": joints.with_wrist(S["pred"][picks["pred"]]) * S["L_mm"]}
    res = {}
    for key in ("gt", "pred"):
        if not len(picks[key]):
            continue
        orient, pose, res[key] = mano.fit_pose(layer, targets[key], beta, log_s)
        out[key] = {int(f): (v, float(r)) for f, v, r in zip(picks[key], mano.meshes(layer, beta, log_s, orient, pose), res[key])}
    log(f"  MANO 맞춤: 모양 β·크기 {float(np.exp(log_s.item())):.3f}배(정답 {len(res_shape)}장, 잔차 평균 {res_shape.mean():.1f} mm), "
        + ", ".join(f"{'정답' if k == 'gt' else '예측'} {len(r)}장 잔차 평균 {r.mean():.1f} mm(95% {np.percentile(r, 95):.1f})" for k, r in res.items())
        + f", {time.time() - start:.0f} s")
    return out


def seg_alpha(S, n, frame_bgr, floor):
    """손만 보이는 판의 알파(render_draw.render_frame의 seg). 랜드마크가 없으면 alpha None."""
    pts = S["img_px"][n]
    if not np.isfinite(pts).all():
        return {"alpha": None}
    side = S["crop_side"][n]
    crop, M = D.crop_frame(frame_bgr, S["crop_c"][n], side, hand_seg.SEG)
    p = pts @ M[:, :2].T + M[:, 2]
    alpha, status, _ = hand_seg.segment(crop, p, hand_seg.hand_size(p, floor * hand_seg.SEG / side), hand_seg.valid_mask(M, frame_bgr.shape), frame_id=n)
    return {"alpha": alpha, "status": status}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="학습한 모델의 결과(정답·예측 손 뼈대)를 세션 영상 위에 그린다")
    parser.add_argument("run", help="학습 결과 폴더(model.pt·dataset.npz·metrics.json)")
    parser.add_argument("--out", required=True, help="출력 폴더(저장소 밖 — 손·얼굴이 찍힌 영상에서 만든다)")
    parser.add_argument("--data", default=hs_io.DEFAULT_DATA)
    parser.add_argument("--session", help="세션 이름(없으면 시험 프레임이 가장 많은 세션)")
    parser.add_argument("--frames", choices=FRAMES, default="test", help="test: 그 실행의 시험 프레임(기본), all: 세션 전체")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--video", help="영상 파일(없으면 <data>/phone/sessions/<세션>/video.mp4. 공개 데이터 변환은 원본 mp4)")
    source.add_argument("--no-video", action="store_true", help="영상 없이 화면 좌표 뼈대만")
    parser.add_argument("--seg", action="store_true", help="왼쪽 패널에서 손만 남긴다(GrabCut)")
    parser.add_argument("--mano", metavar="MANO_LEFT.pkl", help="3D 뼈대 대신 정답·예측 MANO 메시(모델 파일 또는 mano_v1_2를 푼 폴더 — 저장소 밖)")
    parser.add_argument("--check", type=int, metavar="N", help="영상 대신 고르게 뽑은 N장과 첫 제목 카드를 PNG로")
    parser.add_argument("--limit", type=int, metavar="N", help="앞 N장만 그린다")
    parser.add_argument("--fps", type=float, help="출력 초당 장 수(없으면 카메라 프레임 간격 중앙값)")
    parser.add_argument("--font", help="한글 글꼴 파일(.ttf·.ttc)")
    parser.add_argument("--font-bold", help="굵은 한글 글꼴 파일(없으면 --font)")
    args = parser.parse_args()
    out = os.path.abspath(args.out)
    if inside(out, hs_io.REPO):
        print(f"--out {out}이 저장소 안 — 손·얼굴이 찍힌 영상에서 만든 결과는 저장소 밖에 둔다")
        return 1
    try:
        D.set_fonts(args.font, args.font_bold)
        m = hs_io.read_json(os.path.join(args.run, "metrics.json"))
        saved = dataset.load(os.path.join(args.run, "dataset.npz"))
        parts = run_split(saved, m)
        k, counts = pick_session(saved, parts[2], args.session)
        name = str(saved["sessions"][k])
        folder = hs_io.session_folder(args.data, name)
        meta = hs_io.read_json(os.path.join(folder, "meta.json"))
        marks = hs_io.read_landmarks(folder)
        video, size, how = video_source(args, args.data, name, len(marks.frame_ns), meta, marks.dropped)
        if args.seg and video is None:
            raise ValueError("손만 보이기(--seg)에는 영상이 필요함")
        S, device = prepare(args.data, args.run, m, saved, parts, k, args.frames, size)
        segs = plan(S, m["mode"], args.frames)
    except (ValueError, FileNotFoundError) as e:
        print(f"멈춤: {e}")
        return 1
    mode = m["mode"]
    run_name = os.path.basename(os.path.normpath(args.run))
    print(f"{run_name}: {mode} 나누기, 세션 {name}(시험 프레임 {int(counts[k])}개), {args.frames}, 구간 {len(segs)}개, 예측 장치 {device}")
    print(f"  {how}")
    title = "세션 전체 — 학습·검증·시험 프레임 모두" if args.frames == "all" else TITLES[mode]
    gt_name = dataset.GT_NAMES[S["gt"]]
    subtitle = (f"FastViT-T12 · 실행 {run_name} · 세션 {name}({'왼손' if S['hand'] == 'left' else '오른손'}) · 정답 = {gt_name} 손 관절"
                " · 시간은 학습 자료 첫 프레임부터")
    bg = D.static_background(title, subtitle, (NOTE_VIDEO.format(gt=gt_name) if video else NOTE_NO_VIDEO, NOTE_MANO if args.mano else note_3d(S["L_mm"])),
                             S["ymax"], video=video is not None, mean_mm=S["mean_mm"], mano=bool(args.mano), gt_name=gt_name)
    order = [(i, int(f)) for i, s in enumerate(segs) for f in s["frames"]]
    if args.limit:
        order = order[: args.limit]
    if args.check:
        order = [order[j] for j in sorted({int(round(x)) for x in np.linspace(0, len(order) - 1, args.check)})]
    want = {f: i for i, f in order}
    meshes = None
    if args.mano:
        try:
            meshes = mano_fit(S, args.mano, sorted(want))
        except (ValueError, FileNotFoundError) as e:
            print(f"멈춤: {e}")
            return 1
    first_of = {int(s["frames"][0]): i for i, s in enumerate(segs)}
    fps = args.fps or 1e9 / float(np.median(np.diff(S["frame_ns"])))
    card_frames = max(1, int(round(CARD_SEC * fps)))
    floor = 0.0
    if args.seg:
        sizes = [hand_seg.hand_size(px) for px in S["img_px"] if np.isfinite(px).all()]
        floor = hand_seg.SIZE_FLOOR * float(np.median(sizes)) if sizes else 0.0
    stem = f"{run_name}_{name}_{args.frames}{'_seg' if args.seg else ''}{'_mano' if args.mano else ''}"
    os.makedirs(out, exist_ok=True)
    check_dir = os.path.join(out, stem + "-check")
    if args.check:
        os.makedirs(check_dir, exist_ok=True)
        cv2.imwrite(os.path.join(check_dir, "card.png"), card(S, title, subtitle, segs[order[0][0]], order[0][0], len(segs))[:, :, ::-1])
        writer = None
    else:
        writer = D.Mp4Writer(os.path.join(out, stem + ".mp4"), fps)
    start, done, cards, last = time.time(), 0, 0, max(want)
    frames = D.video_frames(video) if video else ((n, None) for n in range(S["n"]))
    for n, fr in frames:
        if n > last:
            break
        if n not in want:
            continue
        i = want[n]
        if writer is not None and n in first_of:
            img = card(S, title, subtitle, segs[i], i, len(segs))
            for _ in range(card_frames):
                writer.write(img)
            cards += card_frames
        img = D.render_frame(S, bg, n, fr, frame_info(S, n, mode, segs[i], i, len(segs)), seg_alpha(S, n, fr, floor) if args.seg else None,
                             timeline(S, segs[i], n),
                             None if meshes is None else {"faces": meshes["faces"], "gt": meshes["gt"].get(n), "pred": meshes["pred"].get(n)})
        if writer is None:
            cv2.imwrite(os.path.join(check_dir, f"frame_{n:05d}.png"), img[:, :, ::-1])
        else:
            writer.write(img)
        done += 1
        if done % LOG_EVERY == 0:
            print(f"  {done}/{len(want)}장, {time.time() - start:.0f} s", flush=True)
    if writer is not None:
        writer.close()
    if done != len(want):
        print(f"멈춤: 영상이 프레임 {done}장을 그린 뒤 끝남(그릴 프레임 {len(want)}장, 마지막 번호 {last}) — 영상과 landmarks.csv 줄 대응을 볼 것")
        return 1
    shown = [f for _, f in order if np.isfinite(S["err"][f])]
    stats = (f"예측을 그린 프레임 {len(shown)}장 · MPJPE 평균 {np.mean(S['err'][shown]):.1f} mm(학습 평균 자세 상수 {np.mean(S['mean_err'][shown]):.1f} mm)"
             if shown else "예측을 그린 프레임 없음")
    where = check_dir if writer is None else f"{writer.path}({os.path.getsize(writer.path) / 1e6:.1f} MB, {writer.count}장 = 카드 {cards}장 포함, {fps:.2f} fps)"
    print(f"{where}: 세션 프레임 {done}장, {stats}, {time.time() - start:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
