"""세션 영상에 HaMeR(단안 3D 손 복원, arXiv 2312.05251)를 돌려 프레임마다 3D 관절 21개를 처리 폴더의 hamer.npz로 쓴다(손목 정답용, 2026-10-31).

    python -I tools/hamer_keypoints.py [--hamer HaMeR 폴더] [--data 데이터 폴더] [--force] [세션 …]
    python -I tools/hamer_keypoints.py [--hamer HaMeR 폴더] --setup --mano <MANO_RIGHT.pkl>   (처음 한 번: 모델 받기·MANO 바꾸기)

- 왜: 손목 정답(wrist.py)의 손등·손바닥 쪽 꺾임은 정면 카메라에서 깊이 방향이다. 손목 굽히기를 숫자 5와 견주면 MediaPipe 3D는 10세션 중 5개만
  바른 방향이었고, HaMeR는 10개 모두 맞혔다(그림 평면의 옆 꺾임은 둘이 같음, 보고서 hand-sonar/2026-10-31-hamer-wrist-check). wrist.py는
  hamer.npz가 있으면 그 3D를 쓴다.
- 입력: 세션 영상(video.mp4, 거울 그림)과 landmarks.csv. 영상 n번째 프레임 = 관절 줄 n이어야 한다(meta video.ok, 프레임 수 같음 — 아니면
  건너뜀). 손 상자는 그 프레임의 MediaPipe 그림 관절점의 최소·최대이고(hand_box), HaMeR가 2배로 넓혀 자른다. 거울 그림에서는 실제 왼손이
  오른손으로 보이므로 오른손(right=1)으로 넣는다. 그러면 결과 관절이 MediaPipe world와 같은 축(거울 그림 기준 카메라 축: x 그림 오른쪽, y 아래,
  z 카메라에서 멀어짐)이다(실데이터 그림 평면 방향 차 중앙 2°). 왼손 세션만 돌린다(wrist.py도 왼손만).
- 출력: processed/<세션>/hamer.npz — frame_ns (M,) int64, keypoints (M, 21, 3) float32(MediaPipe 점 순서, 미터, 손이 없던 프레임은 nan),
  global_orient (M, 3, 3) float32. 영상에서 얼굴이 찍혀 있으므로 PC 안에서만 돌린다.
- HaMeR 설치(저장소 밖, 기본은 저장소 옆 ../hamer): github.com/geopavlakos/hamer(MIT)를 받고 --setup으로 데모 묶음(6.04 GB)을 받으면서
  hamer.ckpt(2.69 GB)와 설정만 꺼낸다(ViTPose 가중치는 건너뛰고 묶음 자체는 디스크에 두지 않음). MANO 오른손(비상업 라이선스, 사용자가
  mano.is.tue.mpg.de에서 받음)은 mano.py의 안전한 읽기로 chumpy 없는 피클로 바꿔 _DATA/data/mano/에 둔다. 추가 패키지: smplx==0.1.28, yacs,
  einops, pytorch-lightning, webdataset. 렌더러(pyrender·EGL)는 쓰지 않으므로 빈 모듈로 막고 init_renderer=False로 불러온다.
- 속도: RTX 5070, fp16, 묶음 32에서 세션 하나(약 4300프레임)에 약 40 s.
"""
import argparse
import os
import pickle
import sys
import tarfile
import time
import types
import urllib.request
import warnings

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # python -I는 3.11부터 -P를 겸해 스크립트 폴더를 경로에 넣지 않는다
import hs_io  # noqa: E402  사용자 설치 numpy·torch 경로도 여기서 붙는다
import numpy as np  # noqa: E402

DEFAULT_HAMER = os.path.join(os.path.dirname(hs_io.REPO), "hamer")  # 저장소 옆 HaMeR 폴더
DEMO_URL = "https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz"
CKPT = os.path.join("_DATA", "hamer_ckpts", "checkpoints", "hamer.ckpt")
MANO_DEST = os.path.join("_DATA", "data", "mano", "MANO_RIGHT.pkl")
MANO_KEYS = ("hands_components", "hands_mean", "shapedirs", "v_template", "J_regressor", "posedirs", "kintree_table", "weights", "f")
OUTPUT = "hamer.npz"
BATCH = 32
RESCALE = 2.0  # 손 상자를 넓히는 배율(HaMeR 데모의 키포인트 상자와 같은 뜻)


def hand_box(image_xy, width, height):
    """그림 관절점 (21, 2)(0..1)을 픽셀 상자 [[x1, y1, x2, y2]] (1, 4) float32로."""
    p = np.asarray(image_xy, dtype=np.float64) * [width, height]
    return np.array([[p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()]], dtype=np.float32)


def fetch(hamer_dir, log=print):
    """데모 묶음을 받으면서 HaMeR 체크포인트·설정만 꺼낸다(ViTPose 가중치·숨김 파일은 건너뜀, 묶음은 디스크에 두지 않는다)."""
    with urllib.request.urlopen(DEMO_URL) as resp, tarfile.open(fileobj=resp, mode="r|gz") as tf:
        for m in tf:
            if "vitpose" in m.name.lower() or os.path.basename(m.name).startswith("."):
                continue
            tf.extract(m, hamer_dir, filter="data")
            if m.isfile():
                log(f"  꺼냄 {m.name}({m.size / 1e6:.1f} MB)")


def setup_mano(mano_path, hamer_dir):
    """MANO_RIGHT.pkl(chumpy 객체가 든 원본)을 chumpy 없는 numpy 피클로 바꿔 HaMeR가 읽는 자리에 둔다."""
    import mano  # 안전한 읽기(numpy·scipy 희소·chumpy 자리만 허용)
    with open(mano_path, "rb") as f, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = mano._Unpickler(f, encoding="latin1").load()
    out = {k: np.ascontiguousarray(mano._array(raw[k])) for k in MANO_KEYS}
    out["f"] = out["f"].astype(np.int64)
    out["kintree_table"] = out["kintree_table"].astype(np.int64)
    if out["v_template"].shape != (778, 3) or out["hands_components"].shape != (45, 45):
        raise ValueError(f"MANO 오른손 모양이 다름: v_template {out['v_template'].shape}, hands_components {out['hands_components'].shape}")
    dest = os.path.join(hamer_dir, MANO_DEST)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        pickle.dump(out, f, protocol=4)
    return dest


def _stub_attr(attr):
    """빈 렌더러 모듈: 타입 표기(pyrender.Node 등)만 통과시키고 모듈 속성(__file__ 등)은 없다고 답한다."""
    if attr.startswith("__"):
        raise AttributeError(attr)
    return type(attr, (), {})


def load(hamer_dir):
    """(모델, 설정). HaMeR 설정의 ./_DATA 상대 경로 때문에 작업 폴더를 HaMeR 폴더로 옮긴다(부르는 쪽은 절대 경로를 쓴다)."""
    for stub in ("pyrender", "trimesh"):
        module = types.ModuleType(stub)
        module.__getattr__ = _stub_attr
        sys.modules[stub] = module  # 진짜 pyrender가 깔려 있어도 막는다(HaMeR 렌더러가 EGL을 고른다)
    sys.path.insert(0, hamer_dir)
    os.chdir(hamer_dir)
    from hamer.configs import get_config
    from hamer.models import HAMER
    cfg = get_config(os.path.join(os.path.dirname(os.path.dirname(CKPT)), "model_config.yaml"), update_cachedir=True)
    cfg.defrost()
    if cfg.MODEL.BACKBONE.TYPE == "vit" and "BBOX_SHAPE" not in cfg.MODEL:
        cfg.MODEL.BBOX_SHAPE = [192, 256]  # HaMeR load_hamer와 같다
    if "PRETRAINED_WEIGHTS" in cfg.MODEL.BACKBONE:
        cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
    cfg.freeze()
    model = HAMER.load_from_checkpoint(CKPT, strict=False, cfg=cfg, init_renderer=False, weights_only=False)
    return model.eval(), cfg


def run_session(data, name, model, cfg, device="cuda"):
    """세션 하나 → processed/<name>/hamer.npz. (쓴 경로 또는 None, 보고 줄)."""
    import cv2
    import torch
    from hamer.datasets.vitdet_dataset import ViTDetDataset
    folder = hs_io.session_folder(data, name)
    phone = hs_io.read_session(folder)
    if phone.hand != "left":
        return None, f"{name}: 건너뜀(왼손 세션만)"
    marks = hs_io.read_landmarks(folder)
    video = phone.meta.get("video") or {}
    path = os.path.join(folder, video.get("file") or "video.mp4")
    if not video.get("ok") or int(video.get("frames", -1)) != len(marks.frame_ns) or marks.dropped or not os.path.exists(path):
        return None, f"{name}: 건너뜀(영상 n = 관절 줄 n이 아님 — video {video}, 줄 {len(marks.frame_ns)}, 버린 줄 {marks.dropped})"
    keypoints = np.full((len(marks.frame_ns), 21, 3), np.nan, dtype=np.float32)
    orient = np.full((len(marks.frame_ns), 3, 3), np.nan, dtype=np.float32)
    pending, t0 = [], time.time()
    right = np.array([1.0], dtype=np.float32)

    def flush():
        if not pending:
            return
        batch = torch.utils.data.default_collate([item for _, item in pending])
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        with torch.no_grad(), torch.autocast(device, dtype=torch.float16):
            out = model(batch)
        j = out["pred_keypoints_3d"].float().cpu().numpy()
        r = out["pred_mano_params"]["global_orient"].float().cpu().numpy().reshape(-1, 3, 3)
        for (i, _), jj, rr in zip(pending, j, r):
            keypoints[i], orient[i] = jj, rr
        pending.clear()

    cap = cv2.VideoCapture(path)
    try:
        for i in range(len(marks.frame_ns)):
            ok, frame = cap.read()
            if not ok:
                return None, f"{name}: 건너뜀(영상 {i}번 프레임을 못 읽음)"
            if not marks.tracked[i]:
                continue
            ds = ViTDetDataset(cfg, frame, hand_box(marks.image[i, :, :2], frame.shape[1], frame.shape[0]), right, rescale_factor=RESCALE)
            pending.append((i, ds[0]))
            if len(pending) == BATCH:
                flush()
        flush()
    finally:
        cap.release()
    out_path = os.path.join(data, "processed", name, OUTPUT)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path[: -len(".npz")] + ".tmp.npz"
    np.savez(tmp, frame_ns=marks.frame_ns, keypoints=keypoints, global_orient=orient)
    os.replace(tmp, out_path)
    done = int(np.isfinite(keypoints[:, 0, 0]).sum())
    return out_path, f"{name}: HaMeR {done}/{len(marks.frame_ns)}프레임, {time.time() - t0:.0f} s"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="세션 영상에 HaMeR를 돌려 손 3D 관절을 hamer.npz로 쓴다(손목 정답용)")
    parser.add_argument("sessions", nargs="*", help="폰 세션 이름(없으면 쓸 수 있는 세션 모두)")
    parser.add_argument("--data", default=hs_io.DEFAULT_DATA)
    parser.add_argument("--hamer", default=DEFAULT_HAMER, help="HaMeR 폴더(github.com/geopavlakos/hamer를 받은 곳)")
    parser.add_argument("--force", action="store_true", help="hamer.npz가 있어도 다시 만든다")
    parser.add_argument("--setup", action="store_true", help="처음 한 번: 모델 묶음을 받아 필요한 파일만 꺼내고(--mano가 있으면) MANO를 바꿔 둔다")
    parser.add_argument("--mano", help="--setup과 함께: MANO_RIGHT.pkl 경로(mano_v1_2/models/)")
    args = parser.parse_args()
    data, hamer_dir = os.path.abspath(args.data), os.path.abspath(args.hamer)
    if not os.path.isdir(os.path.join(hamer_dir, "hamer")):
        print(f"{hamer_dir}에 HaMeR 코드가 없음 — git clone https://github.com/geopavlakos/hamer {hamer_dir}")
        return 1
    if args.setup:
        if not os.path.exists(os.path.join(hamer_dir, CKPT)):
            print(f"모델 묶음 받는 중(6.04 GB, 꺼내 두는 것은 약 2.7 GB): {DEMO_URL}")
            fetch(hamer_dir)
        if args.mano:
            print(f"MANO 바꿈: {setup_mano(os.path.abspath(args.mano), hamer_dir)}")
        return 0
    missing = [p for p in (CKPT, MANO_DEST) if not os.path.exists(os.path.join(hamer_dir, p))]
    if missing:
        print(f"HaMeR 파일이 없음: {missing} — 먼저 --setup --mano <MANO_RIGHT.pkl>")
        return 1
    model, cfg = load(hamer_dir)
    model = model.to("cuda")
    for name, why in hs_io.chosen(data, args.sessions):
        if why:
            print(f"{name}: 건너뜀({why})")
            continue
        if not args.force and os.path.exists(os.path.join(data, "processed", name, OUTPUT)):
            print(f"{name}: {OUTPUT}가 이미 있음(--force면 다시)")
            continue
        try:
            _, line = run_session(data, name, model, cfg)
        except Exception as e:  # noqa: BLE001  한 세션이 넘어져도 다른 세션은 돌린다
            line = f"{name}: 처리 중 오류({type(e).__name__}: {e})"
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
