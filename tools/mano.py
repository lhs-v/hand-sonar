"""MANO 손 모델(Romero 외 2017)을 관절 21점(MediaPipe 순서)에 맞추고 메시를 낸다(render_video.py --mano).

모델 파일은 https://mano.is.tue.mpg.de 에 등록하고 받는 mano_v1_2.zip의 models/MANO_LEFT.pkl이다. 비상업 연구용 라이선스라 저장소에 넣지 않는다.
- 읽기(load): chumpy 없이 읽는다. pkl 안의 chumpy 배열은 상태에 든 ndarray(x)만 꺼낸다. 우리 목표는 모두 왼손으로 접혀 있어(joints.to_left)
  늘 MANO_LEFT를 쓴다. 손끝 정점 번호가 그 손가락 끝마디 관절에 붙어 있는지 읽을 때 확인한다.
- 앞으로 계산(Layer, 선형 블렌드 스키닝): 모양 β(10) → 정점 v = v_template + shapedirs·β, 관절 J = J_regressor·v(16점). 자세는 손목 회전 3값과
  관절 15개의 축-각 45값(hands_mean을 더하지 않은 전체 값). 자세 보정 posedirs·(R − I)를 더하고, 운동학 사슬(kintree_table)로 관절 변환을
  쌓아 정점 가중치(weights)로 섞는다. 길이는 m.
- 관절 21점: MANO 관절 16개 + 손끝 정점 5개(TIPS, 흔히 쓰는 번호)를 MediaPipe 순서로 놓는다(TO_MEDIAPIPE).
- 맞춤: 손목이 원점인 관절 (N, 21, 3) mm에 맞춘다. 모양 β와 크기 s는 세션에 하나(fit_shape, 정답 프레임으로), 손목 회전과 자세는 프레임마다
  (fit_pose) Adam으로 맞춘다. 손목 회전 초깃값은 손바닥 점의 Kabsch 회전이고, 자세는 hands_mean에서 시작해 그 근처로 묶는다(L2 — 관절 위치로는
  보이지 않는 뼈축 비틀림을 정한다). 잔차 = 맞춘 MANO 관절 21점과 목표의 평균 거리(mm).
"""
import os
import pickle
import sys
import warnings

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # python -I는 3.11부터 -P를 겸해 스크립트 폴더를 경로에 넣지 않는다
import hs_io  # noqa: E402,F401  사용자 설치 numpy·scipy·torch 경로도 여기서 붙는다
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

TIPS = (745, 317, 444, 556, 673)  # 손끝 정점: 엄지 · 검지 · 중지 · 약지 · 새끼
TIP_JOINTS = (15, 3, 6, 12, 9)  # 그 손가락의 끝마디 관절(MANO 번호): 손끝 정점의 가중치가 가장 큰 관절이어야 한다
TO_MEDIAPIPE = (0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20)  # [MANO 관절 16개, 손끝 5개] → MediaPipe 21점
PALM = (5, 9, 13, 17)  # 손목 회전 초깃값에 쓰는 손바닥 점(MediaPipe 번호, 손목은 원점이라 뺌)
POSE_WEIGHT = 0.5  # 자세를 hands_mean 근처로 묶는 L2(mm² / rad², 45값 합)
SHAPE_WEIGHT = 0.5  # 모양 β의 L2(mm², 10값 합)
SHAPE_FRAMES = 300  # 모양 β·크기를 맞출 때 쓰는 정답 프레임 수(고르게 뽑음)
SHAPE_STEPS, POSE_STEPS = 400, 400  # Adam 걸음 수
LR = 0.02
CHUNK = 4096  # 한 번에 맞추는 프레임 수


class _Stub:
    """chumpy 객체 자리: 상태(dict)만 담는다."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.state = state


BASIC = {("copy_reg", "_reconstructor"), ("copyreg", "_reconstructor"), ("_codecs", "encode")}  # _codecs.encode: 파이썬 3가 protocol 2로 바이트를 담는 방식
BASIC |= {(m, n) for m in ("__builtin__", "builtins") for n in ("object", "set", "frozenset")}  # chumpy 상태에 set이 든다
NUMPY_NAMES = ("ndarray", "dtype", "_reconstruct", "scalar")
SPARSE_NAMES = ("csc_matrix", "csr_matrix", "coo_matrix")


class _Unpickler(pickle.Unpickler):
    """MANO pkl에 드는 것(numpy 배열·scipy 희소 행렬·chumpy 배열)만 만든다. 그 밖의 객체는 거부한다(pickle은 파일 안의 함수를 부를 수 있다)."""

    def find_class(self, module, name):
        top = module.split(".")[0]
        if top == "chumpy":
            return _Stub
        if not ((module, name) in BASIC or (top == "numpy" and name in NUMPY_NAMES) or (module.startswith("scipy.sparse") and name in SPARSE_NAMES)):
            raise pickle.UnpicklingError(f"MANO pkl에 허용하지 않은 객체 {module}.{name}")
        return super().find_class(module, name)


def _array(v):
    """pkl 값 → ndarray. chumpy 배열(Ch)이면 상태의 x, chumpy 고르기(Select — MANO의 shapedirs는 (778, 3, 20)에서 앞 10성분을 고른 것)면
    a를 펴서 idxs를 고른 뒤 preferred_shape로. scipy 희소 행렬이면 밀집."""
    if isinstance(v, _Stub):
        state = v.state if isinstance(v.state, dict) else {}
        if isinstance(state.get("x"), np.ndarray):
            return state["x"]
        if "a" in state and isinstance(state.get("idxs"), np.ndarray):
            out = _array(state["a"]).ravel()[state["idxs"]]
            return out.reshape(state["preferred_shape"]) if state.get("preferred_shape") else out
        raise ValueError(f"chumpy 값에서 배열을 꺼내지 못함(상태 {sorted(state)})")
    return v.toarray() if hasattr(v, "toarray") else np.asarray(v)


def find(path):
    """MANO_LEFT.pkl 경로: 파일이면 그대로, 폴더면 그 아래에서 찾는다. 없으면 FileNotFoundError."""
    if os.path.isfile(path):
        return path
    for root, _, files in os.walk(path):
        if "MANO_LEFT.pkl" in files:
            return os.path.join(root, "MANO_LEFT.pkl")
    raise FileNotFoundError(f"{path}에 MANO_LEFT.pkl이 없음 — https://mano.is.tue.mpg.de 에서 받은 mano_v1_2.zip을 푼 폴더를 준다")


def load(path):
    """MANO_LEFT.pkl(또는 그것을 담은 폴더) → 모델 dict(numpy). 모양이 다르거나 손끝 정점이 맞지 않으면 ValueError."""
    with open(find(path), "rb") as f, warnings.catch_warnings():
        warnings.simplefilter("ignore")  # 파이썬 2 pkl의 dtype(align=0) — numpy 2.4가 알리기만 한다
        raw = _Unpickler(f, encoding="latin1").load()
    m = {k: _array(raw[k]).astype(np.float64) for k in ("v_template", "shapedirs", "posedirs", "J_regressor", "weights", "hands_mean")}
    m["faces"] = _array(raw["f"]).astype(np.int64)
    m["parents"] = _array(raw["kintree_table"])[0].astype(np.int64)
    m["parents"][0] = -1
    v = len(m["v_template"])
    want = {"v_template": (v, 3), "shapedirs": (v, 3, 10), "posedirs": (v, 3, 135), "J_regressor": (16, v), "weights": (v, 16), "hands_mean": (45,)}
    bad = {k: m[k].shape for k, s in want.items() if m[k].shape != s}
    if bad:
        raise ValueError(f"MANO 배열 모양이 다름: {bad}")
    if tuple(int(j) for j in m["weights"][list(TIPS)].argmax(1)) != TIP_JOINTS:
        raise ValueError(f"손끝 정점 {TIPS}이 끝마디 관절 {TIP_JOINTS}에 붙어 있지 않음 — 다른 MANO 판이면 TIPS를 고친다")
    return m


def rodrigues(aa):
    """축-각 (..., 3) → 회전 행렬 (..., 3, 3)."""
    angle = torch.sqrt((aa * aa).sum(-1, keepdim=True) + 1e-12)
    x, y, z = (aa / angle).unbind(-1)
    zero = torch.zeros_like(x)
    k = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).reshape(*aa.shape[:-1], 3, 3)
    s, c = torch.sin(angle)[..., None], torch.cos(angle)[..., None]
    return torch.eye(3, dtype=aa.dtype, device=aa.device) + s * k + (1 - c) * (k @ k)


class Layer:
    """MANO 앞으로 계산(torch, m). 모양 β는 모든 프레임이 함께 쓴다(한 사람의 세션)."""

    def __init__(self, m, device="cpu"):
        def t(a):
            return torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)

        self.device = device
        self.v_template, self.shapedirs, self.posedirs = t(m["v_template"]), t(m["shapedirs"]), t(m["posedirs"])
        self.J_regressor, self.weights, self.hands_mean = t(m["J_regressor"]), t(m["weights"]), t(m["hands_mean"])
        self.parents = [int(p) for p in m["parents"]]
        self.faces = np.asarray(m["faces"], dtype=np.int64)
        self.tips = list(m.get("tips", TIPS))

    def lbs(self, beta, orient, pose, idx):
        """(관절 16점 (B, 16, 3), 정점 idx (B, k, 3)). beta (10,), orient (B, 3), pose (B, 45)."""
        b = orient.shape[0]
        shaped = self.v_template + self.shapedirs @ beta
        J = (self.J_regressor @ shaped).expand(b, -1, -1)
        R = rodrigues(torch.cat([orient, pose], 1).reshape(b, 16, 3))
        feat = (R[:, 1:] - torch.eye(3, device=R.device)).reshape(b, 135)
        v = shaped[idx] + torch.einsum("kcp,bp->bkc", self.posedirs[idx], feat)
        rel = torch.cat([J[:, :1], J[:, 1:] - J[:, self.parents[1:]]], 1)
        T = torch.cat([torch.cat([R, rel[..., None]], -1), torch.tensor([0.0, 0.0, 0.0, 1.0], device=R.device).expand(b, 16, 1, 4)], -2)
        G = [T[:, 0]]
        for i in range(1, 16):
            G.append(G[self.parents[i]] @ T[:, i])
        G = torch.stack(G, 1)
        A = torch.cat([G[..., :3, :3], (G[..., :3, 3] - (G[..., :3, :3] @ J[..., None])[..., 0])[..., None]], -1)  # (B, 16, 3, 4)
        Tv = torch.einsum("kj,bjxy->bkxy", self.weights[idx], A)
        return G[..., :3, 3], (Tv[..., :3] @ v[..., None])[..., 0] + Tv[..., 3]

    def joints21(self, beta, orient, pose):
        j16, tips = self.lbs(beta, orient, pose, self.tips)
        return torch.cat([j16, tips], 1)[:, list(TO_MEDIAPIPE)]

    def mesh(self, beta, orient, pose):
        """(관절 21점 (B, 21, 3), 정점 (B, V, 3))."""
        j16, v = self.lbs(beta, orient, pose, slice(None))
        return torch.cat([j16, v[:, self.tips]], 1)[:, list(TO_MEDIAPIPE)], v


def kabsch(src, dst):
    """원점 기준 점 src·dst (N, K, 3) → dst ≈ R src인 회전 (N, 3, 3)."""
    U, _, Vt = np.linalg.svd(np.einsum("nki,nkj->nij", src, dst))
    V, Ut = Vt.transpose(0, 2, 1), U.transpose(0, 2, 1)
    D = np.tile(np.eye(3), (len(src), 1, 1))
    D[:, 2, 2] = np.sign(np.linalg.det(V @ Ut))
    return V @ D @ Ut


def _scaled(layer, beta, log_s, orient, pose):
    j = layer.joints21(beta, orient, pose)
    return (j - j[:, :1]) * (1000.0 * torch.exp(log_s))


def _rest(layer, beta, log_s):
    """hands_mean 자세·손목 회전 0의 관절 21점(mm, 손목 원점) numpy."""
    with torch.no_grad():
        return _scaled(layer, beta, log_s, torch.zeros(1, 3, device=layer.device), layer.hands_mean[None])[0].cpu().numpy().astype(np.float64)


def _orient0(layer, beta, log_s, target):
    rest = _rest(layer, beta, log_s)
    R = kabsch(np.broadcast_to(rest[list(PALM)], (len(target), len(PALM), 3)), target[:, list(PALM)])
    return Rotation.from_matrix(R).as_rotvec()


def _fit(layer, target, beta, log_s, steps, free_shape):
    """target (B, 21, 3) mm numpy. free_shape면 beta·log_s도 맞춘다. (beta, log_s, orient, pose, 잔차 (B,) mm)."""
    dev = layer.device
    T = torch.as_tensor(target, dtype=torch.float32, device=dev)
    beta = beta.clone().requires_grad_(free_shape)
    log_s = log_s.clone().requires_grad_(free_shape)
    orient = torch.as_tensor(_orient0(layer, beta.detach(), log_s.detach(), target), dtype=torch.float32, device=dev).requires_grad_()
    pose = layer.hands_mean.expand(len(target), 45).clone().requires_grad_()
    params = [orient, pose] + ([beta, log_s] if free_shape else [])
    opt = torch.optim.Adam(params, lr=LR)
    for _ in range(steps):
        d2 = ((_scaled(layer, beta, log_s, orient, pose) - T) ** 2).sum(-1)
        loss = d2.mean(1).mean() + POSE_WEIGHT * ((pose - layer.hands_mean) ** 2).sum(-1).mean()
        if free_shape:
            loss = loss + SHAPE_WEIGHT * (beta ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        res = torch.sqrt(((_scaled(layer, beta, log_s, orient, pose) - T) ** 2).sum(-1)).mean(1)
    return beta.detach(), log_s.detach(), orient.detach(), pose.detach(), res.cpu().numpy().astype(np.float64)


def fit_shape(layer, target_mm, frames=SHAPE_FRAMES, steps=SHAPE_STEPS):
    """정답 관절 (K, 21, 3) mm(손목 원점)에서 고르게 뽑은 frames장으로 모양 β (10,)와 크기 log s를 맞춘다: (beta, log_s, 잔차 (frames,) mm)."""
    target = np.asarray(target_mm, dtype=np.float64)
    pick = target[np.unique(np.linspace(0, len(target) - 1, min(frames, len(target))).round().astype(int))]
    beta = torch.zeros(10, device=layer.device)
    rest = _rest(layer, beta, torch.zeros((), device=layer.device))
    log_s = torch.tensor(float(np.log(np.median(np.linalg.norm(pick[:, 9], axis=1)) / np.linalg.norm(rest[9]))), device=layer.device)
    beta, log_s, _, _, res = _fit(layer, pick, beta, log_s, steps, True)
    return beta, log_s, res


def fit_pose(layer, target_mm, beta, log_s, steps=POSE_STEPS):
    """모양·크기를 고정하고 프레임마다 손목 회전·자세를 맞춘다: (orient (N, 3), pose (N, 45), 잔차 (N,) mm) torch·numpy."""
    target = np.asarray(target_mm, dtype=np.float64)
    out = [_fit(layer, target[a: a + CHUNK], beta, log_s, steps, False)[2:] for a in range(0, len(target), CHUNK)]
    return torch.cat([o[0] for o in out]), torch.cat([o[1] for o in out]), np.concatenate([o[2] for o in out])


def meshes(layer, beta, log_s, orient, pose, chunk=512):
    """맞춘 자세의 정점 (N, V, 3) mm(손목 원점, 목표와 같은 틀) float32."""
    out = []
    with torch.no_grad():
        for a in range(0, len(orient), chunk):
            j, v = layer.mesh(beta, orient[a: a + chunk], pose[a: a + chunk])
            out.append(((v - j[:, :1]) * (1000.0 * torch.exp(log_s))).cpu().numpy().astype(np.float32))
    return np.concatenate(out) if out else np.zeros((0, len(layer.v_template), 3), np.float32)
