"""공개 데이터로 학습한 모델을 우리 데이터에: 그대로 적용(zeroshot) 또는 미세 조정(finetune). 2026-10-28 실험이며 train.py는 고치지 않는다.

    python -I tools/experiments/transfer.py zeroshot <공개 model.pt> <우리 데이터 폴더> <공개 dataset.npz> <결과 json>
    python -I tools/experiments/transfer.py finetune <공개 model.pt> <학습률> <train.py 인자…>

공개 model.pt·dataset.npz는 public_to_ours.py로 옮긴 폴더에서 `train.py --data <그 폴더> --mode blocks`로 만든다.

- zeroshot: 우리 쓸 수 있는 세션 전부(dataset.build)의 창을 그 모델로 예측하고 train.instruments로 잰다. 평균 자세 기준은 우리 자료 전체의
  평균(모델은 보지 못한 값, 참고 상한)이고, 공개 평균 자세(공개 dataset.npz y의 평균)를 그대로 냈을 때의 오차도 함께 적는다.
- finetune: train.make_model이 공개 가중치를 실은 모델을 돌려주게 바꾸고 학습률만 바꿔 train.main()을 부른다(나누기·손실·평가는 그대로).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tools
import dataset  # noqa: E402  사용자 설치 numpy 경로도 붙는다
import joints  # noqa: E402
import train  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


def weights(path):
    return torch.load(path, map_location="cpu", weights_only=True)["state_dict"]


def zeroshot(model_path, data, pub_ds, out):
    ds, lines = dataset.build(data)
    print("\n".join(lines))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train.make_model()
    model.load_state_dict(weights(model_path))
    model.to(device)
    n = len(ds["y"])
    x = torch.as_tensor(ds["x"], device=device)
    pred = train.predict(model, x, np.arange(n))
    every = np.arange(n)
    m = train.instruments(ds, pred, every, every)
    with np.load(pub_ds, allow_pickle=False) as f:
        pub_mean = f["y"].mean(axis=0)
    true = ds["y"].reshape(-1, 20, 3)
    mm = ds["length_mm"][ds["session"]]
    e_pub = joints.frame_errors(pub_mean.reshape(1, 20, 3), true) * mm
    e_model = joints.frame_errors(pred.reshape(-1, 20, 3), true) * mm
    per = {}
    for k, name in enumerate(ds["sessions"].tolist()):
        s = ds["session"] == k
        own = true[s].reshape(-1, 60).mean(axis=0)
        e_own = joints.frame_errors(own.reshape(1, 20, 3), true[s]) * mm[s]
        per[name] = {"frames": int(s.sum()), "modelMm": float(e_model[s].mean()), "ownMeanMm": float(e_own.mean()), "pubMeanMm": float(e_pub[s].mean())}
    m.update({"run": "zeroshot", "model": model_path, "pubMeanPoseMm": float(e_pub.mean()), "perSession": per})
    with open(out, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=1)
    c = m["collapse"]
    print(f"그대로 적용: FastViT {m['mpjpeMm']:.2f} mm · 우리 평균 자세 {m['baselines']['meanPoseMm']:.2f} mm(비 {m['ratioToMeanPose']:.3f})"
          f" · 공개 평균 자세 {float(e_pub.mean()):.2f} mm")
    print(f"  기울기·상관(움직이는 점): {json.dumps(c, ensure_ascii=False)[:300]}")
    for name, v in per.items():
        print(f"  {name}: 모델 {v['modelMm']:.1f} mm · 그 세션 평균 자세 {v['ownMeanMm']:.1f} · 공개 평균 자세 {v['pubMeanMm']:.1f}")


def finetune(model_path, lr, rest):
    state = weights(model_path)
    make = train.make_model

    def make_model(outputs=train.FINGER_OUT):
        model = make(outputs)
        model.load_state_dict(state)
        return model

    train.make_model = make_model
    train.LR = lr
    sys.argv = ["train.py"] + rest
    return train.main()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if sys.argv[1] == "zeroshot":
        zeroshot(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        sys.exit(finetune(sys.argv[2], float(sys.argv[3]), sys.argv[4:]))
