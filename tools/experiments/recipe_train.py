"""train.py를 고치지 않고 학습 방법만 바꿔 돌린다(2026-10-28 실험: "학습률은 고정이야?"에 답한 판들).

    python -I tools/experiments/recipe_train.py <설정 이름> <train.py 인자…>
    예) python -I tools/experiments/recipe_train.py cosine-reg --mode blocks --epochs 60 --max-minutes 30 --run blocks-cosine-reg

설정(RECIPES):
- adam-long, adam-long-lr3e-4: train.py와 같은 Adam 고정 학습률을 조기 멈춤 없이 끝까지.
- cosine: AdamW(가중치 감쇠 0.05), 3 epoch 선형 warmup 뒤 cosine으로 0까지(반복마다 바꿈).
- cosine-reg: cosine + FastViT drop path 0.1 + 입력 증강(표준화된 창에 학습 때만: 가우스 잡음 σ 0.2, 절반 확률로 시간 줄 16개 가리기,
  절반 확률로 칸 8개 가리기, 두 채널 같이).
모두 조기 멈춤 없이 정한 epoch까지 돌고, 평가는 train.py처럼 검증 손실이 가장 낮던 epoch의 가중치로 한다. 자료·나누기·손실·평가는 train.py 그대로다.
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tools
import train  # noqa: E402  사용자 설치 numpy·torch·timm 경로도 붙는다
import numpy as np  # noqa: E402
import timm  # noqa: E402
import torch  # noqa: E402

RECIPES = {
    "adam-long": {"opt": "adam", "lr": 1e-3, "wd": 0.0, "warmup": 0, "cosine": False, "drop_path": 0.0, "aug": False},
    "adam-long-lr3e-4": {"opt": "adam", "lr": 3e-4, "wd": 0.0, "warmup": 0, "cosine": False, "drop_path": 0.0, "aug": False},
    "cosine": {"opt": "adamw", "lr": 1e-3, "wd": 0.05, "warmup": 3, "cosine": True, "drop_path": 0.0, "aug": False},
    "cosine-reg": {"opt": "adamw", "lr": 1e-3, "wd": 0.05, "warmup": 3, "cosine": True, "drop_path": 0.1, "aug": True},
}


def make_model(recipe):
    return timm.create_model(train.ARCH, pretrained=False, in_chans=2, num_classes=60, drop_path_rate=recipe["drop_path"])


def augment(xb, gen):
    """표준화된 창 (B, 2, 96, 60)에 학습 때만: 잡음 σ 0.2, 절반 확률로 시간 줄 16개 가리기, 절반 확률로 칸 8개 가리기(두 채널 같이)."""
    b, _, t, f = xb.shape
    xb = xb + 0.2 * torch.randn(xb.shape, device=xb.device, generator=gen)
    tr = torch.arange(t, device=xb.device)
    fr = torch.arange(f, device=xb.device)
    use_t = torch.rand(b, device=xb.device, generator=gen) < 0.5
    use_f = torch.rand(b, device=xb.device, generator=gen) < 0.5
    st = torch.randint(0, t - 16 + 1, (b,), device=xb.device, generator=gen)
    sf = torch.randint(0, f - 8 + 1, (b,), device=xb.device, generator=gen)
    mt = (tr[None] >= st[:, None]) & (tr[None] < st[:, None] + 16) & use_t[:, None]  # (b, t)
    mf = (fr[None] >= sf[:, None]) & (fr[None] < sf[:, None] + 8) & use_f[:, None]  # (b, f)
    keep = ~(mt[:, :, None] | mf[:, None, :])  # (b, t, f)
    return xb * keep[:, None].to(xb.dtype)


def fitter(recipe):
    """train.fit과 같은 모양의 학습 함수(조기 멈춤 없음)."""

    def fit(x, y, train_segs, val_segs, epochs, seed, log=print, max_minutes=train.MAX_MINUTES):
        torch.manual_seed(seed)
        model = make_model(recipe).to(x.device)
        params = model.parameters()
        opt = (torch.optim.AdamW(params, lr=recipe["lr"], weight_decay=recipe["wd"]) if recipe["opt"] == "adamw"
               else torch.optim.Adam(params, lr=recipe["lr"]))
        rng = np.random.default_rng(seed)
        gen = torch.Generator(device=x.device)
        gen.manual_seed(seed)
        steps_per_epoch = len(train.batches(train_segs, np.random.default_rng(seed)))
        total, warm = epochs * steps_per_epoch, recipe["warmup"] * steps_per_epoch

        def lr_at(step):
            if step < warm:
                return recipe["lr"] * (step + 1) / warm
            if not recipe["cosine"]:
                return recipe["lr"]
            return recipe["lr"] * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

        best, best_epoch, best_state, history, stopped, step = float("inf"), 0, None, [], "epochs", 0
        start = time.time()
        for epoch in range(1, epochs + 1):
            model.train()
            seen, count = 0.0, 0
            for idx, pair in train.batches(train_segs, rng):
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                i = torch.as_tensor(idx, device=x.device)
                xb = augment(x[i], gen) if recipe["aug"] else x[i]
                loss = train.loss_fn(model(xb), y[i], torch.as_tensor(pair, device=x.device))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                step += 1
                value = loss.item()
                if not math.isfinite(value):
                    raise FloatingPointError(f"epoch {epoch}: 학습 손실이 유한하지 않음({value})")
                seen += value * len(idx)
                count += len(idx)
            val = train.eval_loss(model, x, y, val_segs)
            history.append({"epoch": epoch, "train": seen / count, "val": val})
            if val < best:
                best, best_epoch = val, epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if epoch == 1 or epoch % 10 == 0:
                log(f"  epoch {epoch}: 학습 손실 {seen / count:.5f}, 검증 손실 {val:.5f}(최고 epoch {best_epoch}), 학습률 {lr_at(step - 1):.2e}")
            if epoch < epochs and time.time() - start >= max_minutes * 60:
                stopped = "time"
                break
        model.load_state_dict(best_state)
        return model, history, best_epoch, stopped

    return fit


if __name__ == "__main__":
    name = sys.argv[1]
    if name not in RECIPES:
        sys.exit(f"모르는 설정 {name}: {', '.join(RECIPES)}")
    train.fit = fitter(RECIPES[name])
    sys.argv = ["train.py"] + sys.argv[2:]
    sys.exit(train.main())
