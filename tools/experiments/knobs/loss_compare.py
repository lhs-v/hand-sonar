"""출력·손실 구성별 결과를 metrics.json에서 모은다(톱니 두 세션, HaMeR 정답, 블록 나누기, 씨앗값 0·1·2). 결과는 loss_compare.json.

    python -I tools/experiments/knobs/loss_compare.py --out <저장소 밖 폴더> [--data 우리 자료] [--pub 공개 자료]
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402  tools 경로를 붙이고 --out·--data·--pub를 읽는다
import hs_io  # noqa: E402
import numpy as np  # noqa: E402

CONDITIONS = (("좌표\n회귀", "saw-hamer-s{}", "#c3c2b7"), ("각도\n회귀", "saw-hamer-angles-s{}", "#c3c2b7"),
              ("칸\n좌표 손실만", "saw-hamer-bins-coordonly-s{}", "#2a78d6"), ("칸\n칸 손실만", "saw-hamer-bins-ceonly-s{}", "#2a78d6"),
              ("칸\n둘 다(지금)", "saw-hamer-bins-s{}", "#c2548a"))


def main():
    base = os.path.join(common.ARGS.data, "processed", "training")
    res = {"conditions": []}
    for label, pattern, color in CONDITIONS:
        rows = [json.load(open(os.path.join(base, pattern.format(s), "metrics.json"), encoding="utf-8")) for s in (0, 1, 2)]
        it = {"label": label, "pattern": pattern, "color": color, "ratio": [m["ratioToMeanPose"] for m in rows], "mm": [m["mpjpeMm"] for m in rows],
              "movingMm": [m["movingMpjpeMm"] for m in rows], "slope": [m["collapse"]["slope"]["mean"] for m in rows], "bestEpoch": [m["bestEpoch"] for m in rows],
              "flexion": [m["angles"]["model"]["flexionDeg"] for m in rows] if rows[0].get("angles") else None,
              "abduction": [m["angles"]["model"]["abductionDeg"] for m in rows] if rows[0].get("angles") else None}
        res["conditions"].append(it)
        print(label.replace("\n", " "), "비", np.round(it["ratio"], 3), f"평균 {np.mean(it['ratio']):.3f}", "mm", np.round(it["mm"], 2), f"평균 {np.mean(it['mm']):.2f}", "움직임 mm", round(float(np.mean(it["movingMm"])), 2),
              "기울기", round(float(np.mean(it["slope"])), 3), "최고 epoch", it["bestEpoch"], "굽힘·벌림", None if it["flexion"] is None else (round(float(np.mean(it["flexion"])), 1), round(float(np.mean(it["abduction"])), 1)))
    r = json.load(open(os.path.join(base, "saw-hamer-bins-recheck-s0", "metrics.json"), encoding="utf-8"))
    o = json.load(open(os.path.join(base, "saw-hamer-bins-s0", "metrics.json"), encoding="utf-8"))
    res["recheck"] = {"ratio": r["ratioToMeanPose"], "saved": o["ratioToMeanPose"]}
    print(f"다시 돌림 확인(씨앗값 0, 고친 코드): 비 {r['ratioToMeanPose']:.4f} 대 저장 {o['ratioToMeanPose']:.4f}")
    json.dump(res, open(common.out("loss_compare.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("저장: loss_compare.json")


if __name__ == "__main__":
    main()
