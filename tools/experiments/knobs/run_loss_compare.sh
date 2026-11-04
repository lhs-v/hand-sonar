#!/bin/sh
# 칸 머리의 손실 구성 비교(톱니 두 세션, HaMeR 정답, 블록 나누기, 씨앗값 0·1·2). 기존 saw-hamer-bins-s*가 "둘 다"(좌표 + 0.01 × 칸).
# 다시 돌림 확인 1번 + 좌표만(--bin-weight 0) 3번 + 칸만(--coord-weight 0) 3번. 한 번에 약 90초.
cd "$(dirname "$0")/../../.." || exit 1
S="20261031-230259 20261031-230647"
python -I tools/train.py --mode blocks --gt hamer --head bins --seed 0 --run saw-hamer-bins-recheck-s0 $S | tail -3
for s in 0 1 2; do
  python -I tools/train.py --mode blocks --gt hamer --head bins --bin-weight 0 --seed $s --run saw-hamer-bins-coordonly-s$s $S | tail -3
  python -I tools/train.py --mode blocks --gt hamer --head bins --coord-weight 0 --seed $s --run saw-hamer-bins-ceonly-s$s $S | tail -3
done
