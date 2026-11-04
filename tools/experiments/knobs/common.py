"""tools/experiments/knobs 스크립트의 공통 부분(연구 현황 5.10·5.15): tools를 import 경로에 붙이고 명령줄의 폴더를 읽는다.

    --out   결과(json·png)를 두는 저장소 밖 폴더. 그림 스크립트는 여기 있는 json을 읽는다.
            공개 자료에서 나온 결과가 섞이므로 저장소에 넣지 않는다(docs/public-watchhand-data.md 9절).
    --data  우리 자료 폴더(기본: 저장소 옆 hand-sonar-data)
    --pub   공개 WatchHand를 우리 형식으로 바꾼 폴더(tools/experiments/public_to_ours.py, 기본: 저장소 옆 public-as-ours)
    --run   loss_terms.py가 볼 학습 실행 이름(기본 saw-hamer-bins-s0)
"""
import argparse
import os
import sys

TOOLS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TOOLS)
import hs_io  # noqa: E402  사용자 설치 numpy·scipy 경로도 여기서 붙는다


def _parse():
    parent = os.path.dirname(hs_io.REPO)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.join(parent, "hand-sonar-knobs"))
    parser.add_argument("--data", default=hs_io.DEFAULT_DATA)
    parser.add_argument("--pub", default=os.path.join(parent, "public-as-ours"))
    parser.add_argument("--run", default="saw-hamer-bins-s0")
    return parser.parse_known_args()[0]


ARGS = _parse()


def out(name):
    """결과 폴더 안 파일 경로(폴더가 없으면 만든다)."""
    os.makedirs(ARGS.out, exist_ok=True)
    return os.path.join(ARGS.out, name)
