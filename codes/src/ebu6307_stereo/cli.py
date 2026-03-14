from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    print("EBU6307 stereo bootstrap is installed.")
    print(f"Repository root: {root}")
    print("Next step: implement O1 synthetic stereo generation under codes/src/ebu6307_stereo/.")
    return 0
