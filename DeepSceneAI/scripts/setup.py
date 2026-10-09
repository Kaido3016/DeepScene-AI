"""Create local working directories without overwriting user data."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
def main() -> None:
    if sys.version_info<(3,10): raise SystemExit("Python 3.10 or newer is required.")
    for name in ("models","results/images","results/exports","results/logs"):
        (ROOT/name).mkdir(parents=True,exist_ok=True)
    print("Directories ready. Create a virtual environment, then run:")
    print("  python -m pip install -r requirements.txt")
    print("  streamlit run streamlit_app.py")
    print("Optional image generation: python -m pip install -r requirements-ml.txt")
if __name__=="__main__": main()
