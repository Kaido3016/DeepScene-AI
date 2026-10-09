"""Compatibility entry point for the Streamlit UI."""
from pathlib import Path
import subprocess
import sys

if __name__ == "__main__":
    app = Path(__file__).resolve().with_name("streamlit_app.py")
    raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app), *sys.argv[1:]]))
