"""Explicit smoke test for the configured local Diffusers image model."""
from __future__ import annotations
import argparse, os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.utils.io_utils import generate_ai_image_free

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt",default="A cinematic film still of a lone detective in a rainy neon-lit city, wide composition")
    parser.add_argument("--output",default=str(ROOT/"results"/"images"/"image_generation_smoke_test.png"))
    args=parser.parse_args()
    model=os.getenv("DEEPSCENE_IMAGE_MODEL","stable-diffusion-v1-5/stable-diffusion-v1-5")
    output=generate_ai_image_free(args.prompt,Path(args.output).name,str(Path(args.output).parent))
    try:
        from PIL import Image
        with Image.open(output) as image:
            image.verify()
        with Image.open(output) as image:
            print(f"PASS: valid image {output}; size={image.size}; mode={image.mode}; model={model}")
    except Exception as exc:
        raise SystemExit(f"FAIL: output file is not a readable image: {exc}") from exc
if __name__=="__main__": main()
