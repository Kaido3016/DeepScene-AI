"""Optional local image generation and atomic JSON export helpers."""
from __future__ import annotations
import json, logging, os, re
from pathlib import Path
from typing import Any
logger=logging.getLogger(__name__)
_PIPELINES: dict[tuple[str,str],Any]={}
def generate_ai_image_free(prompt: str, filename: str, folder: str="results/images") -> str:
    """Generate real artwork using optional local Diffusers dependencies; never fake success."""
    if not isinstance(prompt,str) or not prompt.strip(): raise ValueError("prompt must be a non-empty string")
    model_id=os.getenv("DEEPSCENE_IMAGE_MODEL","stable-diffusion-v1-5/stable-diffusion-v1-5")
    try:
        import torch
        from diffusers import StableDiffusionPipeline
    except ImportError as exc:
        raise RuntimeError("Install requirements-ml.txt to enable optional image generation.") from exc
    device="cuda" if torch.cuda.is_available() else "cpu"; key=(model_id,device)
    if key not in _PIPELINES:
        dtype=torch.float16 if device=="cuda" else torch.float32
        _PIPELINES[key]=StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=dtype).to(device)
        if device=="cuda": _PIPELINES[key].enable_attention_slicing()
    safe_name=re.sub(r"[^A-Za-z0-9_.-]+","_",Path(filename).stem).strip("._") or "scene"
    output_dir=Path(folder); output_dir.mkdir(parents=True,exist_ok=True)
    filepath=output_dir/f"{safe_name}.png"
    result=_PIPELINES[key](prompt=prompt,width=512,height=512,num_inference_steps=20)
    if not result.images: raise RuntimeError("The image pipeline returned no images.")
    result.images[0].save(filepath)
    return str(filepath)
def save_json(data: Any, filename: str, folder: str="results/exports") -> str:
    output_dir=Path(folder); output_dir.mkdir(parents=True,exist_ok=True)
    name=Path(filename).name
    if not name.lower().endswith(".json"): name += ".json"
    filepath=output_dir/name; temp_path=filepath.with_suffix(filepath.suffix+".tmp")
    with temp_path.open("w",encoding="utf-8") as file: json.dump(data,file,indent=2,ensure_ascii=False)
    temp_path.replace(filepath)
    return str(filepath)
