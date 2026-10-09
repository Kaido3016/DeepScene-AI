"""Lazy, cached access to real Hugging Face Transformers pipelines."""
from __future__ import annotations
from functools import lru_cache
import os
from typing import Any
DEFAULT_MODELS = {"emotion":"j-hartmann/emotion-english-distilroberta-base","genre":"valhalla/distilbart-mnli-12-1","dialogue":"google/flan-t5-base"}
TASKS = {"emotion":"text-classification","genre":"zero-shot-classification","dialogue":"text2text-generation"}
def configured_model(kind: str) -> str:
    env_name = "DEEPSCENE_MOOD_MODEL" if kind == "emotion" else f"DEEPSCENE_{kind.upper()}_MODEL"
    return os.getenv(env_name, DEFAULT_MODELS[kind]).strip()
@lru_cache(maxsize=4)
def _pipeline(task: str, model_id: str) -> Any:
    try:
        import torch
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError("Transformers inference is enabled but optional AI dependencies are missing. Install requirements-ml.txt and a compatible PyTorch build.") from exc
    device = 0 if torch.cuda.is_available() else -1
    try: return pipeline(task, model=model_id, device=device)
    except Exception as exc:
        raise RuntimeError(f"Could not load model '{model_id}'. Check network/model access, model ID, available RAM/VRAM, and model license requirements.") from exc
def get_pipeline(kind: str) -> Any:
    if kind not in TASKS: raise ValueError(f"Unknown model pipeline kind: {kind}")
    return _pipeline(TASKS[kind], configured_model(kind))
