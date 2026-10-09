"""FastAPI application for DeepScene scene analysis."""
from __future__ import annotations
import hmac, logging, os, sys
from pathlib import Path
from typing import Any
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
ROOT=Path(__file__).resolve().parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.data_loader import SceneDataLoader
from src.preprocess import TextPreprocessor
from src.train import DeepSceneModels
logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"))
logger=logging.getLogger(__name__)
if os.getenv("DEEPSCENE_REQUIRE_API_KEY","0")=="1" and not os.getenv("DEEPSCENE_API_KEY"):
    raise RuntimeError("DEEPSCENE_REQUIRE_API_KEY=1 but DEEPSCENE_API_KEY is missing; refusing insecure startup.")
app=FastAPI(title="DeepScene API",version="1.1.0")
origins=[v.strip() for v in os.getenv("DEEPSCENE_CORS_ORIGINS","http://localhost:8501").split(",") if v.strip()]
app.add_middleware(CORSMiddleware,allow_origins=origins,allow_credentials=False,allow_methods=["GET","POST"],allow_headers=["Content-Type","Authorization","X-API-Key"])
class SceneRequest(BaseModel):
    description: str=Field(min_length=3,max_length=5000)
    style: str=Field(default="cinematic",max_length=200)
class SceneAnalysis(BaseModel):
    genre: str
    genre_analysis: dict[str,Any]
    mood: dict[str,Any]
    characters: list[str]
    setting: str
    image_prompt: str
    dialogue: str
data_loader=SceneDataLoader(ROOT/"data")
preprocessor=TextPreprocessor()
models=DeepSceneModels().initialize_all_models()
app.state.ready=True
@app.get("/health")
def health() -> dict[str,Any]:
    return {"status":"ok" if app.state.ready else "not_ready","version":app.version,
            "inference_backend":models.backend,"ai_models_loaded_lazily":True}
@app.post("/analyze_scene",response_model=SceneAnalysis)
def analyze_scene(request: SceneRequest, x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> SceneAnalysis:
    expected_key=os.getenv("DEEPSCENE_API_KEY")
    if expected_key and (not x_api_key or not hmac.compare_digest(x_api_key,expected_key)):
        raise HTTPException(status_code=401,detail="Invalid or missing API key.")
    try:
        description=preprocessor.clean_text(request.description)
        if not description: raise HTTPException(status_code=422,detail="description must not be blank")
        genre_result=data_loader.classify_scene_genre_details(description)
        genre=genre_result["genre"]
        style=request.style.strip() or data_loader.get_style_prompt(genre)
        return SceneAnalysis(genre=genre,genre_analysis=genre_result,mood=models.classify_scene_mood(description),
            characters=preprocessor.extract_characters(description),setting=preprocessor.extract_setting(description),
            image_prompt=preprocessor.generate_image_prompt(description,style),dialogue=models.generate_dialogue(description))
    except HTTPException: raise
    except ValueError as exc: raise HTTPException(status_code=422,detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("Configured inference backend failed")
        raise HTTPException(status_code=503,detail=str(exc)) from None
    except Exception:
        logger.exception("Scene analysis failed")
        raise HTTPException(status_code=500,detail="Scene analysis failed; see server logs.") from None
