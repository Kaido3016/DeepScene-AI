"""Load scene data and classify genres using rules or zero-shot model inference."""
from __future__ import annotations
import json, logging, os, re
from pathlib import Path
from typing import Any
from src.model_runtime import get_pipeline, configured_model
logger = logging.getLogger(__name__)
_WORD_RE = re.compile(r"[a-z0-9']+")
DEFAULT_TEMPLATES = {
"action":{"keywords":["fight","battle","chase","explosion","combat"],"style":"dynamic composition, dramatic lighting"},
"comedy":{"keywords":["funny","laugh","joke","humorous","silly"],"style":"bright colors, expressive framing"},
"drama":{"keywords":["emotional","grief","conflict","family","loss"],"style":"naturalistic lighting, intimate framing"},
"horror":{"keywords":["scary","frightening","ghost","monster","nightmare"],"style":"low-key lighting, atmospheric shadows"},
"romance":{"keywords":["love","romantic","couple","relationship","kiss","date"],"style":"soft lighting, intimate composition"},
"thriller":{"keywords":["suspense","mystery","tense","conspiracy","detective"],"style":"high contrast, suspenseful framing"},
"general":{"keywords":[],"style":"cinematic, professional photography"}}
class SceneDataLoader:
    def __init__(self, data_dir: str | Path = "data", backend: str | None = None):
        self.data_dir = Path(data_dir)
        self.backend = (backend or os.getenv("DEEPSCENE_INFERENCE_BACKEND","rules")).strip().lower()
        if self.backend not in {"rules","transformers"}:
            raise ValueError("DEEPSCENE_INFERENCE_BACKEND must be 'rules' or 'transformers'")
        self.templates = self._load_json("scene_templates.json", DEFAULT_TEMPLATES)
        self.samples = self._load_json("sample_scenes.json", {})
        if not isinstance(self.templates, dict):
            logger.warning("Scene templates must be a JSON object; using defaults.")
            self.templates = DEFAULT_TEMPLATES.copy()
        self.genre_keywords = self._build_enhanced_keywords()
    def _load_json(self, filename: str, fallback: Any) -> Any:
        path = self.data_dir / filename
        if not path.is_file():
            logger.warning("Optional data file missing: %s", path)
            return fallback.copy() if isinstance(fallback, dict) else fallback
        try:
            with path.open(encoding="utf-8") as file: return json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not load %s: %s", path, exc)
            return fallback.copy() if isinstance(fallback, dict) else fallback
    def _build_enhanced_keywords(self) -> dict[str, list[str]]:
        result = {}
        for genre, values in self.templates.items():
            base = values.get("keywords", []) if isinstance(values, dict) else []
            additions = DEFAULT_TEMPLATES.get(genre, {}).get("keywords", [])
            result[genre] = sorted({str(k).casefold() for k in [*base,*additions] if str(k).strip()})
        result.setdefault("general", [])
        return result
    def get_templates(self) -> dict[str, Any]: return self.templates
    def get_samples(self) -> Any: return self.samples
    def classify_scene_genre_details(self, description: str) -> dict[str, Any]:
        if not isinstance(description, str) or not description.strip():
            raise ValueError("description must be a non-empty string")
        if self.backend == "transformers":
            labels = [genre for genre in self.genre_keywords if genre != "general"]
            classifier = get_pipeline("genre")
            result = classifier(description[:5000], candidate_labels=labels, multi_label=False)
            ranked = [{"genre":str(label),"score":round(float(score),4)}
                      for label,score in zip(result.get("labels",[]),result.get("scores",[]))]
            if not ranked: raise RuntimeError("The zero-shot genre model returned no predictions.")
            return {"genre":ranked[0]["genre"],"confidence":ranked[0]["score"],
                    "confidence_type":"model_score_not_calibrated_probability",
                    "method":"transformers_zero_shot","ranked_candidates":ranked[:3],
                    "model":configured_model("genre")}
        lowered = description.casefold()
        tokens = set(_WORD_RE.findall(lowered))
        scores = {genre:sum(1 for k in keys if k in tokens or (len(k)>4 and k in lowered))
                  for genre,keys in self.genre_keywords.items()}
        best = max(scores.values(), default=0)
        if best == 0: return {"genre":"general","confidence":None,"method":"keyword_rules","scores":scores}
        winners = sorted(g for g,score in scores.items() if score == best)
        genre = winners[0]
        return {"genre":genre,"confidence":round(best/max(sum(scores.values()),1),3),
                "confidence_type":"heuristic_score_not_probability","method":"keyword_rules",
                "ambiguous":len(winners)>1,"scores":scores}
    def classify_scene_genre(self, description: str) -> str:
        return self.classify_scene_genre_details(description)["genre"]
    def get_style_prompt(self, genre: str) -> str:
        template = self.templates.get(genre,{}) if isinstance(self.templates,dict) else {}
        return str(template.get("style",DEFAULT_TEMPLATES["general"]["style"]))
