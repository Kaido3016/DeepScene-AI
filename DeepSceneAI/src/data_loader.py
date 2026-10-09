"""Load scene data and classify genres deterministically."""
from __future__ import annotations
import json, logging, re
from pathlib import Path
from typing import Any
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
    def __init__(self, data_dir: str | Path = "data"):
        self.data_dir = Path(data_dir)
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
    def classify_scene_genre(self, description: str) -> str:
        if not isinstance(description, str) or not description.strip():
            raise ValueError("description must be a non-empty string")
        lowered = description.casefold()
        tokens = set(_WORD_RE.findall(lowered))
        scores = {genre:sum(1 for k in keys if k in tokens or (len(k)>4 and k in lowered))
                  for genre,keys in self.genre_keywords.items()}
        best = max(scores.values(), default=0)
        if best == 0: return "general"
        return sorted(g for g,score in scores.items() if score == best)[0]
    def get_style_prompt(self, genre: str) -> str:
        template = self.templates.get(genre,{}) if isinstance(self.templates,dict) else {}
        return str(template.get("style",DEFAULT_TEMPLATES["general"]["style"]))
