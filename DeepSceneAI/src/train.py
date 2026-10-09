"""Scene mood and dialogue with explicit rules or real Transformers inference."""
from __future__ import annotations
import os
import re
from typing import Any
from src.model_runtime import get_pipeline

MOOD_KEYWORDS: dict[str, tuple[str, ...]] = {
    "happy": ("happy","joy","celebrat","dance","laugh","smile","fun","party"),
    "sad": ("sad","cry","tear","depressed","lonely","heartbreak","loss","grief"),
    "tense": ("tense","suspense","nervous","anxious","worried","stress","chase","fight"),
    "fearful": ("fear","scared","afraid","terrified","horror","frighten","nightmare"),
    "romantic": ("romantic","love","passion","intimate","affection","kiss","date"),
    "energetic": ("energy","exciting","dynamic","action","fast","intense","adventure"),
    "mysterious": ("mystery","secret","unknown","puzzle","curious","enigma","detective"),
    "peaceful": ("calm","peace","quiet","serene","tranquil","relax","gentle")}
_WORD_RE = re.compile(r"[a-z0-9']+")
EMOTION_TO_MOOD = {
    "joy":"happy","happiness":"happy","sadness":"sad","anger":"tense",
    "fear":"fearful","surprise":"mysterious","disgust":"tense","neutral":"undetermined"
}
class DeepSceneModels:
    """Rules are dependency-free; set DEEPSCENE_INFERENCE_BACKEND=transformers for model inference."""
    def __init__(self, device: str | None = None, backend: str | None = None):
        self.device = device or "auto"
        self.backend = (backend or os.getenv("DEEPSCENE_INFERENCE_BACKEND","rules")).strip().lower()
        if self.backend not in {"rules","transformers"}:
            raise ValueError("DEEPSCENE_INFERENCE_BACKEND must be 'rules' or 'transformers'")
        self.models: dict[str, Any] = {}
        self.mood_keywords = MOOD_KEYWORDS
    def initialize_all_models(self) -> "DeepSceneModels":
        self.models = {"mood_classifier": self.backend, "dialogue_generator": self.backend}
        return self
    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(_WORD_RE.findall(text.casefold()))
    def _classify_mood_transformer(self, description: str) -> dict[str, Any]:
        classifier = get_pipeline("emotion")
        predictions = classifier(description[:5000], top_k=None, truncation=True)
        if predictions and isinstance(predictions[0], list): predictions = predictions[0]
        if not predictions: raise RuntimeError("The emotion model returned no predictions.")
        best = max(predictions, key=lambda item: float(item.get("score",0)))
        label = str(best.get("label","")).casefold()
        mood = EMOTION_TO_MOOD.get(label, "undetermined")
        return {"mood":mood,"confidence":round(float(best["score"]),4),
                "confidence_type":"model_score_not_calibrated_probability",
                "method":"transformers_emotion_classifier","model_label":label}
    def classify_scene_mood(self, description: str) -> dict[str, Any]:
        if not isinstance(description, str) or not description.strip():
            raise ValueError("description must be a non-empty string")
        if self.backend == "transformers":
            result = self._classify_mood_transformer(description)
            from src.model_runtime import configured_model
            result["model"] = configured_model("emotion")
            return result
        tokens = self._tokens(description)
        scores = {mood: sum(1 for k in keys if k in tokens or
                  (k.endswith(("ing","ed","at")) and k in description.casefold()))
                  for mood, keys in self.mood_keywords.items()}
        best = max(scores.values(), default=0)
        if not best:
            return {"mood":"undetermined","confidence":None,"method":"keyword_rules","evidence":[]}
        winners = sorted(m for m, score in scores.items() if score == best)
        mood = winners[0]
        evidence = [k for k in self.mood_keywords[mood] if k in tokens or
                    (k.endswith(("ing","ed","at")) and k in description.casefold())]
        return {"mood":mood,"confidence":round(best/max(sum(scores.values()),1),3),
                "confidence_type":"heuristic_score_not_probability","method":"keyword_rules",
                "evidence":evidence,"ambiguous":len(winners)>1}
    def generate_dialogue(self, description: str) -> str:
        if not isinstance(description, str) or not description.strip():
            raise ValueError("description must be a non-empty string")
        if self.backend == "transformers":
            generator = get_pipeline("dialogue")
            prompt = ("Write 2 to 4 natural lines of original film dialogue grounded in this scene. "
                      "Do not describe the task or add a preamble. Scene: " + description[:2500])
            results = generator(prompt, max_new_tokens=112, do_sample=True, temperature=0.8,
                                num_return_sequences=1, truncation=True)
            if not results or not str(results[0].get("generated_text","")).strip():
                raise RuntimeError("The dialogue model returned empty output.")
            return str(results[0]["generated_text"]).strip()
        mood = self.classify_scene_mood(description)["mood"]
        lines = {"happy":'"Come on, let’s enjoy this moment!"',
        "sad":'"I wish things had turned out differently," they said quietly.',
        "tense":'"Stay close, and don’t make a sound," they whispered.',
        "fearful":'"Did you hear that? We need to leave."',
        "romantic":'"There’s nowhere else I’d rather be," they said softly.',
        "energetic":'"Move! We don’t have a second to lose!"',
        "mysterious":'"Something about this place doesn’t add up."',
        "peaceful":'"Let’s stay here a little longer."',
        "undetermined":'"What do you think we should do next?"'}
        return lines[mood]
    def generate_tts(self, text: str) -> None:
        raise NotImplementedError("Text-to-speech is not configured; no audio was generated.")
    def generate_image(self, prompt: str) -> None:
        raise NotImplementedError("Use generate_ai_image_free() with optional image dependencies.")
