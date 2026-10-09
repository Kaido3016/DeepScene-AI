"""Scene analysis primitives: deterministic rules, not a trained model."""
from __future__ import annotations
import re
from typing import Any
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
class DeepSceneModels:
    """Lightweight scene helpers. Heavy model integrations are not auto-loaded."""
    def __init__(self, device: str | None = None):
        self.device = device or "cpu"
        self.models: dict[str, Any] = {}
        self.mood_keywords = MOOD_KEYWORDS
    def initialize_all_models(self) -> "DeepSceneModels":
        self.models = {"mood_classifier": "rules", "dialogue_generator": "templates"}
        return self
    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(_WORD_RE.findall(text.casefold()))
    def classify_scene_mood(self, description: str) -> dict[str, Any]:
        """Confidence is a normalized heuristic score, not a calibrated probability."""
        if not isinstance(description, str) or not description.strip():
            raise ValueError("description must be a non-empty string")
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
