"""Evaluate the rule-based scene classifiers on a small labeled fixture set."""
from __future__ import annotations
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.data_loader import SceneDataLoader
from src.train import DeepSceneModels

def main() -> None:
    cases=json.loads((ROOT/"data"/"evaluation_scenes.json").read_text(encoding="utf-8"))
    genres=SceneDataLoader(ROOT/"data"); moods=DeepSceneModels()
    genre_correct=mood_correct=0
    for case in cases:
        actual_genre=genres.classify_scene_genre(case["description"])
        actual_mood=moods.classify_scene_mood(case["description"])["mood"]
        genre_correct += actual_genre == case["genre"]
        mood_correct += actual_mood == case["mood"]
        print(f'{case["id"]}: genre={actual_genre} (expected {case["genre"]}); mood={actual_mood} (expected {case["mood"]})')
    n=len(cases)
    if not n: raise SystemExit("Evaluation fixture set is empty.")
    print(f"Genre accuracy: {genre_correct}/{n} = {genre_correct/n:.1%}")
    print(f"Mood accuracy:  {mood_correct}/{n} = {mood_correct/n:.1%}")
    print("Small hand-labeled smoke-test set only; these figures are not production-quality benchmarks.")
if __name__=="__main__": main()
