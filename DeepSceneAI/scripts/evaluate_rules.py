"""Evaluate configured scene classifiers on a hand-labeled JSON dataset."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.data_loader import SceneDataLoader
from src.train import DeepSceneModels

def metrics(rows: list[tuple[str,str]]) -> dict[str,float]:
    labels=sorted({expected for expected,_ in rows}|{actual for _,actual in rows})
    result={}
    for label in labels:
        tp=sum(expected==label and actual==label for expected,actual in rows)
        fp=sum(expected!=label and actual==label for expected,actual in rows)
        fn=sum(expected==label and actual!=label for expected,actual in rows)
        precision=tp/(tp+fp) if tp+fp else 0.0
        recall=tp/(tp+fn) if tp+fn else 0.0
        result[label]=2*precision*recall/(precision+recall) if precision+recall else 0.0
    return {"accuracy":sum(a==b for a,b in rows)/len(rows),"macro_f1":sum(result.values())/len(result)}
def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",default=str(ROOT/"data"/"evaluation_scenes.json"))
    args=parser.parse_args()
    cases=json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if not cases: raise SystemExit("Evaluation dataset is empty.")
    loader=SceneDataLoader(ROOT/"data"); models=DeepSceneModels()
    genre_rows=[]; mood_rows=[]
    for case in cases:
        genre=loader.classify_scene_genre(case["description"])
        mood=models.classify_scene_mood(case["description"])["mood"]
        genre_rows.append((case["genre"],genre)); mood_rows.append((case["mood"],mood))
    print(f"Dataset: {args.dataset} ({len(cases)} labeled examples)")
    print(f"Backend: genre={loader.backend}, mood={models.backend}")
    print(f"Genre accuracy={metrics(genre_rows)['accuracy']:.1%}; macro-F1={metrics(genre_rows)['macro_f1']:.3f}")
    print(f"Mood accuracy={metrics(mood_rows)['accuracy']:.1%}; macro-F1={metrics(mood_rows)['macro_f1']:.3f}")
    print("These are smoke-test metrics for a small hand-labeled set, not independent or production-quality benchmarks.")
if __name__=="__main__": main()
