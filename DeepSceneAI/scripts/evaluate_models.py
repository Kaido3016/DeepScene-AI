"""Evaluate the configured scene classifiers on a hand-labeled JSON dataset."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.data_loader import SceneDataLoader
from src.train import DeepSceneModels
def metrics(rows: list[tuple[str,str]]) -> dict[str,float]:
    labels=sorted({e for e,_ in rows}|{a for _,a in rows})
    f1s=[]
    for label in labels:
        tp=sum(e==label and a==label for e,a in rows); fp=sum(e!=label and a==label for e,a in rows); fn=sum(e==label and a!=label for e,a in rows)
        p=tp/(tp+fp) if tp+fp else 0.; r=tp/(tp+fn) if tp+fn else 0.
        f1s.append(2*p*r/(p+r) if p+r else 0.)
    return {"accuracy":sum(e==a for e,a in rows)/len(rows),"macro_f1":sum(f1s)/len(f1s)}
def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",default=str(ROOT/"data"/"evaluation_scenes.json"))
    args=parser.parse_args(); cases=json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if not cases: raise SystemExit("Evaluation dataset is empty.")
    loader=SceneDataLoader(ROOT/"data"); models=DeepSceneModels(); genres=[]; moods=[]
    for case in cases:
        genres.append((case["genre"],loader.classify_scene_genre(case["description"])))
        moods.append((case["mood"],models.classify_scene_mood(case["description"])["mood"]))
    print(f"Dataset: {args.dataset} ({len(cases)} labeled examples)")
    print(f"Backend: genre={loader.backend}, mood={models.backend}")
    for name,rows in (("Genre",genres),("Mood",moods)):
        m=metrics(rows); print(f"{name}: accuracy={m['accuracy']:.1%}; macro-F1={m['macro_f1']:.3f}")
    print("Small development fixture only; not an independent or production-quality benchmark.")
if __name__=="__main__": main()
