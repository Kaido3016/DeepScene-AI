from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.data_loader import SceneDataLoader
from src.preprocess import TextPreprocessor
from src.train import DeepSceneModels

def test_models_initialize_without_fake_model_objects():
    models=DeepSceneModels(device="cpu").initialize_all_models()
    assert models.device=="cpu"
    assert models.models=={"mood_classifier":"rules","dialogue_generator":"rules"}

@pytest.mark.parametrize(("text","expected"),[
    ("A joyful party with dancing and laughter","happy"),
    ("A detective follows a mysterious clue","mysterious"),
    ("A quiet and peaceful garden","peaceful"),
    ("An ordinary day with nothing unusual","undetermined")])
def test_mood_classification_is_deterministic(text,expected):
    model=DeepSceneModels().initialize_all_models()
    assert model.classify_scene_mood(text)["mood"]==expected
    assert model.classify_scene_mood(text)==model.classify_scene_mood(text)

def test_empty_mood_input_rejected():
    with pytest.raises(ValueError): DeepSceneModels().classify_scene_mood("  ")

def test_unknown_genre_is_general_and_deterministic(tmp_path):
    loader=SceneDataLoader(tmp_path)
    assert loader.classify_scene_genre("A completely ordinary moment")=="general"
    assert loader.classify_scene_genre("A completely ordinary moment")=="general"

def test_genre_matching(tmp_path):
    loader=SceneDataLoader(tmp_path)
    assert loader.classify_scene_genre("A detective investigates a crime")=="thriller"
    assert loader.classify_scene_genre("A couple shares a romantic kiss")=="romance"

def test_loader_falls_back_for_invalid_files(tmp_path):
    (tmp_path/"scene_templates.json").write_text("{bad",encoding="utf-8")
    assert "general" in SceneDataLoader(tmp_path).templates

def test_text_cleaning_and_alias():
    processor=TextPreprocessor(use_spacy=False); text="  This   is\n a messy text... with   spaces!!!  "
    assert processor.clean_text(text)=="This is a messy text. with spaces!"
    assert processor.clean_description(text)==processor.clean_text(text)

def test_character_detection_does_not_confuse_woman_with_man():
    assert TextPreprocessor(use_spacy=False).extract_characters("A woman waits in a station")==["Woman"]

@pytest.mark.parametrize(("text","expected"),[
    ("They meet at the coffee shop","coffee shop"),
    ("A detective hides inside a dark warehouse","dark warehouse")])
def test_setting_extraction(text,expected):
    assert expected in TextPreprocessor(use_spacy=False).extract_setting(text).lower()

def test_prompt_generation_and_dialogue():
    processor=TextPreprocessor(use_spacy=False)
    prompt=processor.generate_image_prompt("A romantic dinner","warm lighting")
    assert "A romantic dinner" in prompt and "warm lighting" in prompt and "cinematic" in prompt.lower()
    assert isinstance(DeepSceneModels().generate_dialogue("A romantic dinner"),str)

def test_image_generation_validates_before_model_load(tmp_path):
    from src.utils.io_utils import generate_ai_image_free
    with pytest.raises(ValueError): generate_ai_image_free(" ","scene",str(tmp_path))

def test_transformer_mood_uses_model_scores(monkeypatch):
    import src.train as train
    monkeypatch.setattr(train,"get_pipeline",lambda kind: lambda *a,**k:[{"label":"sadness","score":0.91},{"label":"joy","score":0.09}])
    result=DeepSceneModels(backend="transformers").classify_scene_mood("A quiet scene")
    assert result["mood"]=="sad" and result["confidence"]==0.91
    assert result["method"]=="transformers_emotion_classifier"

def test_transformer_dialogue_uses_text_generation(monkeypatch):
    import src.train as train
    monkeypatch.setattr(train,"get_pipeline",lambda kind: lambda *a,**k:[{"generated_text":"We should leave before sunrise."}])
    assert DeepSceneModels(backend="transformers").generate_dialogue("A dangerous forest")=="We should leave before sunrise."

def test_transformer_genre_uses_zero_shot_scores(monkeypatch,tmp_path):
    import src.data_loader as loader_module
    monkeypatch.setattr(loader_module,"get_pipeline",lambda kind: lambda *a,**k:{"labels":["horror","romance"],"scores":[0.87,0.13]})
    result=SceneDataLoader(tmp_path,backend="transformers").classify_scene_genre_details("A dark scene")
    assert result["genre"]=="horror" and result["method"]=="transformers_zero_shot"
