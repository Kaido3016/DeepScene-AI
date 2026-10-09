"""Streamlit interface for transparent, rule-based scene analysis."""
from __future__ import annotations
import json
from pathlib import Path
import streamlit as st
from src.data_loader import SceneDataLoader
from src.preprocess import TextPreprocessor
from src.train import DeepSceneModels
ROOT=Path(__file__).resolve().parent
loader=SceneDataLoader(ROOT/"data")
processor=TextPreprocessor()
models=DeepSceneModels().initialize_all_models()
st.set_page_config(page_title="DeepScene AI",page_icon="🎬",layout="wide")
st.title("🎬 DeepScene AI")
st.caption("Scene analysis and prompt preparation. Mood and genre use keyword rules, not trained models.")
description=st.text_area("Describe your scene",value="A detective investigates a mysterious crime in a rainy city at night.",max_chars=5000,height=150)
style_override=st.text_input("Visual style (optional)",placeholder="e.g. neo-noir, wide-angle film still")
if st.button("Analyze scene",type="primary"):
    if len(description.strip())<3: st.error("Enter at least 3 characters to analyze a scene.")
    else:
        cleaned=processor.clean_text(description); genre=loader.classify_scene_genre(cleaned)
        style=style_override.strip() or loader.get_style_prompt(genre)
        result={"genre":genre,"mood":models.classify_scene_mood(cleaned),
                "characters":processor.extract_characters(cleaned),"setting":processor.extract_setting(cleaned),
                "dialogue":models.generate_dialogue(cleaned),"image_prompt":processor.generate_image_prompt(cleaned,style)}
        left,right=st.columns(2)
        with left:
            st.subheader("Scene analysis")
            st.write("**Genre:**",result["genre"]); st.write("**Mood:**",result["mood"]["mood"])
            st.caption("Mood confidence is a heuristic score, not a calibrated probability." if result["mood"]["confidence"] is not None else "No mood keyword match; mood is undetermined.")
            st.write("**Characters:**",", ".join(result["characters"])); st.write("**Setting:**",result["setting"])
            st.write("**Dialogue (template-based):**",result["dialogue"])
        with right:
            st.subheader("Image prompt"); st.code(result["image_prompt"],language=None)
            st.download_button("Download analysis JSON",data=json.dumps(result,indent=2),file_name="deepscene_analysis.json",mime="application/json")
            st.info("Image generation is optional and is not run automatically. Install requirements-ml.txt to enable it.")
