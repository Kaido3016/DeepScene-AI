"""Robust text normalization and lightweight scene extraction."""
from __future__ import annotations
import re
from typing import List
try: import spacy
except ImportError: spacy = None
_SPACE_RE = re.compile(r"\s+")
_SETTING_RE = re.compile(r"\b(?:in|at|inside|outside|near|on|within)\s+(?:a|an|the)\s+([^,.!?;]+?(?:warehouse|room|house|building|street|park|forest|beach|office|school|restaurant|bar|club|studio|stage|theater|theatre|station|coffee shop|city|village))\b", re.I)
_ROLE_PATTERNS = {"Detective":r"\bdetective\b","Dancer":r"\bdancers?\b","Man":r"\bman\b","Woman":r"\bwoman\b","Person":r"\bperson\b"}
class TextPreprocessor:
    def __init__(self, use_spacy: bool = True):
        self.nlp = None
        if use_spacy and spacy is not None:
            try: self.nlp = spacy.load("en_core_web_sm")
            except (OSError, ImportError): pass
    def clean_text(self, text: str) -> str:
        if not isinstance(text,str): raise TypeError("text must be a string")
        text = _SPACE_RE.sub(" ",text.replace("\r"," ").replace("\n"," ").replace("\t"," ")).strip()
        return re.sub(r"([.!?])\1+",r"\1",text)
    clean_description = clean_text
    def extract_characters(self, description: str) -> List[str]:
        text=self.clean_text(description); found=[]
        if self.nlp:
            for entity in self.nlp(text).ents:
                if entity.label_=="PERSON" and entity.text not in found: found.append(entity.text)
        for role,pattern in _ROLE_PATTERNS.items():
            if re.search(pattern,text,re.I) and role not in found: found.append(role)
        return found or ["Unspecified character"]
    def extract_setting(self, description: str) -> str:
        text=self.clean_text(description)
        if self.nlp:
            for entity in self.nlp(text).ents:
                if entity.label_ in {"GPE","LOC","FAC"}: return entity.text
        match=_SETTING_RE.search(text)
        return match.group(1).strip() if match else "Unspecified setting"
    def generate_image_prompt(self, description: str, style: str) -> str:
        description=self.clean_text(description)
        style_clean=re.sub(r"^style\s*:\s*","",str(style),flags=re.I).strip()
        parts=[description]
        if style_clean: parts.append(f"cinematic style, {style_clean}")
        parts.append("detailed film still, intentional composition, cinematic lighting")
        return ", ".join(parts)
