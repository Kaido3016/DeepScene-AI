from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from fastapi.testclient import TestClient
from app import app
client=TestClient(app)

def test_health_endpoint_reports_ready():
    response=client.get("/health")
    assert response.status_code==200
    assert response.json()["status"]=="ok"

def test_analyze_scene_returns_consistent_schema():
    response=client.post("/analyze_scene",json={"description":"A detective investigates a crime in a warehouse","style":"neo-noir"})
    assert response.status_code==200
    payload=response.json()
    assert payload["genre"]=="thriller"
    assert payload["mood"]["mood"]=="mysterious"
    assert "image_prompt" in payload and "dialogue" in payload

def test_api_rejects_too_short_input():
    response=client.post("/analyze_scene",json={"description":"x"})
    assert response.status_code==422

def test_api_rejects_oversized_input():
    response=client.post("/analyze_scene",json={"description":"x"*5001})
    assert response.status_code==422
