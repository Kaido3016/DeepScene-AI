# DeepScene AI

DeepScene is a scene-analysis and image-prompt preparation application. Default scene analysis uses deterministic keyword rules, not trained mood/genre models. Dialogue is template-based. Image generation is an optional local Diffusers integration and is never reported as successful when no image was generated.

## Requirements

- Python 3.10–3.12
- A virtual environment is recommended.
- Image generation additionally requires compatible hardware/dependencies, model weights, and acceptance of the model license where applicable.

## Local setup

From this directory, create a virtual environment, activate it, then run:

    python -m pip install -r requirements.txt
    python scripts/setup.py
    streamlit run streamlit_app.py

Run the API in another terminal:

    uvicorn app:app --reload

API endpoints:
- GET http://127.0.0.1:8000/health
- POST http://127.0.0.1:8000/analyze_scene
- Docs: http://127.0.0.1:8000/docs

Example JSON body:

    {"description":"A detective investigates a mysterious crime in a rainy city at night","style":"neo-noir"}

Mood confidence is a normalized heuristic score, not a probability. With no matching keywords, mood is undetermined and confidence is null.

## Optional image generation

Install optional dependencies with python -m pip install -r requirements-ml.txt. Set DEEPSCENE_IMAGE_MODEL to a Diffusers-compatible model repository that you can access. The first run may download large weights. The model's default safety checker is not disabled. Missing dependencies or model-loading failures produce an explicit error; no placeholder is saved under the guise of generated artwork.

## Docker

From this directory run docker compose -f docker/docker-compose.yml up --build. The UI is exposed on port 8501. The API is bound to localhost on host port 8000 by default. Containers run as a non-root user. Unused Redis and unconfigured Nginx services have been removed.

## Tests

Run python -m pytest -q. Tests cover deterministic classification, validation, fallback data loading, preprocessing, and prompt construction. They do not measure real model quality. Use labeled examples and evaluate precision/recall before describing these heuristics as ML performance.

## Structure

- streamlit_app.py: UI
- app.py: FastAPI app
- src/train.py: deterministic scene/mood and dialogue helpers
- src/data_loader.py: data loading and genre rules
- src/preprocess.py: text cleaning and extraction
- src/utils/io_utils.py: optional image generation and JSON export
- requirements-ml.txt: optional heavyweight image-generation dependencies
