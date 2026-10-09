# DeepScene AI

DeepScene analyzes scene descriptions, prepares image prompts, and optionally uses real pretrained models. It does not train these pretrained models itself.

## Quick start

Use Python 3.10–3.12 and a virtual environment:

    python -m pip install -r requirements.txt
    python scripts/setup.py
    streamlit run streamlit_app.py

In another terminal: \`uvicorn app:app --reload\`. API docs are at http://127.0.0.1:8000/docs.

## Real pretrained AI inference

The dependency-light default is transparent rules/templates. To enable real pretrained inference, install the optional dependencies and configure the backend:

    python -m pip install -r requirements-ml.txt
    # Set DEEPSCENE_INFERENCE_BACKEND=transformers in your environment
    streamlit run streamlit_app.py

Model weights download on first use and are cached by Transformers. The defaults are:
- Mood: \`j-hartmann/emotion-english-distilroberta-base\`, an emotion classifier mapped to DeepScene mood labels.
- Genre: \`valhalla/distilbart-mnli-12-1\`, a zero-shot classifier over the configured genre labels.
- Dialogue: \`google/flan-t5-base\`, a pretrained text-to-text generator.

Set \`DEEPSCENE_MOOD_MODEL\`, \`DEEPSCENE_GENRE_MODEL\`, or \`DEEPSCENE_DIALOGUE_MODEL\` to override model IDs. The app loads models lazily; first inference may take time and require substantial RAM/VRAM and network access. It fails visibly if the requested model cannot load rather than silently changing back to rules. Model scores are not necessarily calibrated probabilities. Results require human review; generated dialogue may be incorrect or unsuitable.

To use the rule baseline, set \`DEEPSCENE_INFERENCE_BACKEND=rules\`.

### Docker with AI models

For a CPU-based Docker build that includes Transformers/Diffusers/PyTorch, set \`INSTALL_AI_DEPS=true\` and \`DEEPSCENE_INFERENCE_BACKEND=transformers\` before building. This creates a significantly larger image. CUDA/GPU support requires a host, container runtime, and PyTorch build compatible with your GPU; the default slim CPU image does not promise GPU inference.

## Image-generation validation

Install a compatible PyTorch build and the optional requirements. Then run:

    python scripts/test_image_generation.py

This downloads/loads the configured Diffusers model if needed, performs a real inference, and verifies that the output is a readable image. It can take several minutes and needs substantial RAM/VRAM. This smoke test was not run in GitHub CI because CI does not provision model weights or GPU hardware.

## Evaluation

Run \`python scripts/evaluate_models.py\` for the labeled smoke-test set, or pass \`--dataset path/to/your_labeled_data.json\`. The JSON schema is a list of objects with \`id\`, \`description\`, \`genre\`, and \`mood\`. It reports accuracy and macro-F1 for the configured backend. The included 21 examples are a small hand-labeled development fixture, not an independent benchmark. For credible evaluation, use hundreds or thousands of representative, independently labeled scenes, a held-out split, per-class precision/recall/F1, confidence calibration, and human review of generated dialogue.

## API and security

- \`GET /health\` reports the configured backend; model weights are loaded lazily.
- \`POST /analyze_scene\` accepts JSON such as \`{"description":"A detective investigates a mysterious crime","style":"neo-noir"}\`.
- Set \`DEEPSCENE_API_KEY\` to require \`X-API-Key\` for analysis requests.
- Set \`DEEPSCENE_REQUIRE_API_KEY=1\` to refuse application startup if the key is missing.
- CORS is not authentication. Never commit secrets or expose the API directly to the internet without authentication and TLS.

Local Docker development:

    docker compose -f docker/docker-compose.yml up --build

Both application ports are bound to localhost. For HTTPS deployment, point a DNS A/AAAA record at a server you control, allow inbound ports 80 and 443, set \`CADDY_DOMAIN\` and a long random \`DEEPSCENE_API_KEY\`, then run:

    DEEPSCENE_API_KEY='use-a-long-random-secret' CADDY_DOMAIN='your.domain.example' docker compose -f docker/docker-compose.yml --profile public up --build -d

The public-profile API refuses to start without the key. Caddy obtains and renews certificates automatically when the domain/DNS/network configuration supports it. For pretrained inference in containers, also set \`INSTALL_AI_DEPS=true\` and \`DEEPSCENE_INFERENCE_BACKEND=transformers\`. Before production, use a secret manager, configure trusted CORS origins, restrict inbound ports, pin/scan dependencies and images, monitor logs, rate-limit requests at the edge, and validate the host/provider's backups and patching. This is a deployment template, not a substitute for a security review.

## Tests

    python -m pytest -q

Tests mock model pipelines to validate integration behavior without downloading large weights. Passing tests do not prove model quality or GPU compatibility.
