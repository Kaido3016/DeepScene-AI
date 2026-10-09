FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
ARG INSTALL_AI_DEPS=false
COPY requirements.txt ./requirements.txt
COPY requirements-ml.txt ./requirements-ml.txt
RUN pip install --no-cache-dir -r requirements.txt && \
    if [ "$INSTALL_AI_DEPS" = "true" ]; then pip install --no-cache-dir -r requirements-ml.txt; fi
COPY . .
RUN mkdir -p results/images results/exports results/logs && useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 8501 8000
CMD ["streamlit","run","streamlit_app.py","--server.address=0.0.0.0","--server.port=8501","--server.headless=true"]
