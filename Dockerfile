# Serving image: onnxruntime only (no PyTorch), ~250 MB instead of ~2 GB
FROM python:3.12-slim
RUN useradd --create-home appuser && apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir ".[api]"
COPY api ./api
COPY models ./models
RUN chown -R appuser /app
USER appuser
ENV PYTHONPATH=/app/src:/app PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD curl -fs http://localhost:${PORT}/health || exit 1
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
