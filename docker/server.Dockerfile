# The unified sparkctl server (gateway proxy + metrics aggregation + dash) for server.mode: docker.
# LiteLLM runs as a sidecar container (official image) — this image stays small.
FROM python:3.12-slim
COPY pyproject.toml README.md /app/
COPY src /app/src
RUN pip install --no-cache-dir /app fastapi uvicorn httpx
CMD ["python", "-m", "sparkctl.server"]
