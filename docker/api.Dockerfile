# syntax=docker/dockerfile:1
# Harbor Compliance: Pinned to exact patch version for reproducibility.
# To update: test with newer tag first, then update this line.
FROM python:3.11.9-slim-bookworm

LABEL org.opencontainers.image.title="Agentic Drift Detector — API"
LABEL org.opencontainers.image.description="FastAPI Backend for Clinical Agent Workflow"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: run the FastAPI server
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
