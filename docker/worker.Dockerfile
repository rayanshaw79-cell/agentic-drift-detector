# syntax=docker/dockerfile:1
# Harbor Compliance: Pinned to exact patch version for reproducibility.
# To update: test with newer tag first, then update this line.
FROM python:3.11.9-slim-bookworm

LABEL org.opencontainers.image.title="Agentic Drift Detector — Worker"
LABEL org.opencontainers.image.description="Background telemetry queue drain worker"

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Healthcheck: verify the Python environment is functional
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "from telemetry.worker import run; print('ok')" || exit 1

# Default: run the background worker
CMD ["python", "-m", "telemetry.worker"]
