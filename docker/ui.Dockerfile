# syntax=docker/dockerfile:1
FROM python:3.11-slim

LABEL org.opencontainers.image.title="Agentic Drift Detector — Dashboard"
LABEL org.opencontainers.image.description="Streamlit UI for telemetry visualization"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default: run the Streamlit dashboard
CMD ["python", "-m", "streamlit", "run", "dashboard.py", "--server.address", "0.0.0.0"]
