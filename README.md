# Agentic Drift Detector

![Python](https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.1-purple?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PC9zdmc+)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?logo=streamlit&logoColor=white)
![TimescaleDB](https://img.shields.io/badge/TimescaleDB-Postgres-F9B115?logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![CI](https://github.com/rayanshaw79-cell/agentic-drift-detector/actions/workflows/ci.yml/badge.svg)

An enterprise-grade, distributed AI orchestration platform. Detect **behavioral drift, semantic bias, and degradation** in autonomous, agentic workflows by analyzing execution telemetry — even when the system does not explicitly crash.

---

## 🏗️ Architectural Pillars

This project demonstrates professional system design for productionizing Large Language Models (LLMs):

- **🤖 Deterministic AI Orchestration (LangGraph):** LLM workflows are modeled as state machines. If an agent loops indefinitely, the system deterministically forces an "Agentic Healing" intervention rather than crashing.
- **⚡ Resilience & Cost Engineering:** Implements **Exponential Backoff** (via `tenacity`) for API rate limits and **Semantic Caching** (via `Redis` + embeddings) to bypass the LLM for previously seen reasoning paths, cutting cloud costs by up to 90%.
- **📊 Observability & Telemetry:** Every agentic decision, retry, confidence score, and latency tick is asynchronously streamed into a **TimescaleDB** time-series database for population-level drift analysis.
- **🛠️ Enterprise DevOps:** Fully decoupled microservice architecture (FastAPI, Streamlit UI, Background Worker, Redis, Postgres). Includes a complete **GitHub Actions CI/CD** pipeline enforcing strict `mypy` typing and `ruff` linting.

> **Read the full System Design specifications in our [ARCHITECTURE.md](ARCHITECTURE.md) and [DATA_MODEL.md](docs/DATA_MODEL.md).**

---

## 🏥 OncoLLM Clinical Workflows (PRISM & SYMPHONY)

Beyond IT incident triage, this repository features a sophisticated **Oncology Data Extraction Pipeline** based on the "Constellation" agent architecture, designed for proactive trial matching (PRISM) and longitudinal summarization (SYMPHONY).

1. **Pillar 1: Constellation Router:** An intelligent routing layer that classifies clinical notes (`pathology_report`, `radiology`, `genomics`) to select optimized downstream prompts.
2. **Pillar 2: Guideline-Grounded RAG:** In-process ChromaDB vector store seeded with NCI/AJCC 8th Edition staging criteria to ground the LLM's diagnostic reasoning and prevent staging hallucinations.
3. **Pillar 3: Specialized Prompt Library:** Highly optimized few-shot prompt factories that enforce strict clinical reasoning. Every extracted biomarker or staging fact must include an `evidence_span` exactly matching the raw note.
4. **Pillar 4: Self-Correction Evaluator Loop:** A deterministic LangGraph critic node that audits extraction outputs, rejecting hallucinations and forcing re-extraction if evidence provenance checks fail.
5. **Live Trial Matching:** Dynamically queries the **ClinicalTrials.gov API v2** to recommend recruiting trials based on the patient's exact histological and biomarker profile.

---

## 🚀 Quickstart (Docker Compose)

The entire enterprise architecture is containerized. You can spin up the FastAPI backend, Streamlit Dashboard, Background ML Worker, Redis, and TimescaleDB with a single command.

### 1. Configure Environment
```bash
cp .env.example .env
```
*(Optionally, add your `GOOGLE_API_KEY` for real Gemini LLM reasoning. If omitted, the system falls back to a weighted local simulation).*

### 2. Launch the Microservices
```bash
docker compose up --build -d
```

### 3. Access the Platforms
- **Streamlit Analytics Dashboard:** [http://localhost:8501](http://localhost:8501)
- **FastAPI Interactive Docs (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Seed Demo Telemetry
To populate the dashboard with realistic drift anomalies and clinical data without waiting days for telemetry:
```bash
docker compose exec api python scripts/seed_demo_data.py
```

---

## 💻 Local Development

If you prefer to run the system natively (without Docker) for debugging or contributing:

```bash
# 1. Install strict dependencies
pip install -r requirements.txt

# 2. Run the FastAPI backend
python -m uvicorn api.main:app --reload

# 3. In a separate terminal, launch the Dashboard
streamlit run dashboard.py
```

### CI/CD Checks
Before opening a PR, ensure your code passes our static analysis gates:
```bash
python -m ruff check .
python -m mypy .
pytest
```

---

## 🚨 Understanding "Agentic Drift"

Modern LLM workflows rarely fail loudly; instead, they silently degrade:
- Escalation bias (agents become overly cautious)
- Rising retry loops and latency
- Hallucinated context retention

The **Agentic Drift Detector** identifies this drift mathematically. When it detects an anomaly (e.g., *Retry count exceeds historical average by 2σ*), it alerts human operators and triggers deterministic circuit breakers.

---

## 📜 License
MIT License. Built for technical exploration of Agentic AI reliability.
