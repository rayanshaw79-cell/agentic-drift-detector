<div align="center">
  <h1>🛡️ Agentic Drift Detector</h1>
  <p><strong>An enterprise-grade, distributed AI orchestration platform for clinical environments.</strong></p>
  
  <p>
    <img src="https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white" alt="Python" />
    <img src="https://img.shields.io/badge/FastAPI-0.109-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
    <img src="https://img.shields.io/badge/LangGraph-0.1-purple?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PC9zdmc+" alt="LangGraph" />
    <img src="https://img.shields.io/badge/Streamlit-1.35-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit" />
    <img src="https://img.shields.io/badge/TimescaleDB-Postgres-F9B115?logo=postgresql&logoColor=white" alt="TimescaleDB" />
    <img src="https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white" alt="Docker" />
    <img src="https://github.com/rayanshaw79-cell/agentic-drift-detector/actions/workflows/ci.yml/badge.svg" alt="CI" />
  </p>
</div>

<hr />

## 📖 Overview

Modern LLM workflows rarely fail loudly; instead, they silently degrade. This phenomenon, known as **Agentic Drift**, manifests as escalation bias, hallucinated context retention, and rising retry loops. 

The **Agentic Drift Detector** is built to mathematically identify this drift. By analyzing execution telemetry in real-time, it detects anomalies (e.g., *Retry count exceeds historical average by 2σ*), alerts operators, and triggers deterministic circuit breakers. This ensures autonomous agents remain safe, reliable, and grounded—especially in high-stakes clinical applications.

---

## ✨ Key Features & Architectural Pillars

This project demonstrates professional system design for productionizing Large Language Models (LLMs) at scale:

*   **🤖 Deterministic AI Orchestration (LangGraph):** LLM workflows are modeled as strict state machines. If an agent loops indefinitely, the system deterministically forces an "Agentic Healing" intervention rather than crashing.
*   **⚡ Resilience & Cost Engineering:** Implements **Exponential Backoff** (via `tenacity`) for handling API rate limits and **Semantic Caching** (via `Redis` + embeddings) to bypass the LLM for previously seen reasoning paths, cutting cloud costs by up to 90%.
*   **📊 Observability & Telemetry:** Every agentic decision, retry, confidence score, and latency tick is asynchronously streamed into a **TimescaleDB** time-series database for population-level drift analysis.
*   **🛠️ Enterprise DevOps:** Fully decoupled microservice architecture (FastAPI, Streamlit UI, Background Worker, Redis, Postgres). Includes a complete **GitHub Actions CI/CD** pipeline enforcing strict `mypy` typing and `ruff` linting.

> **Note:** Read the full System Design specifications in our [ARCHITECTURE.md](ARCHITECTURE.md) and [DATA_MODEL.md](docs/DATA_MODEL.md).

---

## 🏥 Clinical Workflows: OncoLLM (PRISM & SYMPHONY)

Beyond standard incident triage, this repository features a sophisticated **Oncology Data Extraction Pipeline** based on the "Constellation" agent architecture, designed for proactive trial matching (PRISM) and longitudinal summarization (SYMPHONY).

### The Four Pillars of OncoLLM
1.  **Constellation Router:** An intelligent routing layer that classifies clinical notes (`pathology_report`, `radiology`, `genomics`) to select optimized downstream prompts.
2.  **Guideline-Grounded RAG:** An in-process ChromaDB vector store seeded with NCI/AJCC 8th Edition staging criteria grounds the LLM's diagnostic reasoning and prevents staging hallucinations.
3.  **Specialized Prompt Library:** Highly optimized, few-shot prompt factories enforce strict clinical reasoning. Every extracted biomarker or staging fact must include an `evidence_span` exactly matching the raw note.
4.  **Self-Correction Evaluator Loop:** A deterministic LangGraph critic node audits extraction outputs, rejecting hallucinations and forcing re-extraction if evidence provenance checks fail.

🚀 **Live Trial Matching:** Dynamically queries the **ClinicalTrials.gov API v2** to recommend recruiting trials based on the patient's exact histological and biomarker profile.

---

## 🚀 Getting Started

The entire enterprise architecture is containerized for seamless deployment. You can spin up the FastAPI backend, Streamlit Dashboard, Background ML Worker, Redis, and TimescaleDB with a single command.

### Prerequisites
*   [Docker](https://www.docker.com/get-started) and Docker Compose
*   *(Optional)* Google Gemini API Key for live LLM reasoning

### 1. Configure Environment
```bash
cp .env.example .env
```
*(Optionally, add your `GOOGLE_API_KEY` to the `.env` file for real Gemini LLM reasoning. If omitted, the system falls back to a weighted local simulation).*

### 2. Launch Microservices
```bash
docker compose up --build -d
```

### 3. Access the Platforms
*   **📊 Streamlit Analytics Dashboard:** [http://localhost:8501](http://localhost:8501)
*   **⚡ FastAPI Interactive Docs (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Seed Demo Telemetry
To populate the dashboard with realistic drift anomalies and clinical data without waiting for organic telemetry generation:
```bash
docker compose exec api python scripts/seed_demo_data.py
```

---

## 💻 Local Development

If you prefer to run the system natively (without Docker) for debugging or contributing:

### Setup
```bash
# 1. Install strict dependencies
pip install -r requirements.txt

# 2. Run the FastAPI backend
python -m uvicorn api.main:app --reload

# 3. In a separate terminal, launch the Dashboard
streamlit run dashboard.py
```

### CI/CD Checks
Before opening a Pull Request, ensure your code passes our static analysis gates:
```bash
python -m ruff check .
python -m mypy .
pytest
```

---

## 🤝 Contributing
We welcome contributions! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📜 License
This project is licensed under the MIT License. Built for technical exploration of Agentic AI reliability.
