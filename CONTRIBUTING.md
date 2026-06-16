# Contributing to Agentic Drift Detector

Thank you for your interest in contributing! This document covers how to set up the project locally, run tests, and submit changes.

---

## Development Setup

### 1. Clone & create a virtual environment

```bash
git clone https://github.com/<your-username>/agentic-drift-detector
cd agentic-drift-detector
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables (optional)

```bash
cp .env.example .env
# Edit .env to add OPENAI_API_KEY, SLACK_WEBHOOK_URL, DISCORD_WEBHOOK_URL
```

> If no API key is set, the system runs in **simulation mode** automatically.

### 3. Build a baseline and launch the dashboard

```bash
python run.py --simulate-batch 60    # Build a healthy baseline
streamlit run dashboard.py           # Open the live dashboard
```

---

## Running Tests

```bash
# Run the full suite with coverage
pytest

# Run a single test file
pytest tests/test_drift_detector.py -v

# Run without coverage (faster)
pytest --no-cov
```

Tests use an **isolated in-memory SQLite database** (`conftest.py`), so they never touch your real `telemetry.db`.

---

## Project Structure

```
agentic-drift-detector/
├── alerts/           # Slack / Discord webhook alerting
├── config/           # (reserved for future config schemas)
├── drift/            # Drift signal functions + score aggregator
├── schemas/          # TypedDict state definitions
├── steps/            # LangGraph node implementations
├── telemetry/        # SQLite persistence layer
├── tests/            # pytest test suite
├── workflows/        # LangGraph StateGraph definition
├── dashboard.py      # Streamlit monitoring dashboard
└── run.py            # CLI entry point
```

---

## Pull Request Checklist

- [ ] All existing tests still pass (`pytest`)
- [ ] New behaviour is covered by at least one test
- [ ] No bare `print()` calls — use `logging.getLogger(__name__)`
- [ ] SQLite connections use context managers (`with sqlite3.connect(...) as conn`)
- [ ] New dependencies are pinned in `requirements.txt`
- [ ] Docstrings updated for any public functions changed

---

## Code Style

- **Python 3.11+** — type hints are encouraged
- **Formatting** — run `black .` before submitting
- **Imports** — stdlib → third-party → local, separated by blank lines

---

## Reporting Issues

Please open a GitHub Issue with:
1. What you expected to happen
2. What actually happened
3. Steps to reproduce (ideally a minimal test case)
