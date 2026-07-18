# System Architecture

Agentic Drift Detector is designed as a distributed, event-driven, microservice architecture that strictly separates stateless reasoning (LangGraph/LLMs), stateful telemetry (TimescaleDB), and background processing. 

This document details the system design via the C4 model and sequence flows.

## 1. C4 Container Architecture

The system is decoupled into 5 core Docker containers. 

```mermaid
C4Context
  title System Architecture: Agentic Drift Detector

  Person(user, "Clinician / Auditor", "Interacts with the dashboard to view clinical extraction and drift telemetry.")
  
  System_Boundary(b0, "Agentic Drift Detector") {
    Container(ui, "Streamlit Dashboard", "Python", "Provides live visualization of LLM execution paths, drift anomalies, and SDOH risk factors.")
    
    Container(api, "FastAPI Service", "Python", "Exposes REST endpoints for triggering clinical coding and incident triage workflows.")
    
    Container(worker, "Background Worker", "Python", "Asynchronously processes telemetry queues and runs Drift ML models.")
    
    ContainerDb(db, "TimescaleDB", "PostgreSQL", "Stores time-series telemetry data, clinical outcomes, and execution steps.")
    
    ContainerDb(cache, "Redis", "Key-Value Store", "Semantic cache for LLM responses and task queues for background workers.")
  }
  
  System_Ext(gemini, "Google Gemini API", "LLM for reasoning, triage, and semantic embeddings.")
  System_Ext(nlm, "NLM RxNav API", "External API for drug/condition validation.")

  Rel(user, ui, "Views dashboards via", "HTTPS")
  Rel(ui, db, "Reads telemetry from", "psycopg2")
  Rel(api, db, "Persists clinical state to", "SQLAlchemy")
  Rel(api, cache, "Checks semantic cache in", "redis-py")
  Rel(api, gemini, "Performs zero-shot reasoning via", "gRPC / HTTPS")
  Rel(api, nlm, "Validates terms via", "HTTPS")
  Rel(worker, cache, "Consumes events from", "Redis Streams")
  Rel(worker, db, "Writes ML drift factors to", "psycopg2")
```

## 2. Clinical Workflow Engine (LangGraph)

The core intelligent processing is handled via LangGraph, treating LLM executions as a robust state machine. This allows for cyclical reasoning, deterministic fallbacks, and agentic healing.

```mermaid
stateDiagram-v2
    [*] --> TriageStep : Incoming Incident
    
    state TriageStep {
        [*] --> SemanticCacheCheck
        SemanticCacheCheck --> LLM_Invoke : Cache Miss
        SemanticCacheCheck --> Return_Cached : Cache Hit
    }
    
    TriageStep --> InvestigationStep
    InvestigationStep --> DecisionStep
    
    DecisionStep --> InterventionStep : Drift Loop Detected (retries >= 2)
    DecisionStep --> NotificationStep : High Confidence
    DecisionStep --> InvestigationStep : Low Confidence (Retry)
    
    InterventionStep --> NotificationStep : Agentic Healing Applied
    
    NotificationStep --> [*]
```

## 3. Resilience & Cost Engineering Sequence

To prevent API rate-limit crashes and reduce cloud costs, we employ a **Semantic Caching** and **Exponential Backoff** strategy around the Gemini API.

```mermaid
sequenceDiagram
    participant API as FastAPI Backend
    participant Cache as Redis (LangChain Cache)
    participant Embed as Google Embeddings Model
    participant LLM as Google Gemini Model

    API->>Embed: Embed Prompt text
    Embed-->>API: Vector [0.1, 0.4, ...]
    
    API->>Cache: Vector Search (Similarity > 0.95)
    
    alt Cache Hit (Exact or Semantic Match)
        Cache-->>API: Return Cached MEAT Validation
    else Cache Miss
        API->>LLM: Invoke Model (Attempt 1)
        
        alt 429 ResourceExhausted
            LLM-->>API: Error 429
            Note over API: Tenacity applies Exponential Backoff
            API->>LLM: Invoke Model (Attempt 2)
        end
        
        LLM-->>API: Return Completion
        API->>Cache: Store Vector + Completion mapping
    end
```

## 4. Key Architectural Decisions (ADRs)

1. **State Machine over Chains**: We chose LangGraph over standard LangChain sequential chains. LLMs are non-deterministic; by using a state graph, we can tightly control retry loops and force "Agentic Healing" (deterministic intervention) if the LLM enters an infinite drift loop.
2. **Semantic Caching**: In clinical settings, the same condition strings (e.g., "Type 2 DM w/ neuropathy") appear thousands of times. Rather than paying the LLM token cost every time, the vector cache intercepts similar queries and responds in milliseconds.
3. **Decoupled Telemetry**: The LLM engine does not block on telemetry writes. Telemetry is emitted and processed asynchronously, ensuring the critical reasoning path remains fast.
