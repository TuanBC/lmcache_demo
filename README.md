# Bank Multi-Agent Expert System

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-purple.svg)](https://github.com/langchain-ai/langgraph)

High-efficiency multi-agent system with **KV Cache Optimization** for querying the Internal Operations & Compliance Manual (~25,000 tokens).

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| **Prefix Caching** | 25k-token manual loaded once, cached for all agents via LMCache |
| **Parallel Execution** | Fan-out to multiple specialized agents using LangGraph |
| **TTFT Optimization** | Inferred cache hit rate from response times |
| **Langfuse Observability** | Full tracing and monitoring from day one |
| **Prompty Templates** | All prompts defined in `.prompty` files for maintainability |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  LOCAL MACHINE (FastAPI + LangGraph Orchestration)          │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  • DeterministicPromptBuilder (whitespace norm)     │    │
│  │  • CacheAwareMetrics (TTFT tracking)                │    │
│  │  • Langfuse integration                             │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ HTTP (OpenAI API)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  REMOTE vLLM SERVER (GPU)                                   │
│  • LMCache with CacheBlend                                  │
│  • Prefix caching for shared manual                         │
│  • Qwen/Qwen3-30B-A3B-Instruct-2507                         │
└─────────────────────────────────────────────────────────────┘
```

### Workflow

```
START → Router → Parallel Agents → Aggregator → END
```

The **Router** classifies queries and selects 1-3 agents. All agents share the same 25k-token manual prefix, maximizing KV cache hits.

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Orchestration** | LangGraph 0.2+ |
| **LLM Framework** | LangChain + LangChain-OpenAI |
| **API** | FastAPI 0.115+ |
| **Prompt Management** | Prompty |
| **Observability** | Langfuse |
| **LLM Backend** | Remote vLLM with LMCache |
| **Package Manager** | uv |

## 📋 Prerequisites

- **Python 3.13+**
- **uv** package manager ([install guide](https://docs.astral.sh/uv/))
- Access to remote vLLM endpoint (or local GPU setup)
- Langfuse account for observability

## 🚀 Quick Start

```bash
# 1. Clone and navigate to project
cd test_tensormesh

# 2. Install dependencies
uv sync

# 3. Configure environment
cp .env.example .env
# Edit .env with your API keys and endpoint

# 4. Run the server
uv run uvicorn src.main:app --reload

# 5. Test a query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the daily ATM withdrawal limit?", "session_id": "test-123"}'
```

## 📁 Project Structure

```
test_tensormesh/
├── src/
│   ├── main.py                  # FastAPI app with cache warming
│   ├── api/
│   │   ├── routes.py            # REST endpoints
│   │   └── schemas.py           # Pydantic request/response models
│   ├── config/
│   │   ├── settings.py          # Pydantic settings management
│   │   └── langfuse.py          # Langfuse observability setup
│   ├── cache/
│   │   └── metrics.py           # CacheAwareMetrics for TTFT tracking
│   ├── prompts/
│   │   ├── manager.py           # DeterministicPromptBuilder
│   │   ├── router.prompty       # Router agent prompt
│   │   ├── technical_specialist.prompty
│   │   ├── compliance_auditor.prompty
│   │   ├── support_concierge.prompty
│   │   ├── aggregator.prompty   # Response aggregation prompt
│   │   └── warmup.prompty       # Cache warming prompt
│   └── graph/
│       ├── builder.py           # LangGraph workflow builder
│       ├── nodes.py             # Node implementations
│       └── state.py             # AgentState TypedDict
├── data/
│   └── operations_manual.txt    # 25k-token bank operations manual
├── tests/
│   ├── conftest.py              # Pytest fixtures
│   └── test_cache_efficiency.py # Cache optimization tests
├── .env.example                 # Environment template
├── pyproject.toml               # Dependencies and project config
└── AGENTS.MD                    # Development guidelines
```

## 🔌 API Endpoints

### `POST /api/v1/query`

Process a user query through the multi-agent workflow.

**Request:**
```json
{
  "query": "What is the daily ATM withdrawal limit?",
  "session_id": "user-session-123",
  "user_id": "optional-user-id"
}
```

**Response:**
```json
{
  "response": "According to Section 3.2 of the manual...",
  "agents_used": ["technical_specialist", "compliance_auditor"],
  "compliance_passed": true,
  "retry_count": 0,
  "ttft_seconds": 2.34
}
```

### `GET /health`

Health check endpoint.

```bash
curl http://localhost:8000/health
```

### `GET /cache/stats`

Cache efficiency metrics for monitoring.

```bash
curl http://localhost:8000/cache/stats
```

**Response:**
```json
{
  "total_requests": 100,
  "inferred_cache_hit_rate": 0.85,
  "prefix_alignment_ok": true,
  "grade": "A - Excellent cache efficiency"
}
```

## 🔧 Configuration

All settings are managed via environment variables. Copy `.env.example` to `.env` and configure:

| Variable | Description | Default |
|----------|-------------|---------|
| `VLLM_BASE_URL` | Remote vLLM endpoint | `http://89.169.108.198:30080/v1` |
| `VLLM_API_KEY` | API key for vLLM | - |
| `VLLM_MODEL` | Model name | `Qwen/Qwen3-30B-A3B-Instruct-2507` |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key | - |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key | - |
| `LANGFUSE_BASE_URL` | Langfuse endpoint | `https://us.cloud.langfuse.com` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `API_HOST` | API host | `0.0.0.0` |
| `API_PORT` | API port | `8000` |
| `MANUAL_PATH` | Path to operations manual | `data/operations_manual.txt` |

## 📊 Cache Efficiency

### Optimization Strategy

| Optimization | Implementation | Expected Impact |
|--------------|----------------|-----------------|
| Deterministic prompts | Whitespace normalization, Unix line endings | 100% prefix match |
| Parallel batching | Fire all agent requests simultaneously | Max cache reuse |
| Startup warming | Pre-load manual on app start via `warmup.prompty` | Fast first request |
| TTFT tracking | Infer cache hits from response time | Visibility |

### Expected Performance

| Metric | Cold Cache | Warm Cache | Improvement |
|--------|------------|------------|-------------|
| TTFT | ~10-30s | ~1-5s | **3-10x faster** |
| Cache Hit Rate | 0% | >80% | - |

## 🧪 Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run cache efficiency tests only
uv run pytest tests/test_cache_efficiency.py -v
```

### Test Coverage

- **Prefix alignment tests**: Verify all agents produce identical prefix hashes
- **TTFT inference tests**: Validate cache hit detection from response times
- **Whitespace normalization**: Ensure consistent prompt formatting

## 🏛️ Key Components

| Component | Purpose |
|-----------|---------|
| `DeterministicPromptBuilder` | Ensures byte-identical prompts for cache hits |
| `CacheAwareMetrics` | Tracks TTFT and infers cache efficiency |
| `AgentState` | LangGraph state with Annotated reducers for parallel execution |
| `build_graph()` | Constructs the LangGraph workflow with checkpointer support |

## 📝 License

Internal use only.
