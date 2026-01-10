# Bank Multi-Agent Expert System

High-Efficiency Multi-Agent system with **KV Cache Optimization** for querying the Internal Operations & Compliance Manual.

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| **Prefix Caching** | 25k-token manual loaded once, cached for all agents |
| **Parallel Execution** | Fan-out to multiple specialized agents |
| **TTFT Optimization** | Inferred cache hit rate from response times |
| **Human Escalation** | Uncertainty markers flag responses for review |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  LOCAL MACHINE (CPU, 16GB RAM)                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  FastAPI + LangGraph Orchestration                  │    │
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
│  • Qwen3-30B-A3B-Instruct                                   │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 3. Run the server
uv run uvicorn src.main:app --reload

# 4. Test a query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the daily ATM withdrawal limit?", "session_id": "test-123"}'
```

## 📊 Cache Efficiency Report

### Optimization Strategy

| Optimization | Implementation | Expected Impact |
|--------------|----------------|-----------------|
| Deterministic prompts | Whitespace normalization, Unix line endings | 100% prefix match |
| Parallel batching | Fire all agent requests simultaneously | Max cache reuse |
| Startup warming | Pre-load manual on app start | Fast first request |
| TTFT tracking | Infer cache hits from response time | Visibility |

### Expected Performance

| Metric | Cold Cache | Warm Cache | Improvement |
|--------|------------|------------|-------------|
| TTFT | ~10-30s | ~1-5s | **3-10x faster** |
| Cache Hit Rate | 0% | >80% | - |

### Monitoring Endpoint

```bash
# Get cache efficiency metrics
curl http://localhost:8000/cache/stats
```

Response:
```json
{
  "total_requests": 100,
  "inferred_cache_hit_rate": 0.85,
  "prefix_alignment_ok": true,
  "grade": "A - Excellent cache efficiency"
}
```

## 🧪 Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run cache efficiency tests only
uv run pytest tests/test_cache_efficiency.py -v
```

## 📁 Project Structure

```
bank-multi-agent/
├── src/
│   ├── main.py           # FastAPI app with cache warming
│   ├── config/           # Settings & Langfuse setup
│   ├── cache/            # CacheAwareMetrics
│   ├── prompts/          # DeterministicPromptBuilder
│   ├── graph/            # LangGraph workflow
│   └── api/              # REST endpoints
├── data/
│   └── operations_manual.txt  # 25k-token manual
├── tests/
│   └── test_cache_efficiency.py
├── AGENTS.MD             # Development rules
└── pyproject.toml
```

## 🔧 Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `VLLM_BASE_URL` | Remote vLLM endpoint | - |
| `VLLM_API_KEY` | API key for vLLM | - |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key | - |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key | - |

## 📝 License

Internal use only.
