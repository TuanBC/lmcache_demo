# Bank Multi-Agent Expert System

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-purple.svg)](https://github.com/langchain-ai/langgraph)

High-efficiency multi-agent system with **KV Cache Optimization** for querying a 25,000-token Internal Operations & Compliance Manual.

---

## Table of Contents

1. [Overview](#overview)
2. [KV Cache Optimization Report](#-kv-cache-optimization-report)
3. [Quick Start](#-quick-start)
4. [Architecture](#-architecture)
5. [Benchmarks](#-benchmarks)
6. [Testing & Scripts](#-testing--scripts)
7. [API Reference](#-api-reference)
8. [Configuration](#-configuration)
9. [Project Structure](#-project-structure)

---

## Overview

This system demonstrates **production-ready KV cache optimization** for LLM inference. The key challenge: efficiently serving a multi-agent workflow where each agent needs access to a large (~25k token) operations manual without re-computing attention for the shared context.

### Key Features

| Feature | Description |
|---------|-------------|
| **Prefix Caching** | 25k-token manual loaded once, cached for all agents |
| **Parallel Execution** | Fan-out to 1-3 specialized agents via LangGraph |
| **TTFT Optimization** | 1.7x faster warm requests vs cold |
| **Cache-Aware Metrics** | Inferred hit rate from response times |
| **Langfuse Observability** | Full tracing and monitoring |

---

## 📊 KV Cache Optimization Report

### The Efficiency Challenge

When multiple agents share a large context (25k tokens), naive implementations recompute the entire KV cache for each agent call. This system solves that by ensuring:

1. **Unified Prefix Structure** - All agents share byte-identical prompts up to the manual boundary
2. **Static vs Dynamic Separation** - Large static content (manual) comes before dynamic content (agent instructions)
3. **Deterministic Formatting** - Whitespace normalization ensures cache key stability

### Prompt Template Strategy

All `.prompty` templates follow this structure to maximize cache hits:

```
┌─────────────────────────────────────┐
│  STATIC PREFIX (cached)             │  ← Same across ALL agents
│  ├── System introduction            │
│  ├── 25k-token Operations Manual    │
│  └── <<< END OF MANUAL >>>          │
├─────────────────────────────────────┤
│  DYNAMIC SUFFIX (not cached)        │  ← Agent-specific
│  ├── Agent role instructions        │
│  ├── Conversation history           │
│  └── User query                     │
└─────────────────────────────────────┘
```

**Why this works:** vLLM/LMCache caches KV values by token prefix. By placing the 25k-token manual BEFORE agent-specific instructions, all agents reuse the same cached prefix.

### TTFT Reduction for Sequential Agents

When the Router → Agent → Aggregator workflow executes:

| Agent | TTFT Behavior |
|-------|---------------|
| **Router** | Cold start (~2.3s) - computes full 25k prefix |
| **Technical Specialist** | Warm (~1.3s) - reuses cached prefix |
| **Aggregator** | Warm (~1.3s) - reuses cached prefix |

**Result:** 2nd and 3rd agents are **1.7x faster** because they skip prefix computation.

### High Cache Hit Rate Strategy

| Technique | Implementation | Impact |
|-----------|----------------|--------|
| **Unified prompty templates** | All agents have identical prefix structure through `<<< END OF MANUAL >>>` | 100% prefix match |
| **Whitespace normalization** | `DeterministicPromptBuilder` strips trailing whitespace, uses Unix line endings | Byte-identical prompts |
| **Tokenizer verification** | `get_tokenizer()` loads the actual model tokenizer for accurate counting | Debug visibility |
| **Prefix hash tracking** | `CacheAwareMetrics` hashes the cacheable prefix and alerts on mismatches | Early detection |

### Production Monitoring

Cache efficiency is monitored via three methods:

#### 1. Application Logs
```
[CACHE] ⏱️ Baseline set: 2.28s (first request)
[CACHE] Agent=technical_specialist TTFT=1.32s -> LIKELY HIT ✅ (58% of baseline)
```

#### 2. Langfuse Traces
Each LLM call is traced with:
- `ttft_seconds` - Time to first token
- `prefix_hash` - Hash of the cacheable prefix
- `cache_hit_inferred` - Boolean based on TTFT ratio

#### 3. Prometheus Metrics (via `/cache/stats`)
```json
{
  "total_requests": 100,
  "inferred_cache_hit_rate": 0.85,
  "prefix_alignment_ok": true,
  "cold_cache_baseline_seconds": 2.28
}
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Configure environment
cp .env.example .env
# Edit .env with your vLLM endpoint and Langfuse keys

# 3. Run the server
uv run uvicorn src.main:app --reload --port 8000

# 4. Test a query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the CTR filing threshold?", "session_id": "test-1"}'
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  CLIENT APPLICATION                                          │
│  FastAPI + LangGraph Orchestration                           │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  DeterministicPromptBuilder (whitespace norm)       │    │
│  │  CacheAwareMetrics (TTFT tracking + hit inference)  │    │
│  │  Langfuse integration (traces, spans, metrics)      │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ OpenAI-compatible API
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  vLLM SERVER (Remote GPU)                                    │
│  • --enable-prefix-caching flag                              │
│  • LMCache with CacheBlend (chunk_size: 256)                 │
│  • Qwen/Qwen3-30B-A3B-Instruct-2507                          │
└─────────────────────────────────────────────────────────────┘
```

### Workflow

```
START → Router → Parallel Agents (1-3) → Aggregator → END
```

1. **Router** classifies the query and selects relevant agents
2. **Parallel Agents** execute simultaneously (fan-out)
3. **Aggregator** synthesizes responses into a final answer

All agents share the same 25k-token prefix, maximizing KV cache reuse.

---

## 📈 Benchmarks

### TTFT Comparison Results

| Metric | Direct Router | End-to-End API |
|--------|---------------|----------------|
| **Cold** | 2.285s | 4.187s |
| **Warm 1** | 1.589s | 3.346s |
| **Warm 2** | 1.320s | 3.745s |
| **Best Warm** | 1.320s | 3.346s |
| **Speedup** | **1.7x** | **1.3x** |

> **Direct Router** = Pure LLM call time (no API overhead)  
> **End-to-End API** = Full workflow including HTTP, routing, aggregation

> ⚠️ **Note:** Cache warm-up is **disabled** in `src/main.py` to enable accurate cold vs warm TTFT benchmarking. For production, uncomment the warm-up block to pre-populate the KV cache on startup.

### How to Run Benchmarks

```bash
# Full cold vs warm comparison
uv run python tests/test_cache_ttft.py

# Direct router agent test only
uv run python tests/test_direct_ttft.py

# Reset cache metrics between runs
curl -X POST http://localhost:8000/cache/reset
```


---

## 🧪 Testing & Scripts

### TTFT Benchmark Test
```bash
uv run python tests/test_cache_ttft.py
```
Measures both direct LLM and end-to-end API performance. Grades: A (>2x), B (>1.5x), C (>1x), F (no improvement).

### Unit Tests
```bash
uv run pytest tests/test_cache_efficiency.py -v
```
- `test_prefix_hash_identical_across_agents` - Verifies unified templates
- `test_whitespace_normalization` - Ensures deterministic formatting
- `test_cache_hit_inference` - Validates TTFT-based hit detection

### Interactive Chat
```bash
uv run python scripts/chat.py
```
Multi-turn conversation with real-time cache metrics.

### Cache Benchmark
```bash
uv run python scripts/benchmark_cache.py
```
Comprehensive cache performance analysis.

---

## 🔌 API Reference

### `POST /api/v1/query`
Process a query through the multi-agent workflow.

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the daily ATM limit?", "session_id": "user-123"}'
```

### `POST /api/v1/query/stream`
Streaming version with real-time token output.

### `GET /cache/stats`
Cache efficiency metrics.

```bash
curl http://localhost:8000/cache/stats
```

### `POST /cache/reset`
Reset cache baseline for fresh measurements.

```bash
curl -X POST http://localhost:8000/cache/reset
```

### `GET /health`
Health check.

---

## 🔧 Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `VLLM_BASE_URL` | vLLM endpoint | `http://89.169.108.198:30080/v1` |
| `VLLM_MODEL` | Model name | `Qwen/Qwen3-30B-A3B-Instruct-2507` |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key | - |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key | - |
| `MANUAL_PATH` | Path to operations manual | `data/operations_manual.txt` |

---

## 📁 Project Structure

```
test_tensormesh/
├── src/
│   ├── main.py                  # FastAPI entry point
│   ├── api/routes.py            # REST endpoints
│   ├── cache/metrics.py         # CacheAwareMetrics
│   ├── prompts/
│   │   ├── manager.py           # DeterministicPromptBuilder
│   │   ├── router.prompty       # Router agent (unified prefix)
│   │   ├── technical_specialist.prompty
│   │   ├── compliance_auditor.prompty
│   │   └── support_concierge.prompty
│   └── graph/
│       ├── builder.py           # LangGraph workflow
│       └── nodes.py             # Node implementations
├── scripts/
│   ├── chat.py                  # Interactive CLI
│   └── benchmark_cache.py       # Cache benchmarking
├── tests/
│   ├── test_cache_ttft.py       # TTFT comparison test
│   └── test_cache_efficiency.py # Unit tests
├── data/
│   └── operations_manual.txt    # 25k-token manual
└── lmcache-config.yaml          # LMCache settings
```

---

## Key Components

| Component | Purpose |
|-----------|---------|
| `DeterministicPromptBuilder` | Normalizes prompts for byte-identical cache keys |
| `CacheAwareMetrics` | Tracks TTFT, infers cache hits (80% threshold) |
| `get_tokenizer()` | Lazy-loads model tokenizer for accurate counting |
| `build_graph()` | Constructs LangGraph workflow with parallel execution |

---

## License

Internal use only.
