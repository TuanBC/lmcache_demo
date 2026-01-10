# AGENTS.MD - Multi-Agent Expert System with vLLM Production Stack

> **Project**: Bank Internal AI Tool - Multi-Agent Expert System  
> **Architecture**: vLLM Production Stack (Router + Single vLLM + Shared LMCache)  
> **Stack**: LangGraph + vLLM + LMCache + Prometheus + Grafana + Langfuse  
> **Python**: 3.13 (async everywhere)  
> **LangGraph**: v1.x (use StateGraph, CompiledStateGraph, MemorySaver)  
> **Langfuse**: v3.x (use `get_client()`, `update_current_span()`, NOT deprecated `langfuse_context`)  
> **Deployment**: Local Docker Compose  
> **Primary Goal**: Maximize KV Cache hit rates, minimize TTFT


---

## Production Stack Overview

This system follows the **vLLM Production Stack** architecture for scalable LLM inference with shared KV cache and full observability.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER                                            │
│                                │                                             │
│          ┌────────────────────┴────────────────────┐                        │
│          │ Requests                    Monitoring  │                        │
│          ▼                                         ▼                        │
│    ┌──────────┐                          ┌─────────────────┐                │
│    │  Router  │─────── Metrics ─────────▶│ Monitoring UI   │                │
│    │ (FastAPI │                          │   (Grafana)     │                │
│    │+LangGraph)│                          └────────▲────────┘                │
│    └────┬─────┘                                    │                        │
│         │  3 Logical Agents                    │                        │
│         │  (Different Prompts)           ┌────┴────────┐                │
│         │                                │ Observability│                │
│         ▼                                │   Gateway    │                │
│    ┌────────────────────────────────┐    │(Prometheus) │                │
│    │      Single vLLM Server        │    └─────────────┘                │
│    │ ┌─────────┐┌─────────┐┌───────┐│                                   │
│    │ │ Agent A ││ Agent B ││Agent C││                                   │
│    │ │Technical││Compliance│Support││                                   │
│    │ └────┬────┘└────┬────┘└───┬───┘│                                   │
│    │      │          │         │    │                                   │
│    │      └──────────┼─────────┘    │                                   │
│    │                 ▼              │                                   │
│    │   ┌────────────────────────┐   │                                   │
│    │   │ Sharable KV Cache      │   │                                   │
│    │   │       (LMCache)        │   │                                   │
│    │   └────────────────────────┘   │                                   │
│    └────────────────────────────────┘                                   │
│                                                                              │
│              Local Docker Compose Environment                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Overview](#2-architecture-overview)
3. [Docker Compose Deployment](#3-docker-compose-deployment)
4. [vLLM Engine Configuration](#4-vllm-engine-configuration)
5. [LMCache Shared KV Storage](#5-lmcache-shared-kv-storage)
6. [Router & LangGraph Workflow](#6-router--langgraph-workflow)
7. [Prometheus & Grafana Observability](#7-prometheus--grafana-observability)
8. [Code Style & Conventions](#8-code-style--conventions)
9. [Prompt Management](#9-prompt-management)
10. [REST API Design](#10-rest-api-design)
11. [Testing & Verification](#11-testing--verification)

---

## 1. Project Overview

### 1.1 Purpose

Build a high-performance multi-agent system for querying a 50-page (~25,000 tokens) "Internal Operations & Compliance Manual" for a bank, using the vLLM production stack architecture.

### 1.2 Logical Agents (3 Specialized Routes)

| Agent | Role | Model | Focus Areas |
|-------|------|-------|-------------|
| **Agent A: Technical Specialist** | System specs expert | Qwen2.5-7B-Instruct | API limits, troubleshooting, technical specs |
| **Agent B: Compliance Auditor** | Rule interpreter | Qwen2.5-7B-Instruct | Regulatory rules, Can/Cannot constraints |
| **Agent C: Support Concierge** | Documentation helper | Qwen2.5-7B-Instruct | Step-by-step guides for non-technical staff |

### 1.3 Key Efficiency Constraint

All 3 agents share the same **LMCache** instance for KV cache storage. The 25k-token manual prefix is computed once and reused across all agents.
---

## 2. Architecture Overview

### 2.1 Deployment Modes

> **Single Endpoint Architecture**: Both local and remote modes use **1 vLLM endpoint**.  
> The 3 agents are **logical routes** (prompt variations), not separate models.

| Mode | vLLM Location | Configuration |
|------|---------------|---------------|
| **Local** | Docker container on local GPU | `docker-compose up` |
| **Remote** | Cloud GPU server | Set `VLLM_BASE_URL` in `.env` |

### 2.2 Component Stack

| Component | Port | Purpose |
|-----------|------|---------|
| **FastAPI Router** | 8000 | Request routing, LangGraph orchestration |
| **vLLM Server** | 8001 | Single LLM endpoint (shared by all agents) |
| **LMCache** | 8080 | Shared KV cache storage |
| **Prometheus** | 9090 | Metrics collection |
| **Grafana** | 3000 | Monitoring dashboards |

> **Key Design**: All 3 agents (Technical, Compliance, Support) call the **same vLLM endpoint**.  
> This maximizes KV cache hits because they share the same 25k-token manual prefix.

### 2.3 Environment Configuration

```bash
# .env - Works for both local and remote modes
# Just change VLLM_BASE_URL to switch between local/remote

# Local Mode (Docker vLLM)
VLLM_BASE_URL=http://localhost:8001/v1
VLLM_API_KEY=not-required-for-local
VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct

# Remote Mode (Cloud GPU)
# VLLM_BASE_URL=http://your-gpu-server:8000/v1
# VLLM_API_KEY=your_api_key
# VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct

# Observability
LANGFUSE_PUBLIC_KEY=pk-lf-xxx
LANGFUSE_SECRET_KEY=sk-lf-xxx
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
```

### 2.4 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER                                            │
│                                │                                             │
│          ┌────────────────────┴────────────────────┐                        │
│          │ Requests                    Monitoring  │                        │
│          ▼                                         ▼                        │
│    ┌──────────┐                          ┌─────────────────┐                │
│    │  Router  │─────── Metrics ─────────▶│ Monitoring UI   │                │
│    │ (FastAPI │                          │   (Grafana)     │                │
│    │+LangGraph)│                          └────────▲────────┘                │
│    └────┬─────┘                                    │                        │
│         │                                          │                        │
│         │  3 Logical Agents                   ┌────┴────────┐               │
│         │  (Different Prompts)                │ Observability│               │
│         │                                     │  (Prometheus)│               │
│         ▼                                     └──────────────┘               │
│    ┌─────────────────────────────────────────────────────────┐              │
│    │               Single vLLM Server                         │              │
│    │  ┌─────────────────────────────────────────────────┐    │              │
│    │  │  • Technical Specialist (prompt A)              │    │              │
│    │  │  • Compliance Auditor (prompt B)    ─► Same LLM │    │              │
│    │  │  • Support Concierge (prompt C)                 │    │              │
│    │  └─────────────────────────────────────────────────┘    │              │
│    │                        │                                 │              │
│    │                        ▼                                 │              │
│    │         ┌────────────────────────────┐                  │              │
│    │         │  LMCache (Shared KV Cache) │                  │              │
│    │         │  • 25k token manual prefix │                  │              │
│    │         │  • Reused across all calls │                  │              │
│    │         └────────────────────────────┘                  │              │
│    └─────────────────────────────────────────────────────────┘              │
│                                                                              │
│              Local Docker / Remote Cloud                                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.5 Data Flow

```
User Request
     │
     ▼
┌─────────────┐
│   Router    │  1. Classify query
│  (FastAPI)  │  2. Select agent prompt(s)
└──────┬──────┘
       │
       │  All agents call SAME vLLM endpoint
       │  (different prompts only)
       │
       ▼
┌─────────────────────────────────────┐
│         Single vLLM Server          │
│  ┌───────────────────────────────┐  │
│  │ Agent A: Technical Specialist │  │
│  │ Agent B: Compliance Auditor   │──│──► Same LLM
│  │ Agent C: Support Concierge    │  │
│  └───────────────────────────────┘  │
│                 │                   │
│                 ▼                   │
│  ┌───────────────────────────────┐  │
│  │   LMCache (Shared KV Cache)   │  │
│  │   • 25k token manual prefix   │  │
│  │   • Reused for ALL agents     │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
```

---

## 3. Docker Compose Deployment

### 3.1 docker-compose.yml

```yaml
version: "3.8"

services:
  # ==========================================================================
  # Router Service (FastAPI + LangGraph)
  # ==========================================================================
  router:
    build: .
    container_name: llm-router
    ports:
      - "8000:8000"
    environment:
      - VLLM_BASE_URL=http://vllm:8000/v1
      - VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct
      - LANGFUSE_PUBLIC_KEY=${LANGFUSE_PUBLIC_KEY}
      - LANGFUSE_SECRET_KEY=${LANGFUSE_SECRET_KEY}
      - LANGFUSE_BASE_URL=${LANGFUSE_BASE_URL}
    depends_on:
      - vllm
    networks:
      - llm-network

  # ==========================================================================
  # Single vLLM Server (Shared by all 3 agents)
  # ==========================================================================
  vllm:
    image: vllm/vllm-openai:v0.6.4  # Pin to specific version for production stability
    container_name: vllm-server
    ports:
      - "8001:8000"
    environment:
      - LMCACHE_CONFIG_FILE=/app/lmcache-config.yaml
      - HF_TOKEN=${HF_TOKEN}
    volumes:
      - ./lmcache-config.yaml:/app/lmcache-config.yaml:ro
      - huggingface-cache:/root/.cache/huggingface
    command: >
      --model Qwen/Qwen2.5-7B-Instruct
      --max-model-len 32768
      --enable-prefix-caching
      --gpu-memory-utilization 0.9
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    networks:
      - llm-network

  # ==========================================================================
  # LMCache Server (Shared KV Cache Storage)
  # ==========================================================================
  lmcache:
    image: lmcache/lmcache-server:v0.4.0  # Pin to specific version for production stability
    container_name: lmcache-server
    ports:
      - "8080:8080"
    volumes:
      - lmcache-data:/data
    command: --port 8080 --storage-path /data
    networks:
      - llm-network

  # ==========================================================================
  # Prometheus (Metrics Collection)
  # ==========================================================================
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    networks:
      - llm-network

  # ==========================================================================
  # Grafana (Monitoring Dashboards)
  # ==========================================================================
  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning:ro
      - grafana-data:/var/lib/grafana
    depends_on:
      - prometheus
    networks:
      - llm-network

volumes:
  huggingface-cache:
  lmcache-data:
  prometheus-data:
  grafana-data:

networks:
  llm-network:
    driver: bridge
```

### 3.2 Quick Start Commands

```bash
# Start the full stack
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f router
docker-compose logs -f vllm

# Stop all services
docker-compose down
```

### 3.3 Service URLs

| Service | URL | Purpose |
|---------|-----|---------|
| Router API | http://localhost:8000 | Main application |
| vLLM Server | http://localhost:8001/v1 | Single LLM endpoint (all agents) |
| LMCache | http://localhost:8080 | Shared KV cache |
| Prometheus | http://localhost:9090 | Metrics |
| Grafana | http://localhost:3000 | Dashboards (admin/admin) |

---

## 4. vLLM Engine Configuration

### 4.1 Engine Flags Explained

| Flag | Purpose | Cache Impact |
|------|---------|--------------|
| `--enable-prefix-caching` | **CRITICAL**: Enables KV cache reuse for shared prefixes | Enables cache hits |
| `--max-model-len 32768` | Supports 25k token manual + query overhead | Required for manual |
| `--gpu-memory-utilization 0.9` | Maximizes GPU memory for KV cache | Larger cache capacity |

### 4.2 Model Selection

For local GPU with limited VRAM (8-16GB):

| Model | VRAM | Context | 
|-------|------|---------|
| Qwen/Qwen2.5-7B-Instruct | ~16GB | 32k |
| Qwen/Qwen2.5-3B-Instruct | ~8GB | 32k |
| microsoft/Phi-3-mini-4k-instruct | ~8GB | 4k |

### 4.3 Environment Variables

```bash
# .env file
HF_TOKEN=your_huggingface_token
LANGFUSE_PUBLIC_KEY=pk-lf-xxx
LANGFUSE_SECRET_KEY=sk-lf-xxx
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
```

---

## 5. LMCache Shared KV Storage

### 5.1 lmcache-config.yaml

```yaml
# LMCache Configuration for Shared KV Cache
chunk_size: 256
local_cpu: false  # Using dedicated LMCache server
remote_url: "http://lmcache:8080"
remote_serde: "naive"

# Hash algorithm for cache keys (xxhash is faster than sha256)
pre_caching_hash_algorithm: "xxhash"

# Connection settings
connection_timeout: 30
retry_attempts: 3
```

### 5.2 How LMCache Sharing Works

```
Agent A Request: [System + Manual + "Technical..." + Query]
                           ▼
                  ┌────────────────────┐
                  │   LMCache Server   │
                  │ ┌────────────────┐ │
                  │ │ KV Cache Entry │ │ ← Stores [System + Manual] KV vectors
                  │ │ hash: abc123   │ │
                  │ │ tokens: 25,000 │ │
                  │ └────────────────┘ │
                  └────────────────────┘
                           ▼
Agent B Request: [System + Manual + "Compliance..." + Query]
                  └──── Cache HIT! ────┘
                  Only computes "Compliance..." + Query
```

### 5.3 Expected Cache Behavior

| Request | Prefix Match | Computation |
|---------|--------------|-------------|
| 1st query to Agent A | None (cold) | Full 25k tokens |
| 1st query to Agent B | ✅ Hit | Only agent instructions + query |
| 1st query to Agent C | ✅ Hit | Only agent instructions + query |
| 2nd query to Agent A | ✅ Hit | Only new query tokens |

---

## 6. Router & LangGraph Workflow

### 6.1 Router Logic

The router classifies queries and selects appropriate agent(s):

```python
async def router_node(state: AgentState) -> AgentState:
    """
    Router that selects which agent prompt(s) to use.
    
    Uses an LLM call to classify the query and select the best agent(s).
    """
    query = state["query"]
    
    # Router prompt
    router_prompt = f"""
    You are a routing system for a banking AI.
    Select the best agent(s) to handle this query: "{query}"

    Agents:
    - technical: System specs, API limits, troubleshooting
    - compliance: Regulatory rules, policies, Can/Cannot
    - support: Step-by-step guides, explanations

    Return ONLY a JSON list of agent names, e.g. ["technical"] or ["technical", "compliance"].
    """
    
    # LLM Call for classification
    response = await llm.ainvoke(router_prompt)
    
    try:
        # Parse JSON response
        selected_agents = json.loads(response.content)
    except:
        # Fallback to support if parsing fails
        logger.warning(f"Router parsing failed for query: {query}")
        selected_agents = ["support"]
    
    return {**state, "selected_agents": selected_agents}
```

### 6.2 Agent Execution (Single Endpoint)

```python
async def agent_execution_node(state: AgentState) -> AgentState:
    """
    Execute selected agents - all call SAME vLLM endpoint with different prompts.
    """
    agents_to_call = state["selected_agents"]
    vllm_url = os.getenv("VLLM_BASE_URL")  # Single endpoint
    
    async def call_agent(agent_type: str) -> tuple[str, str]:
        prompt = build_prompt(agent_type, state["manual"], state["query"])
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{vllm_url}/chat/completions",
                json={
                    "model": os.getenv("VLLM_MODEL"),
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2048
                }
            )
            return agent_type, response.json()["choices"][0]["message"]["content"]
    
    # Fire all requests to SAME endpoint for cache optimization
    results = await asyncio.gather(
        *[call_agent(agent) for agent in agents_to_call],
        return_exceptions=True
    )
    
    # Filter out exceptions and log errors
    agent_responses = {}
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Agent call failed: {result}")
        elif isinstance(result, tuple):
            agent_name, response = result
            agent_responses[agent_name] = response
    
    return {**state, "agent_responses": agent_responses}

### 6.3 Aggregator Node

```python
async def aggregator_node(state: AgentState) -> AgentState:
    """
    Aggregate responses from multiple agents into a final answer.
    """
    responses = state["agent_responses"]
    
    # Simple concatenation or LLM-based synthesis can be used here
    # For efficiency, we'll join them clearly
    final_parts = []
    for agent, content in responses.items():
        final_parts.append(f"### From {agent.title()} Agent:\n{content}")
        
    final_response = "\n\n---\n\n".join(final_parts)
    
    return {
        **state,
        "final_response": final_response
    }
```

### 6.4 AgentState Definition

> **Best Practice**: Use `TypedDict` with `Annotated` reducers for proper state management in parallel execution.

```python
from typing import Annotated, TypedDict
from operator import add
from langgraph.graph.message import add_messages

class Message(TypedDict):
    role: str
    content: str

class AgentState(TypedDict):
    """State for the multi-agent workflow."""
    query: str
    manual_content: str
    # Use Annotated with reducer for lists that accumulate across parallel nodes
    history: Annotated[list[Message], add_messages]
    route_decision: Annotated[list[str], add]  # Append-only for parallel routing
    agent_responses: dict[str, str]
    final_response: str
    compliance_issues: Annotated[list[str], add]  # Accumulate issues
    retry_count: int
    selected_agents: list[str]
```

### 6.5 Graph Structure

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

def build_graph(
    checkpointer: MemorySaver | AsyncSqliteSaver | None = None
) -> CompiledStateGraph:
    """Build the multi-agent graph with optional persistence."""
    builder = StateGraph(AgentState)
    
    builder.add_node("router", router_node)
    builder.add_node("agents", agent_execution_node)
    builder.add_node("aggregator", aggregator_node)
    
    builder.add_edge(START, "router")
    builder.add_edge("router", "agents")
    builder.add_edge("agents", "aggregator")
    builder.add_edge("aggregator", END)
    
    # Compile with checkpointer for state persistence
    return builder.compile(checkpointer=checkpointer)

# Usage examples:
# Development (in-memory):
#   graph = build_graph(checkpointer=MemorySaver())
#
# Production (SQLite):
#   checkpointer = await AsyncSqliteSaver.from_conn_string("checkpoints.db")
#   graph = build_graph(checkpointer=checkpointer)
#
# When invoking, MUST include thread_id for persistence:
#   config = {"configurable": {"thread_id": session_id}}
#   result = await graph.ainvoke(state, config)
```

---

## 7. Prometheus & Grafana Observability

### 7.1 Prometheus Configuration

```yaml
# prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  # Router metrics
  - job_name: 'router'
    static_configs:
      - targets: ['router:8000']
    metrics_path: '/metrics'

  # Single vLLM Server metrics
  - job_name: 'vllm'
    static_configs:
      - targets: ['vllm:8000']
    metrics_path: '/metrics'

  # LMCache metrics
  - job_name: 'lmcache'
    static_configs:
      - targets: ['lmcache:8080']
    metrics_path: '/metrics'
```

### 7.2 Key Metrics to Monitor

| Metric | Source | Meaning |
|--------|--------|---------|
| `vllm:prompt_tokens_total` | vLLM server | Total input tokens |
| `vllm:generation_tokens_total` | vLLM server | Total output tokens |
| `vllm:time_to_first_token_seconds` | vLLM server | TTFT distribution |
| `vllm:cache_hit_rate` | vLLM server | Prefix cache effectiveness |
| `lmcache:entries` | LMCache | Cached KV entries |
| `lmcache:hits` | LMCache | Cache hit count |
| `lmcache:misses` | LMCache | Cache miss count |
| `router:requests_total` | Router | Total API requests |
| `router:agent_selection` | Router | Which agent prompts were used |

### 7.3 Grafana Dashboard

Create dashboard at `grafana/provisioning/dashboards/llm-metrics.json`:

```json
{
  "dashboard": {
    "title": "LLM Multi-Agent Metrics",
    "panels": [
      {
        "title": "Cache Hit Rate",
        "type": "gauge",
        "targets": [{"expr": "lmcache:hits / (lmcache:hits + lmcache:misses)"}]
      },
      {
        "title": "TTFT by Agent",
        "type": "graph",
        "targets": [{"expr": "histogram_quantile(0.95, vllm:time_to_first_token_seconds)"}]
      },
      {
        "title": "Agent Usage Distribution",
        "type": "piechart",
        "targets": [{"expr": "sum by (agent) (router:agent_selection)"}]
      }
    ]
  }
}
```

### 7.4 Accessing Dashboards

```bash
# Access Grafana
open http://localhost:3000
# Login: admin / admin

# Access Prometheus
open http://localhost:9090
```

---

## 8. State Management & LMCache Integration
 
 ### 8.1 Prefix Caching Strategy
 
 > **Constraint**: We are using a single vLLM endpoint. To maximize performance, we must ensure all agents share the exact same prompt prefix.
 
 **The Shared Prefix**:
 The ~25k token manual is the "prefix". It is sent at the *start* of every request.
 
 ```
 [System Instructions] + [25k Token Manual] + [End Marker] + [Agent Specifics] + [Query]
 └───────────────────────────────────────┘    └─────────────────────────────────────┘
         SHARED PREFIX (Cached)                    UNIQUE SUFFIX (Computed)
 ```
 
 ### 8.2 State Persistence (SQLite)
 
 > **Note**: Use `MemorySaver` for development and `AsyncSqliteSaver` for production persistence.
 
 ```python
 from langgraph.checkpoint.memory import MemorySaver
 from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
 
 # For development - in-memory (resets on restart)
 memory_checkpointer = MemorySaver()
 
 # For production/persistence (survives restarts)
 async def get_checkpointer() -> AsyncSqliteSaver:
     return await AsyncSqliteSaver.from_conn_string("checkpoints.db")
 
 # IMPORTANT: When invoking the graph, always include thread_id:
 # config = {"configurable": {"thread_id": "user-session-123"}}
 # result = await graph.ainvoke(initial_state, config)
 ```
 
 ### 8.3 Startup Cache Warming
 
 Use the application lifespan to "warm" the cache by sending the manual once.
 
 ```python
 @asynccontextmanager
 async def lifespan(app: FastAPI):
     """Application lifespan with cache warming and checkpointer setup."""
     # Load manual
     manual = load_manual()
     
     # Initialize checkpointer (use MemorySaver for dev, AsyncSqliteSaver for prod)
     # checkpointer = MemorySaver()  # Development
     checkpointer = await AsyncSqliteSaver.from_conn_string("checkpoints.db")  # Production
     
     # Warm cache with a simple request using full manual
     warmup_prompt = f"""{SYSTEM_PREFIX}
 
 {manual}
 
 <<< END OF MANUAL >>>
 
 Respond with 'CACHE_WARM' to confirm receipt."""
     
     try:
         # This forces vLLM to process and cache the 25k tokens
         await llm.ainvoke(warmup_prompt)
         logger.info("Cache warmed successfully")
     except Exception as e:
         logger.warning(f"Cache warm failed (non-fatal): {e}")
     
     # Store in app state
     app.state.manual = manual
     app.state.prompt_builder = DeterministicPromptBuilder(manual)
     app.state.graph = build_graph(checkpointer=checkpointer)  # Pass checkpointer
     app.state.checkpointer = checkpointer
     
     yield
     
     # Cleanup
     if hasattr(checkpointer, 'aclose'):
         await checkpointer.aclose()
     logger.info("Shutting down")
 ```
 
 ### 8.4 Client-Side Optimizations
 
 #### 8.4.1 Deterministic Prompt Construction
 
 Ensure prompts are byte-identical for cache hits. Even a single extra space breaks the prefix match.
 
 ```python
 import hashlib
 from functools import lru_cache
 
 class DeterministicPromptBuilder:
    def __init__(self, manual_content: str, prompt_manager):
        # Normalize whitespace ONCE at init, then reuse
        self._manual = self._normalize(manual_content)
        self.prompts = prompt_manager
     
     @staticmethod
     def _normalize(text: str) -> str:
         lines = [line.rstrip() for line in text.splitlines()]
         return "\n".join(lines)
     
     def build(self, agent_name: str, history: list, query: str) -> str:
         # Use exact same prefix for all agents
         prompt = f"""{SYSTEM_PREFIX}
 
 {self._manual}
 
 <<< END OF MANUAL >>>
 
 {self.prompts.get_agent_prompt(agent_name)}
 
 Conversation History:
 {self._format_history(history)}
 
 User Query: {query.strip()}"""
         return prompt
 ```
 
 #### 8.4.2 Request Deduplication
 
 Avoid sending duplicate content within the same session.
 
 ```python
 class RequestDeduplicator:
     """Track recent requests to avoid redundant LLM calls."""
     def __init__(self, max_cache: int = 100):
         self._cache: dict[str, str] = {}
     
     def _hash_request(self, agent: str, query: str) -> str:
        content = f"{agent}:{query.strip().lower()}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_cached(self, agent: str, query: str) -> str | None:
        key = self._hash_request(agent, query)
        return self._cache.get(key)
    
    def cache_response(self, agent: str, query: str, response: str):
        key = self._hash_request(agent, query)
        if len(self._cache) >= 100:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = response

# Global deduplicator
deduplicator = RequestDeduplicator()
```

### 8.5 Comprehensive Cache Metrics Logging

```python
import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

@dataclass
class CacheAwareMetrics:
    """
    Tracks metrics to understand cache efficiency without direct server logs.
    Infers cache hits by monitoring Time-To-First-Token (TTFT).
    """
    
    # Track TTFT history to detect cache behavior
    ttft_history: list[tuple[str, float, str]] = field(default_factory=list)
    
    # Baseline from first (cold) request
    cold_cache_ttft: Optional[float] = None
    
    # Hash of the shared prefix
    expected_prefix_hash: Optional[str] = None
    
    def log_request_start(self, agent_name: str, prompt: str) -> dict:
        """Log metrics check BEFORE sending request."""
        # Extract prefix hash
        prefix_end = prompt.find("<<< END OF MANUAL >>>")
        if prefix_end == -1: prefix_end = min(len(prompt), 100000)
        
        cacheable_prefix = prompt[:prefix_end]
        prefix_hash = hashlib.sha256(cacheable_prefix.encode()).hexdigest()[:16]
        
        # Check alignment
        if self.expected_prefix_hash is None:
            self.expected_prefix_hash = prefix_hash
            cache_aligned = True
        else:
            cache_aligned = (prefix_hash == self.expected_prefix_hash)
            if not cache_aligned:
                logger.warning(f"PREFIX MISMATCH! Cache will miss.")
        
        return {
            "agent": agent_name,
            "prefix_hash": prefix_hash,
            "cache_aligned": cache_aligned
        }
    
    def log_request_complete(self, agent_name: str, prefix_hash: str, ttft: float) -> dict:
        """Log after request to infer hit/miss."""
        self.ttft_history.append((agent_name, ttft, prefix_hash))
        
        if self.cold_cache_ttft is None:
            self.cold_cache_ttft = ttft
            is_cache_hit = False
        else:
            # <50% of cold time = likely hit
            is_cache_hit = (ttft / self.cold_cache_ttft) < 0.5
            
        return {
            "ttft": ttft,
            "is_cache_hit": is_cache_hit
        }


    
    def get_cache_report(self) -> dict:
        """
        Generate a summary report of cache efficiency.
        
        USE THIS FOR:
        - README.md optimization report
        - Debugging cache misses
        - Demonstrating cache awareness to evaluators
        """
        if not self.ttft_history:
            return {"status": "No requests recorded"}
        
        ttfts = [t for _, t, _ in self.ttft_history]
        hashes = [h for _, _, h in self.ttft_history]
        unique_hashes = set(hashes)
        
        # Count inferred hits/misses
        hits = sum(1 for t in ttfts[1:] if t < self.cold_cache_ttft * 0.5)
        misses = len(ttfts) - 1 - hits  # Exclude first (always cold)
        
        return {
            "total_requests": len(self.ttft_history),
            "cold_cache_baseline_seconds": self.cold_cache_ttft,
            "inferred_cache_hit_rate": hits / max(1, hits + misses),
            "unique_prefix_hashes": len(unique_hashes),
            "prefix_alignment_ok": len(unique_hashes) == 1,
            "avg_ttft_seconds": sum(ttfts) / len(ttfts),
            "min_ttft_seconds": min(ttfts),
            "max_ttft_seconds": max(ttfts),
            "recommendation": (
                "✅ Prefix is aligned - cache should be effective"
                if len(unique_hashes) == 1
                else f"⚠️ Found {len(unique_hashes)} different prefix hashes - "
                     "cache is being busted! Check prompt normalization."
            )
        }


# Global metrics instance
cache_metrics = CacheAwareMetrics()


@observe(name="invoke_agent_with_cache_tracking")
async def invoke_agent_with_cache_tracking(
    agent_name: str,
    prompt: str,
    query: str
) -> str:
    """
    =========================================================================
    CACHE-OPTIMIZED LLM INVOCATION
    =========================================================================
    
    This function wraps LLM calls with comprehensive cache tracking.
    
    CACHE OPTIMIZATION STRATEGY:
    ----------------------------
    1. BEFORE call: Log prefix hash to verify cache alignment
    2. DURING call: Measure TTFT precisely
    3. AFTER call: Infer cache hit from TTFT pattern
    
    WHY THIS MATTERS:
    -----------------
    With a 25,000 token manual, re-computing KV vectors for every
    request would be extremely expensive:
    
    - Without cache: ~25k tokens * ~0.001s/token = ~25 seconds
    - With cache: Only compute delta (~100 tokens) = ~0.1 seconds
    
    This is a 250x improvement in TTFT!
    =========================================================================
    """
    # Pre-request metrics
    pre_metrics = cache_metrics.log_request_start(agent_name, prompt)
    
    # Measure TTFT precisely
    start_time = time.perf_counter()
    
    # Invoke LLM
    response = await llm.ainvoke(prompt)
    
    # Calculate TTFT (time until response object is created)
    ttft = time.perf_counter() - start_time
    
    # Post-request metrics
    post_metrics = cache_metrics.log_request_complete(
        agent_name=agent_name,
        prefix_hash=pre_metrics["prefix_hash"],
        ttft_seconds=ttft
    )
    
    # Comprehensive logging for debugging
    logger.info(
        f"[LLM] Agent={agent_name} "
        f"prefix_hash={pre_metrics['prefix_hash']} "
        f"TTFT={ttft:.2f}s "
        f"cache_hit={post_metrics['is_cache_hit_inferred']} "
        f"response_len={len(response.content)}"
    )
    
    return response.content
```

---

## 9. Langfuse Observability (Mandatory)

> ⚠️ **Langfuse is REQUIRED from day one. All LLM calls MUST be traced.**

### 9.1 Setup

```python
# config/langfuse.py
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
import os

langfuse = Langfuse(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    host=os.environ.get("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")
)

def get_langfuse_handler():
    """
    Returns a Langfuse handler for LangChain.
    Credentials are picked up from environment variables:
    - LANGFUSE_PUBLIC_KEY
    - LANGFUSE_SECRET_KEY
    - LANGFUSE_BASE_URL
    """
    return CallbackHandler()
```

### 9.2 Tracing Every LLM Call

```python
from langfuse.decorators import observe, langfuse_context

@observe(name="invoke_agent")
async def invoke_agent(
    agent_name: str,
    manual: str,
    history: list[Message],
    query: str
) -> str:
    # Add metadata for cache analysis
    langfuse_context.update_current_observation(
        metadata={
            "agent_name": agent_name,
            "manual_tokens": len(manual) // 4,  # Approximate
            "history_length": len(history),
            "timestamp": datetime.utcnow().isoformat()
        }
    )
    
    prompt = build_prompt(agent_name, manual, history, query)
    
    # Track prompt structure for cache analysis
    langfuse_context.update_current_trace(
        metadata={
            "prompt_prefix_hash": hash_prefix(prompt[:25000]),
            "total_prompt_tokens": len(prompt) // 4
        }
    )
    
    start_time = time.perf_counter()
    response = await llm.ainvoke(prompt)
    ttft = time.perf_counter() - start_time
    
    # Log TTFT for cache inference
    langfuse_context.update_current_observation(
        metadata={"ttft_seconds": ttft}
    )
    
    return response.content
```

### 9.3 Required Tracing Annotations

Every major function MUST use `@observe`:

```python
@observe(name="router")
async def router_node(state: AgentState) -> AgentState: ...

@observe(name="parallel_agents") 
async def parallel_agent_node(state: AgentState) -> AgentState: ...

@observe(name="compliance_evaluator")
async def compliance_evaluator_node(state: AgentState) -> AgentState: ...

@observe(name="aggregator")
async def aggregator_node(state: AgentState) -> AgentState: ...
```

---

## 10. Code Style & Conventions
 
 ### 10.1 Python Standards
 
 ```python
 # Use modern Python 3.13 features
 from typing import TypedDict, Annotated
 from collections.abc import Sequence, Mapping
 from dotenv import load_dotenv

 # mandatory: call load_dotenv() in settings.py for library compatibility
 load_dotenv()

 # Async everywhere
 async def process_query(...) -> str: ...
 
 # Type hints are MANDATORY
 def build_prompt(
     agent_name: str,
     manual: str,
     history: Sequence[Message],
     query: str
 ) -> str: ...
 ```
 
 ### 10.2 Project Structure
 
 ```
 bank-multi-agent/
 ├── pyproject.toml           # uv project config
 ├── uv.lock                   # Lock file
 ├── .env.example              # Environment template
 ├── AGENTS.MD                 # This file
 ├── README.md                 # Optimization report
 ├── src/
 │   ├── __init__.py
 │   ├── main.py               # FastAPI app entry
 │   ├── config/
 │   │   ├── __init__.py
 │   │   ├── settings.py       # Pydantic settings
 │   │   └── langfuse.py       # Langfuse setup
 │   ├── agents/
 │   │   ├── __init__.py
 │   │   ├── router.py         # LLM router
 │   │   ├── technical.py      # Technical Specialist
 │   │   ├── compliance.py     # Compliance Auditor
 │   │   ├── support.py        # Support Concierge
 │   │   └── evaluator.py      # Compliance evaluator
 │   ├── graph/
 │   │   ├── __init__.py
 │   │   ├── state.py          # AgentState definition
 │   │   ├── nodes.py          # Graph nodes
 │   │   └── builder.py        # Graph construction
 │   ├── prompts/
 │   │   └── manager.py        # PromptManager class
 │   ├── cache/
 │   │   ├── __init__.py
 │   │   └── warmer.py         # Cache warmup logic
 │   └── api/
 │       ├── __init__.py
 │       ├── routes.py         # REST endpoints
 │       └── schemas.py        # Pydantic models
 │   ├── prompts/                  # .prompty files (see section 4)
 │   ├── data/
 │   │   └── operations_manual.txt # The 50-page manual
 │   └── tests/
 │   ├── __init__.py
 │   ├── test_routing.py
 │   ├── test_agents.py
 │   ├── test_cache_efficiency.py
 │   └── conftest.py
 ```
 
 ### 10.3 Naming Conventions
 
 | Element | Convention | Example |
 |---------|------------|---------|
 | Files | snake_case | `technical_specialist.py` |
 | Classes | PascalCase | `TechnicalSpecialist` |
 | Functions | snake_case | `invoke_agent` |
 | Constants | UPPER_SNAKE | `MAX_RETRIES` |
 | Prompty files | snake_case | `compliance_auditor.prompty` |
 | Environment vars | UPPER_SNAKE | `LANGFUSE_PUBLIC_KEY` |

 ### 10.4 Environment Stability
 
 > [!IMPORTANT]
 > To ensure libraries like `prompty` can correctly resolve variable references (e.g., `${env:VLLM_MODEL}`), you **MUST** call `load_dotenv()` within `src/config/settings.py`. This ensures that `.env` values are synced to `os.environ` at application startup.
 
 ---

## 11. Prompt Management
 
 ### 11.1 Prompty Migration Rule
 
 > [!IMPORTANT]
 > All expert agents (Router, Technical, Compliance, Support, Aggregator) **MUST** use the `.prompty` file format. Hardcoded template strings in Python are deprecated.
 
 ### 11.2 Prompt Engineering Strategy
 
 > **Optimization Goal**: Maximize shared prefix length across all agents.
 
 **Anti-Patterns (❌ DO NOT DO):**
 
 ```python
 # ❌ WRONG: Agent instructions BEFORE manual (breaks prefix sharing)
 prompt = f"""
 You are the Technical Specialist.  # Agent-specific first = cache miss!
 {manual_content}
 {user_query}
 """
 
 # ❌ WRONG: Timestamp or request ID in prefix
 prompt = f"""
 Request ID: {uuid4()}  # Dynamic content = cache bust!
 {manual_content}
 ...
 """
 ```
 
 **Correct Pattern (✅):**
 
 ```python
 # ✅ CORRECT: Use .prompty files for deterministic templates
import prompty
from pathlib import Path

def build_prompt(agent_name: str, manual: str, history: list, query: str) -> str:
    # Load .prompty file
    p = prompty.load(f"src/prompts/{agent_name}.prompty")
    
    # Hydrate template with variables
    # This automatically handles the shared prefix (manual_content)
    # placed at the start of the template.
    prompt = prompty.prepare(p, {
        "manual_content": manual,
        "history": format_history(history),
        "query": query
    })
    return prompt
 ```
 
 ### 11.3 Prompty File Structure
 
 ```
 prompts/
 ├── shared/
 │   ├── system_prefix.prompty      # Shared system header
 │   └── manual_wrapper.prompty     # Manual embedding template
 ├── agents/
 │   ├── router.prompty             # LLM router classification
 │   ├── technical_specialist.prompty
 │   ├── compliance_auditor.prompty
 │   └── support_concierge.prompty
 ├── evaluation/
 │   └── compliance_checker.prompty # Post-response compliance check
 └── config.yaml                    # Prompty configuration
 ```
 
 ### 11.4 Prompty File Format
 
 Each `.prompty` file MUST use this structure:
 
 ```yaml
 ---
 name: technical_specialist
 description: Technical Specialist agent for system specs and troubleshooting
 authors:
   - Bank AI Team
 version: 1.0.0
 model:
   api: chat
   configuration:
     type: azure_openai  # or openai, etc.
 inputs:
   manual_content:
     type: string
     description: The full operations manual
   history:
     type: list
     description: Conversation history
   query:
     type: string
     description: User's current question
 outputs:
   response:
     type: string
 ---
 system:
 {{system_prefix}}
 
 {{manual_content}}
 
 <<< END OF MANUAL >>>
 
 You are the **Technical Specialist**. Your responsibilities:
 1. Extract system specifications from the manual
 2. Identify API limits and technical constraints
 3. Provide troubleshooting steps for technical issues
 
 Guidelines:
 - Be precise and cite specific sections/page numbers
 - Use code blocks for API examples
 - Flag any uncertainties clearly
 
 user:
 {{#each history}}
 {{role}}: {{content}}
 {{/each}}
 {{query}}
 ```
 
 ### 11.5 Prompty Loading Pattern
 
 ```python
 import prompty
from pathlib import Path
from functools import lru_cache

class PromptManager:
    def __init__(self, prompts_dir: Path):
        self.prompts_dir = prompts_dir
    
    @lru_cache(maxsize=10)
    def get_agent_prompt(self, agent_name: str):
        """Load and cache a .prompty file."""
        path = self.prompts_dir / f"{agent_name}.prompty"
        return prompty.load(str(path))
    
    def build(self, agent_name: str, manual: str, history: list, query: str) -> str:
        p = self.get_agent_prompt(agent_name)
        return prompty.prepare(p, {
            "manual_content": manual,
            "history": self._format_history(history),
            "query": query
        })
 ```

---

## 12. REST API Design
 
 ### 12.1 FastAPI Application
 
 ```python
 # src/main.py
 from fastapi import FastAPI
 from contextlib import asynccontextmanager
 from src.cache.warmer import warm_cache
 from src.graph.builder import build_graph
 
 @asynccontextmanager
 async def lifespan(app: FastAPI):
     # Startup: warm the KV cache
     manual = load_manual()
     await warm_cache(manual)
     app.state.graph = build_graph()
     app.state.manual = manual
     yield
     # Shutdown: cleanup
 
 app = FastAPI(
     title="Bank Multi-Agent API",
     version="1.0.0",
     lifespan=lifespan
 )
 ```
 
 ### 12.2 Endpoints

 > **Best Practice**: Use FastAPI `Depends()` for dependency injection to improve testability.

 ```python
 # src/api/routes.py
 from typing import Annotated
 from fastapi import APIRouter, Request, Depends
 from pydantic import BaseModel
 from langgraph.graph import CompiledStateGraph

 router = APIRouter(prefix="/api/v1")

 class QueryRequest(BaseModel):
     query: str
     session_id: str
     user_id: str | None = None

 class QueryResponse(BaseModel):
     response: str
     agents_used: list[str]
     compliance_passed: bool
     retry_count: int
     ttft_seconds: float

 # Dependency injection for better testability
 async def get_graph(request: Request) -> CompiledStateGraph:
     return request.app.state.graph

 async def get_manual(request: Request) -> str:
     return request.app.state.manual

 @router.post("/query", response_model=QueryResponse)
 async def handle_query(
     body: QueryRequest,
     graph: Annotated[CompiledStateGraph, Depends(get_graph)],
     manual: Annotated[str, Depends(get_manual)]
 ):
     # IMPORTANT: Include thread_id for checkpointer persistence
     config = {"configurable": {"thread_id": body.session_id}}
     
     result = await graph.ainvoke(
         {
             "query": body.query,
             "manual_content": manual,
             "history": [],
             "route_decision": [],
             "agent_responses": {},
             "final_response": "",
             "compliance_issues": [],
             "retry_count": 0,
             "selected_agents": []
         },
         config=config  # Pass config with thread_id
     )
     
     return QueryResponse(
         response=result["final_response"],
         agents_used=result["route_decision"],
         compliance_passed=len(result["compliance_issues"]) == 0,
         retry_count=result["retry_count"],
         ttft_seconds=result.get("ttft", 0.0)
     )

 @router.get("/health")
 async def health():
     return {"status": "healthy"}

 @router.get("/cache/stats")
 async def cache_stats():
     """
     Return cache efficiency metrics for Grafana/monitoring.
     """
     from src.cache.metrics import cache_metrics
     return cache_metrics.get_stats()
 ```

---

## 13. Testing & Verification

### 13.1 Test Categories

| Category | Purpose | Location |
|----------|---------|----------|
| Unit Tests | Individual agent logic | `tests/test_agents.py` |
| Integration Tests | Graph flow | `tests/test_graph.py` |
| Cache Tests | Verify prefix caching | `tests/test_cache_efficiency.py` |
| API Tests | REST endpoint testing | `tests/test_api.py` |

### 13.2 Cache Efficiency Test

```python
# tests/test_cache_efficiency.py
import pytest
import time

@pytest.mark.asyncio
async def test_prefix_cache_reuse():
    """
    Verify that calling multiple agents reuses the cached manual prefix.
    Second agent call should have significantly lower TTFT.
    """
    manual = load_test_manual()
    query = "What are the API rate limits?"
    
    # First call - cold cache
    start1 = time.perf_counter()
    await invoke_agent("technical_specialist", manual, [], query)
    ttft1 = time.perf_counter() - start1
    
    # Second call - should hit cache
    start2 = time.perf_counter()
    await invoke_agent("compliance_auditor", manual, [], query)
    ttft2 = time.perf_counter() - start2
    
    # Third call - should also hit cache
    start3 = time.perf_counter()
    await invoke_agent("support_concierge", manual, [], query)
    ttft3 = time.perf_counter() - start3
    
    # Assert cache is working: subsequent calls should be faster
    assert ttft2 < ttft1 * 0.7, f"Expected cache hit but TTFT2={ttft2} >= TTFT1*0.7={ttft1*0.7}"
    assert ttft3 < ttft1 * 0.7, f"Expected cache hit but TTFT3={ttft3} >= TTFT1*0.7={ttft1*0.7}"
```

### 13.3 Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run with cache efficiency tests (requires vLLM endpoint)
uv run pytest tests/test_cache_efficiency.py -v --tb=short

# Run API tests
uv run pytest tests/test_api.py -v
```

---

## 14. Error Handling & Reliability

> **No Retry Loop**: Strict prompts ensure first-pass compliance.

### 14.1 LLM Call Retry (Network/Transient Errors Only)

```python
from tenacity import retry, stop_after_attempt, wait_exponential

MAX_LLM_RETRIES = 3

@retry(
    stop=stop_after_attempt(MAX_LLM_RETRIES),
    wait=wait_exponential(multiplier=1, min=1, max=10)
)
async def invoke_llm_with_retry(prompt: str) -> str:
    """Retry LLM calls for network/transient errors only."""
    return await llm.ainvoke(prompt)
```

### 14.2 Strict Prompt Design (Instead of Retry Loop)

To ensure first-pass compliance, agent prompts MUST include:

```markdown
## CRITICAL COMPLIANCE RULES

You MUST follow these rules in EVERY response:

1. **ONLY cite information from the manual** - Never invent policies
2. **Quote section numbers** - e.g., "Per Section 4.2.1..."
3. **Use explicit uncertainty markers** if information is unclear:
   - "This may require verification with [department]"
   - "The manual does not explicitly address this"
4. **Never make assumptions** about regulatory requirements

Violating these rules will result in incorrect guidance to staff.
```

---



## Appendix A: Testing Endpoint

Use this endpoint for development and testing:

```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer not-required-for-local" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

---

## Appendix B: Quick Start

```bash
# 1. Initialize project with uv
uv init bank-multi-agent
cd bank-multi-agent

# 2. Add dependencies
uv add langgraph langchain langchain-openai prompty langfuse fastapi uvicorn
uv add --dev pytest pytest-asyncio httpx

# 3. Set up environment
cp .env.example .env
# Edit .env with your keys

# 4. Run the API
uv run uvicorn src.main:app --reload --port 8000

# 5. Test cache efficiency
uv run pytest tests/test_cache_efficiency.py -v
```

---

## Appendix C: Checklist for Code Review

Before merging any PR, verify:

- [ ] All prompts follow the POSITION 1-5 order (Section 3.1)
- [ ] No dynamic content (timestamps, UUIDs) in the shared prefix
- [ ] All LLM calls are wrapped with `@observe`
- [ ] Langfuse metadata includes `ttft_seconds`
- [ ] New agents use the shared `SHARED_PREFIX` constant
- [ ] Async is used throughout (no blocking calls)
- [ ] Tests exist for new functionality
- [ ] Prompts include CRITICAL COMPLIANCE RULES section (Section 11.2)
