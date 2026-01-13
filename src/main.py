"""
=============================================================================
Bank Multi-Agent Expert System - Main Application
=============================================================================

FastAPI application entry point.

CACHE OPTIMIZATION:
-------------------
The application loads the operations manual ONCE at startup and
stores it in app.state. This ensures the manual content is byte-identical
across all requests, maximizing KV cache hits.

AGENTS.MD COMPLIANT:
--------------------
- Initializes MemorySaver checkpointer for state persistence
- Passes checkpointer to build_graph()
- Stores checkpointer in app.state for cleanup

Startup sequence:
1. Load operations manual
2. Initialize checkpointer
3. Warm cache with initial request
4. Build LangGraph workflow with checkpointer
5. Start REST API
=============================================================================
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.memory import MemorySaver

from src.api.routes import router as api_router
from src.config.settings import get_settings
from src.graph.builder import build_graph
from src.prompts.manager import DeterministicPromptBuilder, load_manual

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan with cache warming and checkpointer setup.

    CACHE OPTIMIZATION:
    -------------------
    We warm the cache at startup by sending a simple request
    with the full manual. This pre-populates the server's KV cache
    so subsequent requests are faster.

    CHECKPOINTER (AGENTS.MD):
    -------------------------
    Initializes MemorySaver for development. For production, replace
    with AsyncSqliteSaver for persistent state across restarts.
    """
    settings = get_settings()

    logger.info("[STARTUP] Initializing Bank Multi-Agent Expert System...")

    # 1. Load the operations manual
    manual_path = Path(settings.manual_path)
    manual = load_manual(manual_path)

    # Store in app state (shared across all requests)
    app.state.manual = manual

    # 2. Build the prompt builder
    prompt_builder = DeterministicPromptBuilder(manual)
    app.state.prompt_builder = prompt_builder

    logger.info(f"[STARTUP] Manual loaded: {len(manual)} chars, hash={prompt_builder.manual_hash}")

    # 3. Initialize checkpointer (AGENTS.MD compliant)
    # Use MemorySaver for development (resets on restart)
    # For production, use: checkpointer = await AsyncSqliteSaver.from_conn_string("checkpoints.db")
    checkpointer = MemorySaver()
    app.state.checkpointer = checkpointer

    logger.info("[STARTUP] Checkpointer initialized (MemorySaver)")

    logger.info("[STARTUP] Checkpointer initialized (MemorySaver)")

    # 4. Warmup removed to allow accurate Cold vs Warm TTFT testing
    # The application will now start "cold". The first user request will incur prefill latency.
    # To re-enable production warmup, uncomment the block below.

    # logger.info(f"[STARTUP] Warming cache with full prefix...")
    # ... (warmup logic commented out) ...

    # 5. Build the graph with checkpointer

    # 5. Build the graph with checkpointer
    graph = build_graph(checkpointer=checkpointer)
    app.state.graph = graph

    logger.info("[STARTUP] Application ready!")

    yield

    # Shutdown - cleanup checkpointer if needed
    logger.info("[SHUTDOWN] Shutting down...")

    # For AsyncSqliteSaver, you would call: await checkpointer.aclose()
    # MemorySaver doesn't require cleanup


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Bank Multi-Agent Expert System",
        description="High-Efficiency Multi-Agent system for bank operations manual queries",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routes
    app.include_router(api_router)

    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
