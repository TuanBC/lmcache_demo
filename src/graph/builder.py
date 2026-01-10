"""
=============================================================================
LangGraph Builder
=============================================================================

Constructs the multi-agent workflow graph.

GRAPH STRUCTURE:
----------------
START -> router -> parallel_agents -> aggregator -> END

CHECKPOINTER SUPPORT (AGENTS.MD Compliant):
--------------------------------------------
The build_graph function accepts an optional checkpointer parameter
for state persistence. Use MemorySaver for development and
AsyncSqliteSaver for production.
=============================================================================
"""

import logging
from typing import Union

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.graph.nodes import aggregator_node, parallel_agents_node, router_node
from src.graph.state import AgentState

logger = logging.getLogger(__name__)


def build_graph(
    checkpointer: Union[MemorySaver, BaseCheckpointSaver, None] = None
) -> CompiledStateGraph:
    """
    Build and compile the multi-agent workflow graph.
    
    Args:
        checkpointer: Optional checkpointer for state persistence.
                      Use MemorySaver() for development,
                      AsyncSqliteSaver for production.
    
    FLOW:
    -----
    1. Router determines which agents to call
    2. Selected agents execute in parallel
    3. Aggregator combines responses
    
    CACHE OPTIMIZATION:
    -------------------
    All nodes use the same shared prefix, so the KV cache is
    warmed by the router and reused by all subsequent agents.
    
    USAGE EXAMPLES (from AGENTS.MD):
    ---------------------------------
    # Development (in-memory, resets on restart):
    graph = build_graph(checkpointer=MemorySaver())
    
    # Production (SQLite persistence):
    checkpointer = await AsyncSqliteSaver.from_conn_string("checkpoints.db")
    graph = build_graph(checkpointer=checkpointer)
    
    # When invoking, MUST include thread_id for persistence:
    config = {"configurable": {"thread_id": session_id}}
    result = await graph.ainvoke(state, config)
    """
    logger.info("[GRAPH] Building multi-agent workflow graph")
    
    builder = StateGraph(AgentState)
    
    # Add nodes - Single pass, no retry loop
    builder.add_node("router", router_node)
    builder.add_node("parallel_agents", parallel_agents_node)
    builder.add_node("aggregator", aggregator_node)
    
    # Add edges - Simple linear flow
    builder.add_edge(START, "router")
    builder.add_edge("router", "parallel_agents")
    builder.add_edge("parallel_agents", "aggregator")
    builder.add_edge("aggregator", END)
    
    # Compile with optional checkpointer for state persistence
    graph = builder.compile(checkpointer=checkpointer)
    
    logger.info(
        f"[GRAPH] Graph compiled successfully "
        f"(checkpointer={'enabled' if checkpointer else 'disabled'})"
    )
    
    return graph
