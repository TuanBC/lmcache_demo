"""
=============================================================================
LangGraph Node Implementations
=============================================================================

This module contains the node functions for the multi-agent workflow:
1. Router Node - LLM-based classification
2. Parallel Agents Node - Fan-out execution
3. Aggregator Node - Combine responses

CACHE OPTIMIZATION:
-------------------
All nodes use the same shared prefix (system + manual) to maximize
KV cache hits. The DeterministicPromptBuilder ensures byte-identical
prefixes across all agent calls.

AGENTS.MD COMPLIANT:
--------------------
- Uses compliance_issues list instead of boolean needs_human_review
- Proper exception handling for asyncio.gather
- Returns ONLY the updated fields (avoids redundant state accumulation)
- Tenacity retry for network errors
- Langfuse tracing with callbacks
=============================================================================
"""

import asyncio
import json
import logging
import time

import httpx
from langchain_openai import ChatOpenAI
from langfuse import get_client
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.cache.metrics import cache_metrics
from src.config.langfuse import get_langfuse_handler, observe, propagate_attributes
from src.config.settings import get_settings
from src.graph.state import AgentState
from src.prompts.manager import DeterministicPromptBuilder

logger = logging.getLogger(__name__)


def get_llm() -> ChatOpenAI:
    """Get the LLM client configured for vLLM."""
    settings = get_settings()
    return ChatOpenAI(
        base_url=settings.vllm_base_url,
        api_key=settings.vllm_api_key,
        model=settings.vllm_model,
        temperature=0.7,
        max_tokens=2048,
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.ConnectError, httpx.TimeoutException)),
)
async def invoke_llm_with_retry(
    llm: ChatOpenAI, prompt: str, metadata: dict | None = None
) -> tuple[str, float]:
    """
    Retry LLM calls for network/transient errors only.

    Returns (response_content, ttft_seconds).
    """
    langfuse_handler = get_langfuse_handler()

    start_time = time.perf_counter()

    response = await llm.ainvoke(prompt, config={"callbacks": [langfuse_handler]})

    ttft = time.perf_counter() - start_time

    # Update Langfuse observation with TTFT metadata (AGENTS.MD section 9.2)
    # Langfuse v3 uses get_client().update_current_span()
    try:
        langfuse_client = get_client()
        langfuse_client.update_current_span(metadata={"ttft_seconds": ttft, **(metadata or {})})
    except Exception as e:
        # Don't fail the LLM call if Langfuse update fails
        logger.debug(f"Langfuse span update skipped: {e}")

    return response.content, ttft


@observe(name="router_node")
async def router_node(state: AgentState) -> dict:
    """
    LLM-based router that determines which agents to invoke.

    CACHE OPTIMIZATION:
    -------------------
    The router uses the same shared prefix as other agents, so its
    KV cache can warm the prefix for subsequent agent calls.

    NOTE: Returns ONLY the updates to avoid redundant accumulation in reducers.
    """
    logger.info(f"[ROUTER] Processing query: {state['query'][:100]}...")

    with propagate_attributes(
        session_id=state.get("session_id", "unknown"), metadata={"node": "router"}
    ):
        prompt_builder = DeterministicPromptBuilder(state["manual_content"])
        prompt = prompt_builder.build("router", state.get("history", []), state["query"])

        # Log cache metrics
        pre_metrics = cache_metrics.log_request_start("router", prompt)

        # Invoke LLM with retry
        llm = get_llm()

        try:
            response, ttft = await invoke_llm_with_retry(llm, prompt, metadata={"node": "router"})

            # Log completion metrics
            cache_metrics.log_request_complete("router", pre_metrics["prefix_hash"], ttft)

            # Parse router response - handle both string and structured responses
            content = str(response)

            try:
                result = json.loads(content)
                agents = result.get("agents", ["technical_specialist"])
            except json.JSONDecodeError:
                logger.warning("[ROUTER] Failed to parse JSON, using default agent")
                agents = ["technical_specialist"]

            logger.info(f"[ROUTER] Selected agents: {agents}")

            return {
                "route_decision": agents,
                "selected_agents": agents,
                "ttft_seconds": ttft,
            }

        except Exception as e:
            logger.error(f"[ROUTER] Error: {e}")
            return {
                "route_decision": ["technical_specialist"],
                "selected_agents": ["technical_specialist"],
                "ttft_seconds": 0.0,
            }


@observe(name="parallel_agents_node")
async def parallel_agents_node(state: AgentState) -> dict:
    """
    Execute selected agents in parallel (fan-out).

    CACHE OPTIMIZATION:
    -------------------
    Strategy: Send requests in quick succession so the server
    can batch them and reuse the computed prefix.

    All agents share the same prefix (system + manual), so after
    the first agent computes the KV cache, subsequent agents
    can reuse it.

    EXCEPTION HANDLING (AGENTS.MD):
    --------------------------------
    Properly filters exceptions from asyncio.gather results
    instead of failing on dict() conversion.

    NOTE: Returns ONLY the updates to avoid redundant accumulation in reducers.
    """
    agents_to_call = state.get("selected_agents", state.get("route_decision", []))
    logger.info(f"[PARALLEL] Executing {len(agents_to_call)} agents: {agents_to_call}")

    session_id = state.get("session_id", "unknown")

    with propagate_attributes(
        session_id=session_id, metadata={"node": "parallel_agents", "agents": agents_to_call}
    ):
        prompt_builder = DeterministicPromptBuilder(state["manual_content"])
        llm = get_llm()

        # Pre-build all prompts (ensures they're ready simultaneously)
        prompts = {
            agent: prompt_builder.build(agent, state.get("history", []), state["query"])
            for agent in agents_to_call
        }

        async def call_agent(agent_name: str) -> tuple[str, str, float]:
            """Call a single agent and return (name, response, ttft)."""
            prompt = prompts[agent_name]

            # Log cache metrics
            pre_metrics = cache_metrics.log_request_start(agent_name, prompt)

            try:
                response, ttft = await invoke_llm_with_retry(
                    llm, prompt, metadata={"agent": agent_name}
                )

                # Log completion metrics
                cache_metrics.log_request_complete(agent_name, pre_metrics["prefix_hash"], ttft)

                logger.info(f"[AGENT] {agent_name} completed in {ttft:.2f}s")

                return agent_name, str(response), ttft

            except Exception as e:
                logger.error(f"[AGENT] {agent_name} failed: {e}")
                raise

        # Fire all requests at once (maximize batching opportunity)
        results = await asyncio.gather(
            *[call_agent(name) for name in agents_to_call],
            return_exceptions=True,
        )

        # Process results - AGENTS.MD compliant exception handling
        agent_responses: dict[str, str] = {}
        total_ttft = 0.0
        successful_count = 0

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"[AGENT] Call failed: {result}")
            elif isinstance(result, tuple):
                agent_name, content, ttft = result
                agent_responses[agent_name] = content
                total_ttft += ttft
                successful_count += 1

        return {
            "agent_responses": agent_responses,
            "ttft_seconds": total_ttft / successful_count if successful_count > 0 else 0.0,
        }


# Alias for AGENTS.MD compatibility
agent_execution_node = parallel_agents_node


@observe(name="aggregator_node")
async def aggregator_node(state: AgentState) -> dict:
    """
    Aggregate responses from all agents.

    COMPLIANCE TRACKING (AGENTS.MD):
    ---------------------------------
    Uses compliance_issues list instead of boolean needs_human_review.
    This allows accumulating specific issues found during processing.

    NOTE: Returns ONLY the updates to avoid redundant accumulation in reducers.
    """
    session_id = state.get("session_id", "unknown")

    with propagate_attributes(session_id=session_id, metadata={"node": "aggregator"}):
        responses = state.get("agent_responses", {})

        logger.info(f"[AGGREGATOR] Aggregating {len(responses)} responses")

        # Check for uncertainty markers that indicate compliance issues
        uncertainty_markers = [
            "I'm not certain",
            "This may require verification",
            "Please consult",
            "unclear from the manual",
            "This may require verification with",
            "The manual does not explicitly address",
        ]

        # Collect specific compliance issues
        compliance_issues: list[str] = []
        for agent_name, response in responses.items():
            for marker in uncertainty_markers:
                if marker.lower() in response.lower():
                    compliance_issues.append(f"{agent_name}: Contains uncertainty - '{marker}'")
                    break

        # Build aggregated response
        if len(responses) == 1:
            final_response = list(responses.values())[0]
        else:
            parts = []
            for agent_name, response in responses.items():
                display_name = agent_name.replace("_", " ").title()
                parts.append(f"## {display_name}\n\n{response}")

            final_response = "\n\n---\n\n".join(parts)

        if compliance_issues:
            final_response += "\n\n---\n\n⚠️ **Note**: This response has been flagged for review."
            final_response += "\n\nIssues found:\n" + "\n".join(
                f"- {issue}" for issue in compliance_issues
            )

        logger.info(f"[AGGREGATOR] Complete. compliance_issues={len(compliance_issues)}")

        return {
            "final_response": final_response,
            "compliance_issues": compliance_issues,
        }
