"""
=============================================================================
LangGraph State Definition
=============================================================================

Defines the state schema for the multi-agent workflow.

STATE DESIGN NOTES (AGENTS.MD Compliant):
-----------------------------------------
- Uses Annotated with reducers for proper parallel execution
- manual_content is loaded ONCE and passed through the entire workflow
- agent_responses stores outputs from parallel agent execution
- compliance_issues accumulates any issues found during processing
=============================================================================
"""

from operator import add
from typing import Annotated, TypedDict


class Message(TypedDict):
    """A single message in the conversation history."""

    role: str  # "user" or "assistant"
    content: str


class AgentState(TypedDict):
    """
    State schema for the multi-agent workflow.

    This state is passed through all nodes in the LangGraph.

    BEST PRACTICE (from AGENTS.MD):
    --------------------------------
    Uses Annotated with reducer functions (operator.add) to ensure
    parallel node executions APPEND to lists instead of OVERWRITING them.
    """

    # Input
    query: str  # User's current question
    manual_content: str  # The 25k-token operations manual (loaded once)

    # Use Annotated with reducer for lists that accumulate across parallel nodes
    history: Annotated[list[Message], add]  # Conversation history

    # Router output
    route_decision: Annotated[list[str], add]  # Append-only for parallel routing
    selected_agents: list[str]  # Which agents to call (set by router)

    # Agent outputs
    agent_responses: dict[str, str]  # agent_name -> response

    # Final output
    final_response: str  # Aggregated response from all agents
    compliance_issues: Annotated[list[str], add]  # Accumulate issues found
    retry_count: int  # Number of retry attempts (if any)

    # Session tracking
    session_id: str  # Session ID for tracing and history

    # Metrics
    ttft_seconds: float  # Time to first token (for cache analysis)
