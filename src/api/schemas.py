"""
=============================================================================
API Schemas
=============================================================================

Pydantic models for request/response validation.

AGENTS.MD COMPLIANT:
--------------------
- Uses compliance_passed (boolean derived from compliance_issues list)
- Includes retry_count field
=============================================================================
"""

from pydantic import BaseModel


class QueryRequest(BaseModel):
    """Request body for /api/v1/query endpoint."""
    
    query: str
    session_id: str
    user_id: str | None = None


class QueryResponse(BaseModel):
    """
    Response body for /api/v1/query endpoint.
    
    AGENTS.MD Compliant:
    - compliance_passed: True if no compliance issues found
    - retry_count: Number of retry attempts (always 0 with current design)
    """
    
    response: str
    agents_used: list[str]
    compliance_passed: bool  # True if compliance_issues list is empty
    retry_count: int  # Number of retry attempts
    ttft_seconds: float


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""
    
    status: str
    version: str = "0.1.0"


class CacheStatsResponse(BaseModel):
    """Response body for /cache/stats endpoint."""
    
    total_requests: int
    cold_cache_baseline_seconds: float | None
    inferred_cache_hit_rate: float
    unique_prefix_hashes: int
    prefix_alignment_ok: bool
    avg_ttft_seconds: float
    min_ttft_seconds: float
    max_ttft_seconds: float
    recommendation: str
    grade: str
    interpretation: dict
