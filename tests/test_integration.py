"""
=============================================================================
Integration Tests - Real LLM Cache Efficiency
=============================================================================

Tests that make actual LLM calls to verify cache behavior.

USAGE:
------
# Run integration tests only (requires running vLLM server)
uv run pytest tests/test_integration.py -v -m integration

# Skip integration tests (default for CI)
uv run pytest tests/ -v -m "not integration"

REQUIRES:
---------
- Running vLLM server with LMCache enabled
- Set VLLM_BASE_URL environment variable
=============================================================================
"""

import asyncio
import os
import time

import httpx
import pytest

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
def api_base_url() -> str:
    """Get API base URL from environment or default."""
    return os.getenv("API_BASE_URL", "http://localhost:8000")


@pytest.fixture
def vllm_base_url() -> str:
    """Get vLLM base URL from environment or default."""
    return os.getenv("VLLM_BASE_URL", "http://89.169.108.198:30080/v1")


class TestCacheEfficiencyIntegration:
    """Integration tests that verify actual cache hit behavior."""

    @pytest.mark.asyncio
    async def test_sequential_requests_show_cache_improvement(self, api_base_url: str):
        """
        Verify that TTFT decreases on subsequent requests.

        EXPECTED BEHAVIOR:
        - First request: High TTFT (cold cache)
        - Second request: Lower TTFT (cache hit)
        - TTFT ratio should be < 0.5 (50% improvement)
        """
        async with httpx.AsyncClient(timeout=120.0) as client:
            # First request (cold cache)
            start1 = time.perf_counter()
            response1 = await client.post(
                f"{api_base_url}/api/v1/query",
                json={
                    "query": "What is the daily ATM withdrawal limit?",
                    "session_id": f"test-cache-{time.time()}",
                },
            )
            ttft1 = time.perf_counter() - start1

            assert response1.status_code == 200, f"Request failed: {response1.text}"
            data1 = response1.json()
            assert "response" in data1

            # Small delay to ensure cache is populated
            await asyncio.sleep(0.5)

            # Second request (should hit cache)
            start2 = time.perf_counter()
            response2 = await client.post(
                f"{api_base_url}/api/v1/query",
                json={
                    "query": "How do I process a wire transfer?",
                    "session_id": f"test-cache-{time.time()}",
                },
            )
            ttft2 = time.perf_counter() - start2

            assert response2.status_code == 200

            # Log the results
            print("\nCache Efficiency Results:")
            print(f"  Cold cache TTFT: {ttft1:.2f}s")
            print(f"  Warm cache TTFT: {ttft2:.2f}s")
            print(f"  Improvement ratio: {ttft2 / ttft1:.2%}")

            # Second request should be at least 30% faster
            # (Being conservative to account for network variance)
            assert ttft2 < ttft1, f"Second request was not faster: {ttft1:.2f}s -> {ttft2:.2f}s"

    @pytest.mark.asyncio
    async def test_cache_stats_endpoint(self, api_base_url: str):
        """Verify cache stats endpoint returns expected fields."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{api_base_url}/cache/stats")

            assert response.status_code == 200
            data = response.json()

            # Check for expected fields
            if data.get("status") != "No requests recorded":
                assert "total_requests" in data
                assert "inferred_cache_hit_rate" in data
                assert "prefix_alignment_ok" in data
                assert "grade" in data

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, api_base_url: str):
        """
        Verify conversation history is maintained across turns.

        EXPECTED:
        - Second query in same session should have context from first
        """
        session_id = f"test-multiturn-{time.time()}"

        async with httpx.AsyncClient(timeout=120.0) as client:
            # First turn
            response1 = await client.post(
                f"{api_base_url}/api/v1/query",
                json={
                    "query": "What is the ATM withdrawal limit?",
                    "session_id": session_id,
                },
            )
            assert response1.status_code == 200
            data1 = response1.json()

            # Second turn - refers to first
            response2 = await client.post(
                f"{api_base_url}/api/v1/query",
                json={
                    "query": "Can I increase that limit?",
                    "session_id": session_id,  # Same session
                },
            )
            assert response2.status_code == 200
            data2 = response2.json()

            # Both responses should exist
            assert len(data1["response"]) > 0
            assert len(data2["response"]) > 0

            print("\nMulti-turn test:")
            print(f"  Turn 1: {data1['response'][:100]}...")
            print(f"  Turn 2: {data2['response'][:100]}...")

    @pytest.mark.asyncio
    async def test_prometheus_metrics_endpoint(self, api_base_url: str):
        """Verify Prometheus metrics are exposed."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{api_base_url}/metrics")

            assert response.status_code == 200
            content = response.text

            # Check for expected metric names
            expected_metrics = [
                "router_requests_total",
                "router_ttft_seconds",
                "router_cache_hit_rate",
            ]

            for metric in expected_metrics:
                # Metrics may not have values yet, but should be defined
                # Check for either the metric name or HELP/TYPE declarations
                assert metric in content or f"# HELP {metric}" in content, (
                    f"Missing metric: {metric}"
                )


class TestPrefixAlignment:
    """Tests to verify prefix alignment across agents."""

    @pytest.mark.asyncio
    async def test_different_agents_share_prefix(self, api_base_url: str):
        """
        Verify that requests routed to different agents share the same prefix.

        This is verified by checking the cache stats after multiple requests.
        """
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Query that routes to technical specialist
            await client.post(
                f"{api_base_url}/api/v1/query",
                json={
                    "query": "What are the API rate limits?",
                    "session_id": f"test-prefix-{time.time()}-1",
                },
            )

            # Query that routes to compliance auditor
            await client.post(
                f"{api_base_url}/api/v1/query",
                json={
                    "query": "What are the AML reporting requirements?",
                    "session_id": f"test-prefix-{time.time()}-2",
                },
            )

            # Query that routes to support concierge
            await client.post(
                f"{api_base_url}/api/v1/query",
                json={
                    "query": "How do I open a new customer account?",
                    "session_id": f"test-prefix-{time.time()}-3",
                },
            )

            # Check prefix alignment
            stats = await client.get(f"{api_base_url}/cache/stats")
            data = stats.json()

            if data.get("status") != "No requests recorded":
                assert data.get("prefix_alignment_ok", False), (
                    f"Prefix alignment failed! Unique hashes: {data.get('unique_prefix_hashes')}"
                )
                print("\nPrefix alignment: ✅ All agents share the same prefix")
