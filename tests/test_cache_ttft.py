#!/usr/bin/env python3
"""
=============================================================================
LMCache Cold vs Warm TTFT Test
=============================================================================

Reference: https://docs.lmcache.ai/kv_cache/storage_backends/cpu_ram.html

This script tests the Time to First Token (TTFT) differential between:
- Cold request: First request, KV cache must be computed
- Warm request: Second request with same prefix, KV cache is reused

USAGE:
------
    # Ensure server is running first:
    uv run uvicorn src.main:app --port 8000

    # Run the test:
    uv run python tests/test_cache_ttft.py

EXPECTED RESULT:
----------------
With LMCache enabled, warm TTFT should be significantly faster than cold TTFT.
The test asserts that warm_ttft < cold_ttft.

=============================================================================
"""

import sys
import time

import httpx

# Configuration
API_BASE_URL = "http://localhost:8000"
STREAM_ENDPOINT = f"{API_BASE_URL}/api/v1/query/stream"
QUERY_ENDPOINT = f"{API_BASE_URL}/api/v1/query"

# Test queries - same prefix (system + manual), different user questions
COLD_QUERY = "What is the CTR filing threshold for cash transactions?"
WARM_QUERY = "What is the CTR filing threshold for cash transactions?"


def measure_ttft_streaming(query: str, session_id: str) -> tuple[float, str]:
    """
    Measure TTFT using the streaming endpoint.

    Returns:
        (ttft_seconds, response_content)
    """
    payload = {"query": query, "session_id": session_id}

    start_time = time.perf_counter()
    first_token_time = None
    response_content = ""

    with httpx.stream("POST", STREAM_ENDPOINT, json=payload, timeout=120.0) as response:
        if response.status_code != 200:
            raise RuntimeError(f"API error: {response.status_code}")

        for line in response.iter_lines():
            if not line:
                continue

            if line.startswith("event:"):
                event_type = line[6:].strip()
                continue

            if line.startswith("data:"):
                import json

                try:
                    data = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue

                if event_type == "metadata":
                    # First token received!
                    first_token_time = time.perf_counter()
                    ttft = first_token_time - start_time
                    print(f"  ⚡ TTFT: {ttft:.3f}s")

                elif event_type == "token":
                    content = data.get("content", "")
                    response_content += content
                    print(content, end="", flush=True)

                elif event_type == "done":
                    print()  # New line after streaming

    if first_token_time is None:
        raise RuntimeError("No tokens received")

    return first_token_time - start_time, response_content


def measure_ttft_non_streaming(query: str, session_id: str) -> tuple[float, str]:
    """
    Measure TTFT using the non-streaming endpoint.
    Returns the ttft_seconds from the API response.

    Returns:
        (ttft_seconds, response_content)
    """
    payload = {"query": query, "session_id": session_id}

    with httpx.Client(timeout=120.0) as client:
        response = client.post(QUERY_ENDPOINT, json=payload)

        if response.status_code != 200:
            raise RuntimeError(f"API error: {response.status_code}")

        data = response.json()
        ttft = data.get("ttft_seconds", 0.0)
        content = data.get("response", "")

        print(f"  TTFT: {ttft:.3f}s")
        print(f"  Agents: {data.get('agents_used', [])}")
        print(f"  Response: {content[:100]}...")

        return ttft, content


def run_ttft_comparison_test(use_streaming: bool = True) -> dict:
    """
    Run the cold vs warm TTFT comparison test.

    Args:
        use_streaming: If True, use streaming endpoint for real TTFT measurement.
                      If False, use non-streaming endpoint with server-reported TTFT.

    Returns:
        dict with test results
    """
    print("=" * 70)
    print("LMCache Cold vs Warm TTFT Comparison Test")
    print("=" * 70)
    print(f"Mode: {'Streaming' if use_streaming else 'Non-streaming'}")
    print(f"API: {API_BASE_URL}")
    print()

    measure_ttft = measure_ttft_streaming if use_streaming else measure_ttft_non_streaming

    # Test 1: Cold request (first request, no cache)
    print("─" * 70)
    print("📥 COLD REQUEST (first request, no cache)")
    print("─" * 70)
    print(f"Query: {COLD_QUERY}")
    print()

    # Use consistent session_id to maximize cache affinity
    session_id = "test-cache-optimization-v1"

    cold_ttft, cold_response = measure_ttft(COLD_QUERY, session_id)
    print()

    # Small delay to ensure cache is written
    time.sleep(0.5)

    # Test 2: Warm request (same prefix, should hit cache)
    print("─" * 70)
    print("🔥 WARM REQUEST (same prefix, should hit cache)")
    print("─" * 70)
    print(f"Query: {WARM_QUERY}")
    print()

    warm_ttft, warm_response = measure_ttft(WARM_QUERY, session_id)
    print()

    # Calculate improvement
    ttft_improvement = cold_ttft - warm_ttft
    ttft_ratio = cold_ttft / warm_ttft if warm_ttft > 0 else float("inf")

    # Results summary
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Cold TTFT:  {cold_ttft:.3f}s")
    print(f"Warm TTFT:  {warm_ttft:.3f}s")
    print(f"Improvement: {ttft_improvement:.3f}s ({ttft_ratio:.1f}x faster)")
    print()

    # Grade the result
    if warm_ttft < cold_ttft * 0.5:
        grade = "A"
        status = "✅ EXCELLENT - Significant cache hit detected!"
    elif warm_ttft < cold_ttft * 0.8:
        grade = "B"
        status = "✅ GOOD - Cache is working"
    elif warm_ttft < cold_ttft:
        grade = "C"
        status = "⚠️ MARGINAL - Slight improvement detected"
    else:
        grade = "F"
        status = "❌ FAIL - No cache improvement detected"

    print(f"Grade: {grade}")
    print(f"Status: {status}")
    print("=" * 70)

    return {
        "cold_ttft": cold_ttft,
        "warm_ttft": warm_ttft,
        "improvement_seconds": ttft_improvement,
        "improvement_ratio": ttft_ratio,
        "grade": grade,
        "passed": warm_ttft < cold_ttft,
    }


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Test LMCache cold vs warm TTFT performance")
    parser.add_argument(
        "--no-stream", action="store_true", help="Use non-streaming endpoint instead of streaming"
    )
    parser.add_argument(
        "--assert",
        action="store_true",
        dest="do_assert",
        help="Assert that warm TTFT is faster than cold TTFT (for CI)",
    )

    args = parser.parse_args()

    try:
        results = run_ttft_comparison_test(use_streaming=not args.no_stream)

        if args.do_assert:
            assert results["passed"], (
                f"Cache test FAILED: warm_ttft ({results['warm_ttft']:.3f}s) "
                f"is not faster than cold_ttft ({results['cold_ttft']:.3f}s)"
            )
            print("\n✅ Assertion passed: warm_ttft < cold_ttft")

        sys.exit(0 if results["passed"] else 1)

    except httpx.ConnectError:
        print("❌ ERROR: Cannot connect to server at", API_BASE_URL)
        print("   Make sure the server is running:")
        print("   uv run uvicorn src.main:app --port 8000")
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
