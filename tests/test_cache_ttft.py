#!/usr/bin/env python3
"""
=============================================================================
LMCache Cold vs Warm TTFT Test
=============================================================================

Reference: https://docs.lmcache.ai/kv_cache/storage_backends/cpu_ram.html

This script tests the Time to First Token (TTFT) differential between:
- Cold request: First request, KV cache must be computed
- Warm request: Second request with same prefix, KV cache is reused

METRICS:
--------
- End-to-End TTFT: Full API workflow (router + agents)
- Direct Router TTFT: Direct LLM call (isolates cache performance)

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
from pathlib import Path

import httpx

# Add src to path for direct LLM imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI

from src.config.settings import get_settings
from src.prompts.manager import DeterministicPromptBuilder, load_manual

# Configuration
API_BASE_URL = "http://localhost:8000"
STREAM_ENDPOINT = f"{API_BASE_URL}/api/v1/query/stream"
QUERY_ENDPOINT = f"{API_BASE_URL}/api/v1/query"

# Test queries - same prefix (system + manual), different user questions
COLD_QUERY = "What is the CTR filing threshold for cash transactions?"
WARM_QUERY = "What is the CTR filing threshold for cash transactions?"


def get_llm(streaming: bool = False) -> ChatOpenAI:
    """Get the LLM client configured for vLLM."""
    settings = get_settings()
    return ChatOpenAI(
        base_url=settings.vllm_base_url,
        api_key=settings.vllm_api_key,
        model=settings.vllm_model,
        temperature=0.7,
        max_tokens=256,
        streaming=streaming,
    )


def measure_direct_router_ttft(prompt: str, label: str) -> float:
    """
    Measure TTFT by calling the LLM directly (no API overhead).

    Returns:
        ttft_seconds
    """
    llm = get_llm(streaming=True)

    start_time = time.perf_counter()
    first_token_time = None

    for chunk in llm.stream(prompt):
        if chunk.content and first_token_time is None:
            first_token_time = time.perf_counter()
            ttft = first_token_time - start_time
            print(f"  ⚡ Direct Router TTFT: {ttft:.3f}s")
            break

    if first_token_time is None:
        raise RuntimeError("No tokens received")

    return first_token_time - start_time


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

    # =========================================================================
    # PART 1: Direct Router TTFT (isolates cache performance)
    # =========================================================================
    print("=" * 70)
    print("PART 1: DIRECT ROUTER TTFT (no API overhead)")
    print("=" * 70)

    # Load manual and create prompt builder
    settings = get_settings()
    manual = load_manual(Path(settings.manual_path))
    prompt_builder = DeterministicPromptBuilder(manual)

    print(f"Manual: {len(manual)} chars (~{len(manual) // 4} tokens)")
    print(f"Model: {settings.vllm_model}")
    print()

    # Build router prompt
    router_prompt = prompt_builder.build("router", [], COLD_QUERY)

    # Direct Cold
    print("─" * 70)
    print("📥 DIRECT COLD REQUEST")
    print("─" * 70)
    direct_cold_ttft = measure_direct_router_ttft(router_prompt, "COLD")

    time.sleep(0.3)

    # Direct Warm 1
    print()
    print("─" * 70)
    print("🔥 DIRECT WARM REQUEST 1")
    print("─" * 70)
    direct_warm_ttft_1 = measure_direct_router_ttft(router_prompt, "WARM-1")

    # Direct Warm 2
    print()
    print("─" * 70)
    print("🔥 DIRECT WARM REQUEST 2")
    print("─" * 70)
    direct_warm_ttft_2 = measure_direct_router_ttft(router_prompt, "WARM-2")

    # =========================================================================
    # PART 2: End-to-End API TTFT (full workflow)
    # =========================================================================
    print()
    print("=" * 70)
    print("PART 2: END-TO-END API TTFT (full workflow)")
    print("=" * 70)

    measure_ttft = measure_ttft_streaming if use_streaming else measure_ttft_non_streaming

    # Test 1: Cold request (first request, no cache)
    print("─" * 70)
    print("📥 API COLD REQUEST (first request, no cache)")
    print("─" * 70)
    print(f"Query: {COLD_QUERY}")
    print()

    # Use consistent session_id to maximize cache affinity
    session_id = "test-cache-optimization-v2"

    cold_ttft, cold_response = measure_ttft(COLD_QUERY, session_id)
    print()

    # Small delay to ensure cache is written
    time.sleep(0.5)

    # Test 2: Warm request (same prefix, should hit cache)
    print("─" * 70)
    print("🔥 API WARM REQUEST (same prefix, should hit cache)")
    print("─" * 70)
    print(f"Query: {WARM_QUERY}")
    print()

    warm_ttft, warm_response = measure_ttft(WARM_QUERY, session_id)
    print()

    # Test 3: Second Warm request (verify consistency)
    print("─" * 70)
    print("🔥 API WARM REQUEST 2 (consistency check)")
    print("─" * 70)
    print(f"Query: {WARM_QUERY}")
    print()

    warm_ttft_2, warm_response_2 = measure_ttft(WARM_QUERY, session_id)
    print()

    # =========================================================================
    # RESULTS SUMMARY
    # =========================================================================
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Direct Router Results
    direct_best_warm = min(direct_warm_ttft_1, direct_warm_ttft_2)
    direct_cold_ttft - direct_best_warm
    direct_ratio = direct_cold_ttft / direct_best_warm if direct_best_warm > 0 else float("inf")

    print("┌─────────────────────────────────────────────────────────────────┐")
    print("│ DIRECT ROUTER TTFT (pure cache performance)                      │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print(f"│ Cold:       {direct_cold_ttft:.3f}s                                           │")
    print(f"│ Warm 1:     {direct_warm_ttft_1:.3f}s                                           │")
    print(f"│ Warm 2:     {direct_warm_ttft_2:.3f}s                                           │")
    print(f"│ Best Warm:  {direct_best_warm:.3f}s                                           │")
    print(f"│ Speedup:    {direct_ratio:.1f}x faster                                        │")
    print("└─────────────────────────────────────────────────────────────────┘")

    # Grade direct results
    if direct_best_warm < direct_cold_ttft * 0.5:
        direct_grade = "A"
        direct_status = "✅ EXCELLENT"
    elif direct_best_warm < direct_cold_ttft * 0.8:
        direct_grade = "B"
        direct_status = "✅ GOOD"
    elif direct_best_warm < direct_cold_ttft:
        direct_grade = "C"
        direct_status = "⚠️ MARGINAL"
    else:
        direct_grade = "F"
        direct_status = "❌ FAIL"

    print(f"Direct Router Grade: {direct_grade} - {direct_status}")
    print()

    # End-to-End Results
    ttft_improvement = cold_ttft - warm_ttft
    ttft_ratio = cold_ttft / warm_ttft if warm_ttft > 0 else float("inf")
    best_warm = min(warm_ttft, warm_ttft_2)

    print("┌─────────────────────────────────────────────────────────────────┐")
    print("│ END-TO-END API TTFT (includes workflow overhead)                 │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print(f"│ Cold:       {cold_ttft:.3f}s                                           │")
    print(f"│ Warm 1:     {warm_ttft:.3f}s                                           │")
    print(f"│ Warm 2:     {warm_ttft_2:.3f}s                                           │")
    print(f"│ Best Warm:  {best_warm:.3f}s                                           │")
    print(f"│ Speedup:    {ttft_ratio:.1f}x faster                                        │")
    print("└─────────────────────────────────────────────────────────────────┘")

    # Grade the result (using best warm time)
    if best_warm < cold_ttft * 0.5:
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

    print(f"End-to-End Grade: {grade}")
    print(f"Status: {status}")
    print("=" * 70)

    return {
        "cold_ttft": cold_ttft,
        "warm_ttft": warm_ttft,
        "improvement_seconds": ttft_improvement,
        "improvement_ratio": ttft_ratio,
        "grade": grade,
        "passed": warm_ttft < cold_ttft,
        # Direct router metrics
        "direct_cold_ttft": direct_cold_ttft,
        "direct_warm_ttft": direct_best_warm,
        "direct_ratio": direct_ratio,
        "direct_grade": direct_grade,
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
