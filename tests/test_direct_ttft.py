#!/usr/bin/env python3
"""
=============================================================================
LMCache Direct Router TTFT Test
=============================================================================

This script tests TTFT by calling the LLM directly (not through the API).
This isolates the cache performance from workflow overhead.

METRICS:
--------
- Router TTFT: Direct LLM call for router agent
- End-to-End TTFT: Full API workflow (for comparison)

USAGE:
------
    uv run python tests/test_direct_ttft.py

=============================================================================
"""

import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI

from src.config.settings import get_settings
from src.prompts.manager import DeterministicPromptBuilder, load_manual


def get_llm(streaming: bool = False) -> ChatOpenAI:
    """Get the LLM client configured for vLLM."""
    settings = get_settings()
    return ChatOpenAI(
        base_url=settings.vllm_base_url,
        api_key=settings.vllm_api_key,
        model=settings.vllm_model,
        temperature=0.7,
        max_tokens=256,  # Short response for faster testing
        streaming=streaming,
    )


def measure_direct_ttft(prompt: str, label: str) -> float:
    """
    Measure TTFT by calling the LLM directly.

    Returns:
        ttft_seconds
    """
    llm = get_llm(streaming=True)

    start_time = time.perf_counter()
    first_token_time = None

    print(f"\n  [{label}] Sending request...")

    for chunk in llm.stream(prompt):
        if chunk.content and first_token_time is None:
            first_token_time = time.perf_counter()
            ttft = first_token_time - start_time
            print(f"  ⚡ TTFT: {ttft:.3f}s")
            print(f"  First token: {chunk.content[:50]}...")
            break  # We only need the first token

    if first_token_time is None:
        raise RuntimeError("No tokens received")

    return first_token_time - start_time


def run_direct_ttft_test():
    """Run the cold vs warm TTFT comparison using direct LLM calls."""
    print("=" * 70)
    print("LMCache Direct Router TTFT Test")
    print("=" * 70)
    print("This test calls the LLM directly (no API overhead)")
    print()

    # Load manual and create prompt builder
    settings = get_settings()
    manual = load_manual(Path(settings.manual_path))
    prompt_builder = DeterministicPromptBuilder(manual)

    print(f"Manual loaded: {len(manual)} chars")
    print(f"Model: {settings.vllm_model}")
    print(f"Endpoint: {settings.vllm_base_url}")
    print()

    # Build router prompt
    query = "What is the CTR filing threshold for cash transactions?"
    router_prompt = prompt_builder.build("router", [], query)

    print(f"Prompt length: {len(router_prompt)} chars (~{len(router_prompt) // 4} tokens)")
    print()

    # Test 1: Cold request
    print("─" * 70)
    print("📥 COLD REQUEST (first request, no cache)")
    print("─" * 70)
    cold_ttft = measure_direct_ttft(router_prompt, "COLD")

    # Brief pause to ensure cache is written
    time.sleep(0.5)

    # Test 2: Warm request 1
    print()
    print("─" * 70)
    print("🔥 WARM REQUEST 1 (should hit cache)")
    print("─" * 70)
    warm_ttft_1 = measure_direct_ttft(router_prompt, "WARM-1")

    # Test 3: Warm request 2
    print()
    print("─" * 70)
    print("🔥 WARM REQUEST 2 (consistency check)")
    print("─" * 70)
    warm_ttft_2 = measure_direct_ttft(router_prompt, "WARM-2")

    # Results
    print()
    print("=" * 70)
    print("DIRECT ROUTER TTFT RESULTS")
    print("=" * 70)
    print(f"Cold TTFT:   {cold_ttft:.3f}s")
    print(f"Warm TTFT 1: {warm_ttft_1:.3f}s")
    print(f"Warm TTFT 2: {warm_ttft_2:.3f}s")

    best_warm = min(warm_ttft_1, warm_ttft_2)
    improvement = cold_ttft - best_warm
    ratio = cold_ttft / best_warm if best_warm > 0 else float("inf")

    print(f"Best Warm:   {best_warm:.3f}s")
    print(f"Improvement: {improvement:.3f}s ({ratio:.1f}x faster)")
    print()

    # Grade
    if best_warm < cold_ttft * 0.5:
        grade = "A"
        status = "✅ EXCELLENT - Significant cache hit detected!"
    elif best_warm < cold_ttft * 0.8:
        grade = "B"
        status = "✅ GOOD - Cache is working"
    elif best_warm < cold_ttft:
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
        "warm_ttft_1": warm_ttft_1,
        "warm_ttft_2": warm_ttft_2,
        "best_warm": best_warm,
        "improvement": improvement,
        "ratio": ratio,
        "grade": grade,
    }


if __name__ == "__main__":
    try:
        results = run_direct_ttft_test()
        sys.exit(0 if results["grade"] != "F" else 1)
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
