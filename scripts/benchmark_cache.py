#!/usr/bin/env python
"""
=============================================================================
Cache Efficiency Benchmark Script
=============================================================================

Demonstrates and measures cache efficiency of the multi-agent system.

USAGE:
------
# Run benchmark against local server
uv run python scripts/benchmark_cache.py

# Run against remote server
uv run python scripts/benchmark_cache.py --url http://your-server:8000

OUTPUT:
-------
Cold Cache TTFT: 12.34s
Warm Cache TTFT: 1.56s
Cache Speedup: 7.9x
Inferred Cache Hit Rate: 87%
=============================================================================
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass

import httpx


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    cold_ttft: float
    warm_ttfts: list[float]
    avg_warm_ttft: float
    speedup: float
    cache_hit_rate: float


async def run_benchmark(
    base_url: str,
    num_warm_requests: int = 5,
) -> BenchmarkResult:
    """Run cache efficiency benchmark."""

    queries = [
        "What is the daily ATM withdrawal limit?",
        "How do I process a wire transfer?",
        "What are the AML reporting requirements?",
        "How do I open a new business account?",
        "What are the API rate limits for customer lookup?",
        "Can employees accept gifts from vendors?",
    ]

    async with httpx.AsyncClient(timeout=120.0) as client:
        # First request - cold cache
        print("\n🔄 Sending first request (cold cache)...")
        session_id = f"benchmark-{time.time()}"

        start = time.perf_counter()
        response = await client.post(
            f"{base_url}/api/v1/query",
            json={
                "query": queries[0],
                "session_id": session_id,
            },
        )
        cold_ttft = time.perf_counter() - start

        if response.status_code != 200:
            print(f"❌ Request failed: {response.text}")
            sys.exit(1)

        print(f"✅ Cold cache TTFT: {cold_ttft:.2f}s")

        # Wait for cache to settle
        await asyncio.sleep(1.0)

        # Subsequent requests - should hit cache
        print(f"\n🔄 Sending {num_warm_requests} warm cache requests...")
        warm_ttfts = []

        for i in range(num_warm_requests):
            query = queries[(i + 1) % len(queries)]
            session_id = f"benchmark-{time.time()}-{i}"

            start = time.perf_counter()
            response = await client.post(
                f"{base_url}/api/v1/query",
                json={
                    "query": query,
                    "session_id": session_id,
                },
            )
            ttft = time.perf_counter() - start

            if response.status_code == 200:
                warm_ttfts.append(ttft)
                data = response.json()
                agents = data.get("agents_used", [])
                print(f"  Request {i + 1}: {ttft:.2f}s (agents: {', '.join(agents)})")
            else:
                print(f"  Request {i + 1}: FAILED - {response.status_code}")

        # Calculate stats
        avg_warm_ttft = sum(warm_ttfts) / len(warm_ttfts) if warm_ttfts else cold_ttft
        speedup = cold_ttft / avg_warm_ttft if avg_warm_ttft > 0 else 1.0

        # Infer cache hits (warm requests < 50% of cold)
        cache_hits = sum(1 for t in warm_ttfts if t < cold_ttft * 0.5)
        cache_hit_rate = cache_hits / len(warm_ttfts) if warm_ttfts else 0.0

        return BenchmarkResult(
            cold_ttft=cold_ttft,
            warm_ttfts=warm_ttfts,
            avg_warm_ttft=avg_warm_ttft,
            speedup=speedup,
            cache_hit_rate=cache_hit_rate,
        )


async def get_cache_stats(base_url: str) -> dict:
    """Get cache stats from the API."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(f"{base_url}/cache/stats")
        if response.status_code == 200:
            return response.json()
        return {}


def print_report(result: BenchmarkResult, stats: dict):
    """Print benchmark report."""
    print("\n" + "=" * 60)
    print("📊 CACHE EFFICIENCY BENCHMARK REPORT")
    print("=" * 60)

    print(f"\n🥶 Cold Cache TTFT:     {result.cold_ttft:.2f}s")
    print(f"🔥 Avg Warm Cache TTFT: {result.avg_warm_ttft:.2f}s")
    print(f"⚡ Cache Speedup:       {result.speedup:.1f}x")
    print(f"✅ Inferred Cache Hit:  {result.cache_hit_rate * 100:.0f}%")

    # Grade the result
    if result.speedup >= 5.0 and result.cache_hit_rate >= 0.8:
        grade = "A - Excellent"
        emoji = "🏆"
    elif result.speedup >= 3.0 and result.cache_hit_rate >= 0.5:
        grade = "B - Good"
        emoji = "👍"
    elif result.speedup >= 2.0:
        grade = "C - Acceptable"
        emoji = "🆗"
    else:
        grade = "D - Needs Improvement"
        emoji = "⚠️"

    print(f"\n{emoji} Grade: {grade}")

    # Server stats if available
    if stats and stats.get("status") != "No requests recorded":
        print("\n📈 Server-Side Metrics:")
        print(f"   Total Requests:      {stats.get('total_requests', 'N/A')}")
        print(
            f"   Prefix Alignment:    {'✅ OK' if stats.get('prefix_alignment_ok') else '❌ MISMATCH'}"
        )
        print(f"   Unique Prefix Hashes: {stats.get('unique_prefix_hashes', 'N/A')}")

    print("\n" + "=" * 60)


async def main():
    parser = argparse.ArgumentParser(description="Benchmark cache efficiency")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the API server",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=5,
        help="Number of warm cache requests to send",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    args = parser.parse_args()

    print(f"🎯 Benchmarking: {args.url}")
    print(f"📝 Warm requests: {args.requests}")

    try:
        result = await run_benchmark(args.url, args.requests)
        stats = await get_cache_stats(args.url)

        if args.json:
            output = {
                "cold_ttft_seconds": result.cold_ttft,
                "avg_warm_ttft_seconds": result.avg_warm_ttft,
                "speedup": result.speedup,
                "cache_hit_rate": result.cache_hit_rate,
                "server_stats": stats,
            }
            print(json.dumps(output, indent=2))
        else:
            print_report(result, stats)

    except httpx.ConnectError:
        print(f"\n❌ Could not connect to {args.url}")
        print("   Make sure the server is running: uv run uvicorn src.main:app")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
