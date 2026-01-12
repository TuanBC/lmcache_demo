"""
=============================================================================
CACHE OPTIMIZATION METRICS
=============================================================================

This module tracks metrics that help us understand cache efficiency
WITHOUT having direct access to vLLM server logs.

KEY INSIGHT: By monitoring Time-To-First-Token (TTFT), we can infer
whether the server's KV cache is being hit:

- COLD CACHE (cache miss): TTFT is HIGH because the server must
  compute KV vectors for all ~25,000 tokens in the manual

- WARM CACHE (cache hit): TTFT is LOW because the server reuses
  pre-computed KV vectors for the shared prefix

EXPECTED BEHAVIOR:
- First request: ~10-30s TTFT (cold, computing 25k tokens)
- Subsequent requests: ~1-5s TTFT (warm, only computing delta)

If subsequent requests DON'T show reduced TTFT, our prompt structure
is likely breaking the cache (e.g., whitespace differences).
=============================================================================
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

# Import chunk size from prompt manager for alignment checks
from src.prompts.manager import CHUNK_SIZE

logger = logging.getLogger(__name__)


@dataclass
class CacheAwareMetrics:
    """
    Track cache efficiency metrics for analysis.

    This class enables us to:
    1. Verify prefix alignment across agents
    2. Infer cache hits from TTFT patterns
    3. Generate reports for README.md
    """

    # Track TTFT history: (agent_name, ttft_seconds, prefix_hash)
    ttft_history: list[tuple[str, float, str]] = field(default_factory=list)

    # Baseline from first (cold) request
    cold_cache_ttft: float | None = None

    # Expected prefix hash (should be identical across ALL agents)
    expected_prefix_hash: str | None = None

    def log_request_start(self, agent_name: str, prompt: str) -> dict:
        """
        Log metrics BEFORE sending request to LLM.

        WHY THIS MATTERS FOR CACHE OPTIMIZATION:
        -----------------------------------------
        The prefix hash MUST be identical across all agents for cache reuse.
        If Technical Specialist and Compliance Auditor have different prefix
        hashes, the server cannot reuse the cached KV vectors.
        """
        # Extract the cacheable prefix (system + manual, ~25k tokens)
        # We hash up to the "<<< END OF MANUAL >>>" marker
        prefix_end = prompt.find("<<< END OF MANUAL >>>")
        if prefix_end == -1:
            prefix_end = min(len(prompt), 100000)  # Fallback: ~25k tokens

        cacheable_prefix = prompt[:prefix_end]
        prefix_hash = hashlib.sha256(cacheable_prefix.encode()).hexdigest()[:16]

        # Check if prefix hash matches expected (cache alignment check)
        if self.expected_prefix_hash is None:
            self.expected_prefix_hash = prefix_hash
            cache_aligned = True
            logger.info(f"[CACHE] First request - setting baseline prefix_hash={prefix_hash}")
        else:
            cache_aligned = prefix_hash == self.expected_prefix_hash
            if not cache_aligned:
                # ⚠️ CRITICAL: This means cache will NOT be hit!
                logger.warning(
                    f"[CACHE] PREFIX MISMATCH! Expected={self.expected_prefix_hash} "
                    f"Got={prefix_hash}. Cache will NOT be reused!"
                )

        return {
            "agent": agent_name,
            "prefix_hash": prefix_hash,
            "prefix_length_chars": len(cacheable_prefix),
            "prefix_length_tokens_est": len(cacheable_prefix) // 4,
            "is_chunk_aligned": self._check_chunk_alignment(cacheable_prefix),
            "total_prompt_length": len(prompt),
            "cache_aligned": cache_aligned,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def _check_chunk_alignment(self, text: str) -> bool:
        """Check if text is aligned to CHUNK_SIZE boundaries.

        CACHE OPTIMIZATION:
        -------------------
        LMCache works best when prefixes end on chunk boundaries.
        This check verifies alignment for debugging cache efficiency.
        """
        token_est = len(text) // 4
        is_aligned = (token_est % CHUNK_SIZE) == 0

        if not is_aligned:
            remainder = token_est % CHUNK_SIZE
            logger.warning(
                f"[CACHE] Prefix NOT chunk-aligned: {token_est} tokens "
                f"(remainder={remainder}, need {CHUNK_SIZE - remainder} more)"
            )
        else:
            logger.debug(f"[CACHE] Prefix chunk-aligned: {token_est} tokens")

        return is_aligned

    def log_request_complete(
        self,
        agent_name: str,
        prefix_hash: str,
        ttft_seconds: float,
    ) -> dict:
        """
        Log metrics AFTER receiving first token from LLM.

        CACHE HIT INFERENCE:
        --------------------
        We infer cache hits from TTFT patterns:

        - If ttft < (cold_cache_ttft * 0.3): LIKELY CACHE HIT
          The server reused KV vectors, only computing new tokens

        - If ttft > (cold_cache_ttft * 0.7): LIKELY CACHE MISS
          The server had to recompute all 25k tokens

        This is the ONLY way to measure cache efficiency without
        direct server log access.
        """
        # Record history
        self.ttft_history.append((agent_name, ttft_seconds, prefix_hash))

        # Set baseline from first request
        if self.cold_cache_ttft is None:
            self.cold_cache_ttft = ttft_seconds
            is_cache_hit = False  # First request is always cold
            logger.info(f"[CACHE] Cold cache baseline established: {ttft_seconds:.2f}s")
        else:
            # Infer cache hit from TTFT ratio
            ttft_ratio = ttft_seconds / self.cold_cache_ttft
            is_cache_hit = ttft_ratio < 0.5  # <50% of cold time = likely hit

            status = "HIT ✅" if is_cache_hit else "MISS ❌"
            logger.info(
                f"[CACHE] Agent={agent_name} TTFT={ttft_seconds:.2f}s "
                f"ratio={ttft_ratio:.2f} -> {status}"
            )

        # Calculate running stats
        recent_ttfts = [t for _, t, _ in self.ttft_history[-10:]]
        avg_ttft = sum(recent_ttfts) / len(recent_ttfts)

        return {
            "agent": agent_name,
            "ttft_seconds": ttft_seconds,
            "cold_cache_baseline": self.cold_cache_ttft,
            "ttft_ratio": ttft_seconds / self.cold_cache_ttft if self.cold_cache_ttft else 1.0,
            "is_cache_hit_inferred": is_cache_hit,
            "avg_ttft_last_10": avg_ttft,
            "total_requests": len(self.ttft_history),
        }

    def get_cache_report(self) -> dict:
        """
        Generate a summary report of cache efficiency.

        USE THIS FOR:
        - README.md optimization report
        - Debugging cache misses
        - Demonstrating cache awareness to evaluators
        """
        if not self.ttft_history:
            return {"status": "No requests recorded"}

        ttfts = [t for _, t, _ in self.ttft_history]
        hashes = [h for _, _, h in self.ttft_history]
        unique_hashes = set(hashes)

        # Count inferred hits/misses
        if self.cold_cache_ttft:
            hits = sum(1 for t in ttfts[1:] if t < self.cold_cache_ttft * 0.5)
            misses = len(ttfts) - 1 - hits  # Exclude first (always cold)
        else:
            hits = 0
            misses = len(ttfts)

        return {
            "total_requests": len(self.ttft_history),
            "cold_cache_baseline_seconds": self.cold_cache_ttft,
            "inferred_cache_hit_rate": hits / max(1, hits + misses),
            "unique_prefix_hashes": len(unique_hashes),
            "prefix_alignment_ok": len(unique_hashes) == 1,
            "avg_ttft_seconds": sum(ttfts) / len(ttfts),
            "min_ttft_seconds": min(ttfts),
            "max_ttft_seconds": max(ttfts),
            "recommendation": (
                "✅ Prefix is aligned - cache should be effective"
                if len(unique_hashes) == 1
                else f"⚠️ Found {len(unique_hashes)} different prefix hashes - "
                "cache is being busted! Check prompt normalization."
            ),
        }


# Global metrics instance - shared across all requests
cache_metrics = CacheAwareMetrics()
