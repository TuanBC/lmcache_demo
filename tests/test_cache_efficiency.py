"""
=============================================================================
Cache Efficiency Tests
=============================================================================

Tests to verify that prefix caching is working correctly.

WHAT THESE TESTS VERIFY:
------------------------
1. Prefix hashes are identical across all agents
2. TTFT decreases on subsequent requests (cache hit)
3. Whitespace normalization is consistent
=============================================================================
"""

import pytest

from src.cache.metrics import CacheAwareMetrics
from src.prompts.manager import DeterministicPromptBuilder


class TestPrefixAlignment:
    """Tests to verify prefix alignment across agents."""
    
    def test_prefix_hash_identical_across_agents(self, sample_manual: str):
        """
        Verify that all agents produce the same prefix hash.
        
        This is CRITICAL for cache efficiency. If agents have different
        prefix hashes, the KV cache cannot be reused.
        """
        builder = DeterministicPromptBuilder(sample_manual)
        
        agents = [
            "router",
            "technical_specialist",
            "compliance_auditor",
            "support_concierge",
        ]
        
        # Build prompts for each agent
        prompts = {
            agent: builder.build(agent, [], "What is the ATM limit?")
            for agent in agents
        }
        
        # Extract prefix hashes (up to <<< END OF MANUAL >>>)
        hashes = {}
        for agent, prompt in prompts.items():
            end_idx = prompt.find("<<< END OF MANUAL >>>")
            prefix = prompt[:end_idx] if end_idx > 0 else prompt[:1000]
            hashes[agent] = hash(prefix)
        
        # All hashes should be identical
        unique_hashes = set(hashes.values())
        assert len(unique_hashes) == 1, (
            f"Prefix hashes differ across agents! "
            f"This means the KV cache cannot be reused. "
            f"Hashes: {hashes}"
        )
    
    def test_whitespace_normalization(self):
        """Verify that whitespace is normalized consistently."""
        builder = DeterministicPromptBuilder
        
        # Test various whitespace scenarios
        text1 = "Hello world  \nLine 2  \r\nLine 3"
        text2 = "Hello world\nLine 2\nLine 3"
        
        normalized1 = builder._normalize(text1)
        normalized2 = builder._normalize(text2)
        
        assert normalized1 == normalized2, (
            f"Whitespace normalization failed!\n"
            f"Text1: {repr(normalized1)}\n"
            f"Text2: {repr(normalized2)}"
        )
    
    def test_bom_removal(self):
        """Verify that BOM characters are removed."""
        builder = DeterministicPromptBuilder
        
        text_with_bom = "\ufeffHello world"
        text_without_bom = "Hello world"
        
        normalized1 = builder._normalize(text_with_bom)
        normalized2 = builder._normalize(text_without_bom)
        
        assert normalized1 == normalized2


class TestCacheMetrics:
    """Tests for cache metrics tracking."""
    
    def test_cold_cache_baseline(self):
        """Verify cold cache baseline is set on first request."""
        metrics = CacheAwareMetrics()
        
        # First request should set baseline
        metrics.log_request_complete("router", "hash123", 10.0)
        
        assert metrics.cold_cache_ttft == 10.0
    
    def test_cache_hit_inference(self):
        """Verify cache hit is inferred from TTFT ratio."""
        metrics = CacheAwareMetrics()
        
        # First request (cold)
        result1 = metrics.log_request_complete("router", "hash123", 10.0)
        assert result1["is_cache_hit_inferred"] is False
        
        # Second request (should be cache hit if TTFT < 50% of baseline)
        result2 = metrics.log_request_complete("technical", "hash123", 3.0)
        assert result2["is_cache_hit_inferred"] is True
        
        # Third request (cache miss if TTFT >= 50% of baseline)
        result3 = metrics.log_request_complete("compliance", "hash123", 8.0)
        assert result3["is_cache_hit_inferred"] is False
    
    def test_prefix_mismatch_detection(self, sample_manual: str):
        """Verify prefix mismatch is detected."""
        metrics = CacheAwareMetrics()
        
        # First request sets expected hash
        prompt1 = f"System\n\n{sample_manual}\n\n<<< END OF MANUAL >>>\n\nAgent 1"
        result1 = metrics.log_request_start("router", prompt1)
        assert result1["cache_aligned"] is True
        
        # Same prefix should be aligned
        prompt2 = f"System\n\n{sample_manual}\n\n<<< END OF MANUAL >>>\n\nAgent 2"
        result2 = metrics.log_request_start("technical", prompt2)
        assert result2["cache_aligned"] is True
        
        # Different prefix should NOT be aligned
        prompt3 = f"Different System\n\n{sample_manual}\n\n<<< END OF MANUAL >>>\n\nAgent 3"
        result3 = metrics.log_request_start("compliance", prompt3)
        assert result3["cache_aligned"] is False
    
    def test_cache_report_generation(self):
        """Verify cache report is generated correctly."""
        metrics = CacheAwareMetrics()
        
        # Add some test data
        metrics.log_request_complete("router", "hash123", 10.0)
        metrics.log_request_complete("tech", "hash123", 3.0)  # Hit
        metrics.log_request_complete("comp", "hash123", 4.0)  # Hit
        metrics.log_request_complete("supp", "hash123", 2.0)  # Hit
        
        report = metrics.get_cache_report()
        
        assert report["total_requests"] == 4
        assert report["cold_cache_baseline_seconds"] == 10.0
        assert report["unique_prefix_hashes"] == 1
        assert report["prefix_alignment_ok"] is True
        assert report["inferred_cache_hit_rate"] == 1.0  # All non-first were hits
