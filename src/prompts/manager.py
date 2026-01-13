"""
=============================================================================
Prompt Manager - Prompty-based Prompt Construction
=============================================================================

This module manages prompts using the .prompty format for better
observability and maintainability.

AGENTS.MD COMPLIANT:
--------------------
- Uses .prompty files for all agent templates
- Uses prompty.load() and prompty.prepare()
- Maintains deterministic manual content injection for KV cache hits
=============================================================================
"""

import hashlib
import logging
from functools import lru_cache
from pathlib import Path

import prompty

logger = logging.getLogger(__name__)

# =============================================================================
# LMCache Optimization Constants
# =============================================================================
# These values MUST match the LMCache server configuration.
# Chunk alignment ensures prompts end on cache block boundaries.
CHUNK_SIZE = 256  # tokens per cache chunk (matches lmcache-config.yaml)

# TURN_BOUNDARY must match blend_special_str in lmcache-config.yaml
# This enables CacheBlend to efficiently reuse KV cache in multi-turn conversations
TURN_BOUNDARY = "\n<<< TURN >>>\n"  # Aligned with blend_special_str

# The intro text that appears BEFORE the manual in all prompty templates
# This is included in the cacheable prefix, so we must account for it in padding
INTRO_TEXT_EST_TOKENS = 30  # ~120 chars of intro ("You are an AI assistant...")
END_MARKER = "<<< END OF MANUAL >>>"
END_MARKER_TOKENS = 6  # ~24 chars


class DeterministicPromptBuilder:
    """
    Ensures prompt text is exactly reproducible for cache hits.
    Uses the 'prompty' library to load templates from .prompty files.
    """

    def __init__(self, manual_content: str):
        """
        Initialize with the operations manual.

        The manual is normalized ONCE at init, then reused for all prompts.
        This ensures the prefix is byte-identical across all requests.
        """
        self._manual = self._normalize(manual_content)
        self._manual_hash = hashlib.sha256(self._manual.encode()).hexdigest()[:16]

        # Determine prompts directory path
        self._prompts_dir = Path(__file__).parent.absolute()

        logger.info(
            f"[PROMPT] Initialized with manual: "
            f"len={len(self._manual)} chars, hash={self._manual_hash}"
        )

        # Use normalized manual directly - NO PADDING
        # Padding was removed because client-side token estimation (len//4) is inexact
        # and likely causes misalignment on the server side.
        # Natural stability of the prefix is better for cache hits.
        self._padded_manual = self._manual
        token_est = self._estimate_tokens(self._manual) + INTRO_TEXT_EST_TOKENS + END_MARKER_TOKENS
        logger.info(f"[PROMPT] Manual loaded (no padding): ~{token_est} tokens total prefix")

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from character count.

        Uses 4 chars ≈ 1 token heuristic (conservative for English).
        This is compatible with most tokenizers including Qwen.
        """
        return len(text) // 4

    # _pad_to_chunk_boundary_full_prefix REMOVED - Inexact padding hurts cache hits

    @property
    def padded_manual(self) -> str:
        """Get manual content (no longer padded, name kept for compatibility)."""
        return self._padded_manual

    @property
    def prefix_tokens_est(self) -> int:
        """Estimated token count of the prefix."""
        return self._estimate_tokens(self._padded_manual)

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for consistent hashing and cache hits."""
        if not text:
            return ""

        # If prompty returned a list of messages, join them
        if isinstance(text, list):
            # Convert list of messages to a single string
            parts = []
            for msg in text:
                role = msg.get("role", "unknown").upper()
                content = msg.get("content", "").strip()
                if content:
                    parts.append(f"{role}: {content}")
            return "\n\n".join(parts)

        if not isinstance(text, str):
            text = str(text)

        # Remove BOM if present
        if text.startswith("\ufeff"):
            text = text[1:]

        # Strip trailing whitespace per line
        lines = [line.rstrip() for line in text.splitlines()]

        # Use Unix line endings
        return "\n".join(lines)

    @lru_cache(maxsize=10)
    def _get_prompty(self, agent_name: str):
        """Load and cache a .prompty file."""
        prompty_path = self._prompts_dir / f"{agent_name}.prompty"
        if not prompty_path.exists():
            logger.warning(f"[PROMPT] Prompty file not found: {prompty_path}")
            return None

        return prompty.load(str(prompty_path))

    def build(self, agent_name: str, history: list, query: str) -> str:
        """
        Build prompt by loading a .prompty file and hydrating it.
        """
        p = self._get_prompty(agent_name)

        if p is None:
            # Fallback for missing prompty files
            return f"Agent: {agent_name}\nQuery: {query}\n\nManual snippet: {self._manual[:100]}..."

        # Prepare inputs for hydrate
        # Use padded manual for chunk alignment
        inputs = {
            "manual_content": self._padded_manual,  # Use padded version!
            "history": self._format_history(history),
            "query": query.strip(),
        }

        # Use prompty to hydrate the template
        # Note: Depending on the prompty configuration, this might return a string or a list of messages
        prompt_content = prompty.prepare(p, inputs)

        # Ensure normalization of the entire generated prompt
        return self._normalize(prompt_content)

    def _format_history(self, history: list) -> str:
        """Deterministic history formatting with turn boundaries.

        CACHE OPTIMIZATION:
        -------------------
        Uses TURN_BOUNDARY separator for multi-turn caching.
        This enables better cache reuse when CacheBlend is enabled server-side.
        """
        if not history:
            return "(No previous conversation)"

        formatted_turns = [
            f"{msg.get('role', 'unknown').upper()}: {msg.get('content', '').strip()}"
            for msg in history
        ]
        return TURN_BOUNDARY.join(formatted_turns)

    @property
    def manual_hash(self) -> str:
        return self._manual_hash


def load_manual(path: Path | str) -> str:
    """Load and normalize the operations manual."""
    path = Path(path)
    if not path.exists():
        return "# Placeholder Manual\nSection 1: No data found."

    content = path.read_text(encoding="utf-8")
    return content
