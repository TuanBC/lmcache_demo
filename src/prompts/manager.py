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

# Padding character - use space which tokenizes predictably
PADDING_CHAR = " "

# Lazy-loaded tokenizer for accurate token counting
_tokenizer = None
_tokenizer_name = None


def get_tokenizer(model_name: str = None):
    """Get tokenizer for accurate token counting (lazy loaded)."""
    global _tokenizer, _tokenizer_name

    if model_name is None:
        # Default to Qwen model from settings
        try:
            from src.config.settings import get_settings

            model_name = get_settings().vllm_model
        except Exception:
            model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"

    if _tokenizer is None or _tokenizer_name != model_name:
        try:
            from transformers import AutoTokenizer

            logger.info(f"[PROMPT] Loading tokenizer: {model_name}")
            _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            _tokenizer_name = model_name
        except Exception as e:
            logger.warning(f"[PROMPT] Failed to load tokenizer: {e}. Using estimation.")
            return None

    return _tokenizer


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

        # Store the normalized manual (no padding - server-side tokenizer may differ)
        self._padded_manual = self._manual

        # Calculate and log token count for debugging
        sample_prefix = self._build_sample_prefix(self._manual)
        total_tokens = self._count_tokens(sample_prefix)
        remainder = total_tokens % CHUNK_SIZE

        if remainder == 0:
            logger.info(
                f"[PROMPT] ✅ Prefix aligned: {total_tokens} tokens (chunk_size={CHUNK_SIZE})"
            )
        else:
            padding_needed = CHUNK_SIZE - remainder
            logger.info(
                f"[PROMPT] Prefix: {total_tokens} tokens "
                f"(remainder={remainder}, would need +{padding_needed} for alignment)"
            )

    def _count_tokens(self, text: str) -> int:
        """Count tokens using the actual tokenizer."""
        tokenizer = get_tokenizer()
        if tokenizer is not None:
            try:
                return len(tokenizer.encode(text))
            except Exception as e:
                logger.warning(f"[PROMPT] Tokenizer encode failed: {e}")
        # Fallback to estimation
        return len(text) // 4

    def _pad_to_chunk_boundary(self, text: str) -> str:
        """Pad text so the full prefix aligns to CHUNK_SIZE boundary.

        LMCache stores KV cache in chunks of CHUNK_SIZE tokens.
        Aligning the prefix to chunk boundaries improves cache hit rate.

        Uses linear search with predictable padding tokens.
        """
        # Build a sample full prefix to count total tokens accurately
        sample_prefix = self._build_sample_prefix(text)
        initial_tokens = self._count_tokens(sample_prefix)

        remainder = initial_tokens % CHUNK_SIZE
        if remainder == 0:
            logger.info(f"[PROMPT] ✅ Already aligned: {initial_tokens} tokens")
            return text

        # Calculate tokens needed to reach next chunk boundary
        tokens_needed = CHUNK_SIZE - remainder

        # Use a padding marker that tokenizes predictably (1-2 tokens per unit)
        # "." typically tokenizes to 1 token in most tokenizers
        padding_marker = " ."

        # Linear search - add padding until aligned
        padding = "\n<!-- padding:"
        for i in range(tokens_needed * 3):  # 3x buffer for safety
            padding += padding_marker

            padded_text = text + padding + " -->"
            sample_prefix = self._build_sample_prefix(padded_text)
            new_tokens = self._count_tokens(sample_prefix)
            new_remainder = new_tokens % CHUNK_SIZE

            if new_remainder == 0:
                logger.info(
                    f"[PROMPT] ✅ Padding successful: {initial_tokens} -> {new_tokens} tokens "
                    f"(aligned to {CHUNK_SIZE})"
                )
                return padded_text

        # Best effort - return padded anyway
        final_text = text + padding + " -->"
        sample_prefix = self._build_sample_prefix(final_text)
        final_tokens = self._count_tokens(sample_prefix)
        final_remainder = final_tokens % CHUNK_SIZE

        logger.warning(
            f"[PROMPT] ⚠️ Could not achieve perfect alignment: {final_tokens} tokens "
            f"(remainder={final_remainder})"
        )
        return final_text

    def _build_sample_prefix(self, manual_text: str) -> str:
        """Build a sample prefix string to count tokens accurately.

        This simulates what the prompty template produces without needing
        to load and render the full template.
        """
        # Approximate the prompty template structure
        prefix = f"""You are an AI assistant for ABC Bank's internal operations.
You help bank employees query the Internal Operations & Compliance Manual.

IMPORTANT: Always cite specific sections from the manual when answering.
If information is not in the manual, clearly state this.

Below is the complete Internal Operations & Compliance Manual:

{manual_text}

<<< END OF MANUAL >>>"""
        return prefix

    @property
    def padded_manual(self) -> str:
        """Get chunk-aligned manual content."""
        return self._padded_manual

    @property
    def prefix_tokens_est(self) -> int:
        """Actual token count of the prefix using tokenizer."""
        return self._count_tokens(self._padded_manual)

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
