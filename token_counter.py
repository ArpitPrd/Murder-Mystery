"""
Shared token counter for aggregating OpenAI-compatible API usage across
all LLM calls made during a single investigation run.

Passed by reference to Interrogator and ReportGenerator so every call
site accumulates into one global record for the computational cost report.
"""

import threading


class TokenCounter:
    """
    Thread-safe accumulator for prompt and completion token counts.

    Call add(response.usage) after every chat-completion call.
    Read total_tokens, prompt_tokens, completion_tokens at investigation end.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_calls: int = 0

    def add(self, usage) -> None:
        """
        Record token usage from one API response usage object.

        Args:
            usage: The .usage attribute of a chat-completion response.
                   Expected to have prompt_tokens and completion_tokens.
                   Silently ignored if None.
        """
        if usage is None:
            return
        with self._lock:
            self.prompt_tokens += getattr(usage, "prompt_tokens", 0)
            self.completion_tokens += getattr(usage, "completion_tokens", 0)
            self.total_calls += 1

    @property
    def total_tokens(self) -> int:
        """Sum of prompt and completion tokens across all calls."""
        return self.prompt_tokens + self.completion_tokens

    def summary(self) -> dict:
        """Return a plain dict suitable for logging and report sections."""
        return {
            "total_calls": self.total_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }
