"""Optional compressors that depend on extra packages or an API.

These register so they appear in `ccl methods`, but report themselves
unavailable (and the benchmark skips them) unless their dependency is present.
This keeps the core lab dependency-free while letting users plug in a SOTA
learned compressor or an LLM rewriter for comparison.
"""

from __future__ import annotations

import os

from .base import Compressor, register


@register
class LLMLingua2(Compressor):
    """LLMLingua-2 (Microsoft): a BERT-level token-classification compressor
    trained on GPT-4-distilled data. Requires `pip install '.[llmlingua]'`."""

    name = "llmlingua2"
    category = "filtering/learned (external)"
    requires_extra = "llmlingua"
    rate = 0.5

    def is_available(self) -> bool:
        try:
            import llmlingua  # noqa: F401

            return True
        except Exception:
            return False

    def compress(self, text: str) -> str:
        from llmlingua import PromptCompressor

        comp = getattr(self, "_comp", None)
        if comp is None:
            comp = PromptCompressor(
                model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                use_llmlingua2=True,
            )
            self._comp = comp
        return comp.compress_prompt(text, rate=self.rate)["compressed_prompt"]


@register
class LLMAbstractive(Compressor):
    """Abstractive compression: ask Claude to rewrite the text more tersely
    while preserving every fact. Requires `pip install '.[llm]'` + ANTHROPIC_API_KEY."""

    name = "llm-abstractive"
    category = "paraphrasing/LLM (external)"
    requires_extra = "llm"

    def is_available(self) -> bool:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return False
        try:
            import anthropic  # noqa: F401

            return True
        except Exception:
            return False

    def compress(self, text: str) -> str:
        import anthropic

        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=1024,
            system=[{
                "type": "text",
                "text": (
                    "Rewrite the user's text as tersely as possible while preserving "
                    "every fact, name, number, and entity. Drop filler and redundancy. "
                    "Return only the compressed text."
                ),
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": text}],
        )
        return "".join(b.text for b in resp.content if b.type == "text").strip()
