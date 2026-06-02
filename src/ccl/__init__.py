"""context-compression-lab (ccl) — a benchmark lab for token minimization.

Run many text-compression methods over a corpus and measure, as percentages,
how much each reduces tokens versus how much context it preserves.
"""

from . import benchmark, corpus, metrics
from .compressors import available, get, register
from .tokenization import count_tokens, tokenizer_name

__version__ = "0.1.0"
__all__ = [
    "benchmark",
    "corpus",
    "metrics",
    "available",
    "get",
    "register",
    "count_tokens",
    "tokenizer_name",
]
