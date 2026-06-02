"""Command-line interface.

    ccl methods                          list registered compressors + availability
    ccl compress --method stopwords "…"  compress one text, show token stats
    ccl bench [--embedding] [--llm-eval] [--methods a,b]
"""

from __future__ import annotations

import argparse
import sys

from . import benchmark, metrics, report
from .compressors import available, get
from .tokenization import count_tokens, tokenizer_name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ccl", description="Context-compression benchmark lab.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("methods", help="List compressors.")

    pc = sub.add_parser("compress", help="Compress a single string.")
    pc.add_argument("text")
    pc.add_argument("--method", "-m", required=True)

    pb = sub.add_parser("bench", help="Run the full benchmark.")
    pb.add_argument("--embedding", action="store_true", help="Add embedding-cosine metric (embed extra).")
    pb.add_argument("--llm-eval", action="store_true", help="Add Claude task-accuracy (llm extra + key).")
    pb.add_argument("--runs", type=int, default=3, help="LLM eval passes to average (default 3).")
    pb.add_argument("--methods", help="Comma-separated subset of method names.")

    px = sub.add_parser("cost", help="Project $ savings at a given volume and token price.")
    px.add_argument("--volume", type=int, default=1_000_000, help="Input tokens per day (default 1,000,000).")
    px.add_argument("--price", type=float, default=5.0, help="USD per 1M input tokens (default 5.0).")

    args = parser.parse_args(argv)

    if args.command == "methods":
        print(f"Tokenizer: {tokenizer_name()}\n")
        for c in available():
            tag = "" if c.is_available() else f"  (unavailable, needs '{c.requires_extra}')"
            print(f"  {c.name:<20} {c.category}{tag}")
        return 0

    if args.command == "compress":
        comp = get(args.method)
        out = comp.compress(args.text)
        s = metrics.score(args.text, out)
        print(out)
        print(f"\n[{count_tokens(args.text)} → {count_tokens(out)} tokens | "
              f"reduction {s.token_reduction:.1f}% | preservation {s.preservation:.1f}%]")
        return 0

    if args.command == "bench":
        methods = args.methods.split(",") if args.methods else None
        results = benchmark.run(embedding=args.embedding, methods=methods)
        if args.llm_eval:
            print(f"Running LLM task-accuracy eval, {args.runs} run(s) (this calls the API)...",
                  file=sys.stderr)
            benchmark.add_task_accuracy(results, methods=methods, runs=args.runs)
        report.write_all(results)
        print(report.markdown_table(results))
        print("\nFigures written to reports/figures/, table to reports/results.md")
        return 0

    if args.command == "cost":
        from . import cost

        results = benchmark.run()
        savings = cost.project_all(results, daily_input_tokens=args.volume, price_per_million=args.price)
        print(cost.markdown_table(savings, daily_input_tokens=args.volume, price_per_million=args.price))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
