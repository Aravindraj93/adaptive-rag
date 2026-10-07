"""
adaptive-rag CLI dispatcher.

Provides the top-level `adaptive-rag` command that dispatches to sub-commands:
    adaptive-rag doctor  -- diagnose a corpus
    adaptive-rag evaluate -- run benchmarks (alias for adaptive-rag-evaluate)
    adaptive-rag version -- show version info
"""

from __future__ import annotations

import sys


USAGE = """\
adaptive-rag — CPU-first RAG retrieval for everyone

Usage:
    adaptive-rag <command> [options]

Commands:
    doctor      Diagnose a RAG corpus and get a health score
    evaluate    Run retrieval benchmarks
    version     Show version information

Examples:
    adaptive-rag doctor --corpus ./my_docs/
    adaptive-rag doctor --corpus ./docs/ --queries ./questions.json --json
    adaptive-rag evaluate --repetitions 5 --output report.json
    adaptive-rag version

Run `adaptive-rag <command> --help` for command-specific help.
"""


def main(argv: list[str] | None = None) -> None:
    args = argv if argv is not None else sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(USAGE)
        sys.exit(0)

    command = args[0]
    rest = args[1:]

    if command == "doctor":
        from adaptive_rag.doctor import doctor_command
        sys.exit(doctor_command(rest))

    elif command in ("evaluate", "eval"):
        from adaptive_rag.benchmark import main as eval_main
        sys.argv = ["adaptive-rag-evaluate"] + rest
        eval_main()

    elif command == "version":
        try:
            from importlib.metadata import version
            ver = version("adaptive-rag")
        except Exception:
            ver = "unknown"
        print(f"adaptive-rag {ver}")
        print("Python", sys.version)
        print("Platform:", sys.platform)
        # Check optional deps
        for dep in ["numpy", "pypdf", "pillow", "langchain_core", "llama_index"]:
            try:
                __import__(dep)
                print(f"  ✓ {dep}")
            except ImportError:
                print(f"  ✗ {dep} (optional)")
        sys.exit(0)

    else:
        print(f"Error: unknown command '{command}'\n", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
