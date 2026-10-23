"""Dispatch to independent tool modules without probing anything at startup."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence

from . import __version__


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(
            "usage: agent-workbench {environment,models} [arguments]\n\n"
            "  environment   Local environment and peripheral diagnostics\n"
            "  models        Explicit auxiliary-model probes and invocations\n\n"
            "Use '<command> --help' for options. No probes run for this help."
        )
        return 0
    if args == ["--version"]:
        print(__version__)
        return 0
    command = args.pop(0)
    if command not in {"environment", "models"}:
        print("Unknown command. Use 'agent-workbench --help'.", file=sys.stderr)
        return 2
    module = importlib.import_module(f"agent_workbench.{command}")
    return module.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
