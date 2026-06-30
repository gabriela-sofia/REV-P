"""REV-P v2cy - controlled product link discovery."""
from __future__ import annotations

import argparse
from pathlib import Path

from revp_v2cx_to_v2dd_common import add_repo_args, run_discovery


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    add_repo_args(parser)
    args = parser.parse_args(argv)
    return run_discovery(Path(args.repo_root), allow_network=args.allow_network, force=args.force, timeout=args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
