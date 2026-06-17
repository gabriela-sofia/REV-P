"""REV-P v2cx - real source availability verifier."""
from __future__ import annotations

import argparse
from pathlib import Path

from revp_v2cx_to_v2dd_common import add_repo_args, run_availability


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    add_repo_args(parser)
    args = parser.parse_args(argv)
    return run_availability(Path(args.repo_root), allow_network=args.allow_network, force=args.force, timeout=args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
