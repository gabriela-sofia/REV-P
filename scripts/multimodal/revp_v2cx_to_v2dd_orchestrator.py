"""REV-P v2cx-v2dd integrated external evidence readiness orchestrator."""
from __future__ import annotations

import argparse
from pathlib import Path

from revp_v2cx_to_v2dd_common import add_repo_args, run_integrated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    add_repo_args(parser)
    args = parser.parse_args(argv)
    return run_integrated(
        Path(args.repo_root),
        offline=args.offline or not args.allow_network,
        allow_network=args.allow_network,
        allow_downloads=args.allow_downloads,
        force=args.force,
    )


if __name__ == "__main__":
    raise SystemExit(main())
