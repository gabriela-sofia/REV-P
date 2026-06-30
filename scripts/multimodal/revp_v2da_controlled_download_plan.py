"""REV-P v2da - controlled download plan."""
from __future__ import annotations

import argparse
from pathlib import Path

from revp_v2cx_to_v2dd_common import add_repo_args, run_download_plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    add_repo_args(parser)
    args = parser.parse_args(argv)
    return run_download_plan(Path(args.repo_root), allow_downloads=args.allow_downloads, force=args.force, max_size_mb=args.max_size_mb)


if __name__ == "__main__":
    raise SystemExit(main())
