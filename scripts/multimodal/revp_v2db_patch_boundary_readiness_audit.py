"""REV-P v2db - patch boundary readiness audit."""
from __future__ import annotations

import argparse
from pathlib import Path

from revp_v2cx_to_v2dd_common import add_repo_args, run_boundary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    add_repo_args(parser)
    args = parser.parse_args(argv)
    return run_boundary(Path(args.repo_root), force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
