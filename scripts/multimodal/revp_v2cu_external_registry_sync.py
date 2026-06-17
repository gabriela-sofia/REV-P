"""REV-P v2cu - safe external acquisition registry sync."""
from __future__ import annotations

import argparse
from pathlib import Path

from revp_v2cs_to_v2cw_common import add_repo_args, run_registry_sync


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    add_repo_args(parser)
    args = parser.parse_args(argv)
    return run_registry_sync(Path(args.repo_root), force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
