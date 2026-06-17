"""REV-P v2cv - external product discovery checklist."""
from __future__ import annotations

import argparse
from pathlib import Path

from revp_v2cs_to_v2cw_common import add_repo_args, run_checklist


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    add_repo_args(parser)
    args = parser.parse_args(argv)
    return run_checklist(Path(args.repo_root), force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
