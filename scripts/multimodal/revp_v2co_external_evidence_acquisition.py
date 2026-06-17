"""REV-P v2co - controlled external evidence acquisition."""
from __future__ import annotations

import argparse
from pathlib import Path

from revp_v2cn_to_v2cr_common import add_repo_force_args, run_acquisition


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    add_repo_force_args(parser)
    parser.add_argument("--offline", action="store_true", help="Use only local registry metadata; this is the default.")
    parser.add_argument("--allow-downloads", action="store_true", help="Download only allowlisted registry URLs.")
    args = parser.parse_args(argv)
    return run_acquisition(Path(args.repo_root), allow_downloads=args.allow_downloads, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
