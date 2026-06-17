"""REV-P v2cr - external evidence to patch candidate pairing."""
from __future__ import annotations

import argparse
from pathlib import Path

from revp_v2cn_to_v2cr_common import add_repo_force_args, run_patch_pairing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    add_repo_force_args(parser)
    args = parser.parse_args(argv)
    return run_patch_pairing(Path(args.repo_root), force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
