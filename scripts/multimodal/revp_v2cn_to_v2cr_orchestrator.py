"""REV-P v2cn-v2cr integrated external evidence orchestrator."""
from __future__ import annotations

import argparse
from pathlib import Path

from revp_v2cn_to_v2cr_common import run_integrated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--offline", action="store_true", help="Run without network or downloads; default behavior.")
    parser.add_argument("--allow-downloads", action="store_true", help="Allow downloads only from registered rows.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    return run_integrated(Path(args.repo_root), allow_downloads=args.allow_downloads, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
