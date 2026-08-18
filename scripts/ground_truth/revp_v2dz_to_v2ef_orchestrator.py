from __future__ import annotations

from pathlib import Path

from revp_v2dz_to_v2ef_common import parse_args, run_integrated


def main() -> None:
    args = parse_args()
    run_integrated(Path(args.repo_root), args.force)


if __name__ == "__main__":
    main()
