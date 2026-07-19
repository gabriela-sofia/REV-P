from __future__ import annotations

from pathlib import Path

from mv2_15_cronograma_common import OUTPUT_DIR, PROJECT_ROOT, build_engineering_backlog, ensure_dirs


def main() -> None:
    ensure_dirs()
    rows = build_engineering_backlog()
    print(f"[mv2_15_backlog] items={len(rows)}")
    print(f"[mv2_15_backlog] output={(OUTPUT_DIR / 'mv2_15_engineering_backlog.csv').relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
