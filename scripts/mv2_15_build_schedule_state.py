from __future__ import annotations

from pathlib import Path

from mv2_15_cronograma_common import OUTPUT_DIR, PROJECT_ROOT, build_schema, build_schedule_state, ensure_dirs


def main() -> None:
    ensure_dirs()
    build_schema()
    rows = build_schedule_state()
    print(f"[mv2_15_schedule] days={len(rows)}")
    print(f"[mv2_15_schedule] output={(OUTPUT_DIR / 'mv2_15_schedule_state.csv').relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
