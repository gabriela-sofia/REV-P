from __future__ import annotations

from pathlib import Path

from mv2_15_cronograma_common import OUTPUT_DIR, PROJECT_ROOT, build_allowed_programming_actions, ensure_dirs


def main() -> None:
    ensure_dirs()
    rows = build_allowed_programming_actions()
    allowed = sum(1 for row in rows if row["allowed_now"] == "true")
    print(f"[mv2_15_actions] allowed={allowed} blocked={len(rows) - allowed}")
    print(f"[mv2_15_actions] output={(OUTPUT_DIR / 'mv2_15_allowed_programming_actions.csv').relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
