from __future__ import annotations

from pathlib import Path

from mv2_15_cronograma_common import OUTPUT_DIR, PROJECT_ROOT, build_summary, ensure_dirs, write_reports


def main() -> None:
    ensure_dirs()
    summary = build_summary()
    write_reports()
    print(f"[mv2_15_summary] allowed={summary['allowed_programming_actions_count']} blocked={summary['blocked_actions_count']}")
    print(f"[mv2_15_summary] output={(OUTPUT_DIR / 'mv2_15_cronograma_summary.json').relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
