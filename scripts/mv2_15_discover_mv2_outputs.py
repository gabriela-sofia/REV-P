from __future__ import annotations

from pathlib import Path

from mv2_15_cronograma_common import OUTPUT_DIR, PROJECT_ROOT, discover_mv2_outputs, ensure_dirs


def main() -> None:
    ensure_dirs()
    rows = discover_mv2_outputs()
    found = sum(1 for row in rows if row["exists"] == "true")
    print(f"[mv2_15_discovery] stages_found={found}/{len(rows)}")
    print(f"[mv2_15_discovery] output={(OUTPUT_DIR / 'mv2_15_mv2_output_discovery.csv').relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
