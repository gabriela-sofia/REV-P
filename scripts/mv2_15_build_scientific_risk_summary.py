from __future__ import annotations

from pathlib import Path

from mv2_15_cronograma_common import OUTPUT_DIR, PROJECT_ROOT, build_scientific_risk_summary, ensure_dirs


def main() -> None:
    ensure_dirs()
    rows = build_scientific_risk_summary()
    print(f"[mv2_15_risk] risks={len(rows)}")
    print(f"[mv2_15_risk] output={(OUTPUT_DIR / 'mv2_15_scientific_risk_summary.csv').relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
