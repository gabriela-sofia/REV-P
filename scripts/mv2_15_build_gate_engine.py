from __future__ import annotations

from pathlib import Path

from mv2_15_cronograma_common import OUTPUT_DIR, PROJECT_ROOT, build_gate_engine, ensure_dirs


def main() -> None:
    ensure_dirs()
    rows = build_gate_engine()
    print(f"[mv2_15_gates] gates={len(rows)}")
    print(f"[mv2_15_gates] output={(OUTPUT_DIR / 'mv2_15_gate_engine.csv').relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
