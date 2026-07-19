from __future__ import annotations

from pathlib import Path

from mv2_15_cronograma_common import OUTPUT_DIR, PROJECT_ROOT, build_artifact_registry, ensure_dirs


def main() -> None:
    ensure_dirs()
    rows = build_artifact_registry()
    print(f"[mv2_15_artifacts] artifacts={len(rows)}")
    print(f"[mv2_15_artifacts] output={(OUTPUT_DIR / 'mv2_15_artifact_registry.csv').relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
