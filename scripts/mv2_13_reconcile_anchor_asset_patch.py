"""MV2-13 Tarefa 6 — Reconcilia anchor->asset->patch; produz a matriz de binding."""

from __future__ import annotations

from pathlib import Path

from mv2_13_sentinel_binding_common import (
    BINDING_COLUMNS,
    OUTPUT_DIR,
    PROJECT_ROOT,
    ensure_output_dir,
    reconcile,
    write_csv,
)


def main() -> None:
    ensure_output_dir()
    rows = reconcile()
    out = OUTPUT_DIR / "mv2_13_binding_matrix.csv"
    write_csv(out, BINDING_COLUMNS, rows)
    from collections import Counter
    dist = Counter(r["binding_strength"] for r in rows)
    print(f"[mv2_13_binding] bindings: {len(rows)} | {dict(dist)}")
    print(f"[mv2_13_binding] saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
