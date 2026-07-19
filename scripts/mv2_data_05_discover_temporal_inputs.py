"""MV2-DATA-05 Tarefa 1 — Descobrir inputs temporais.

Inventaria as fontes possíveis de janela temporal: o template público DATA-04 (vazio) e os
candidatos a template preenchido em locais privados (git-ignored). Não lê segredo, não
promove nada.

Saída: outputs_public/mv2_data_temporal_window_intake/mv2_data_05_input_discovery.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_data_05_common import (
    DATA04_TEMPLATE,
    FILLED_TEMPLATE_RELPATHS,
    PROJECT_ROOT,
    PUBLIC_DIR,
    ensure_public_dir,
    read_csv_rows,
    write_csv,
)

COLUMNS = [
    "input_id",
    "path_or_ref",
    "location_type",
    "exists",
    "is_public_template",
    "is_filled_candidate",
    "rows",
    "read_status",
    "notes",
]


def _row_count(p: Path) -> int:
    try:
        return len(read_csv_rows(p))
    except OSError:
        return 0


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    n = 0

    # Template público DATA-04 (entrada vazia de fallback).
    n += 1
    pub_exists = DATA04_TEMPLATE.exists()
    rows.append({
        "input_id": f"MV2_DATA_05_IN_{n:02d}",
        "path_or_ref": DATA04_TEMPLATE.relative_to(PROJECT_ROOT).as_posix(),
        "location_type": "PUBLIC_TEMPLATE",
        "exists": "true" if pub_exists else "false",
        "is_public_template": "true",
        "is_filled_candidate": "false",
        "rows": _row_count(DATA04_TEMPLATE) if pub_exists else 0,
        "read_status": "READ_OK" if pub_exists else "MISSING",
        "notes": "template DATA-04 (campos temporais vazios; fallback sem promocao)",
    })

    # Candidatos privados a template preenchido.
    for rel in FILLED_TEMPLATE_RELPATHS:
        n += 1
        p = PROJECT_ROOT / rel
        exists = p.exists()
        rows.append({
            "input_id": f"MV2_DATA_05_IN_{n:02d}",
            "path_or_ref": rel,
            "location_type": "PRIVATE_FILLED_CANDIDATE",
            "exists": "true" if exists else "false",
            "is_public_template": "false",
            "is_filled_candidate": "true" if exists else "false",
            "rows": _row_count(p) if exists else 0,
            "read_status": "READ_OK" if exists else "NOT_PRESENT",
            "notes": "git-ignored; humano deposita o template preenchido aqui",
        })
    return rows


def main() -> None:
    ensure_public_dir()
    rows = build_rows()
    out = PUBLIC_DIR / "mv2_data_05_input_discovery.csv"
    write_csv(out, COLUMNS, rows)
    filled = sum(1 for r in rows if r["is_filled_candidate"] == "true")
    print(
        f"[mv2_data_05_discover] {len(rows)} inputs | filled_candidates={filled} | "
        f"saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
