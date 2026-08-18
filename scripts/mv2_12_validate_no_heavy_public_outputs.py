"""MV2-12 — Validador: outputs_public/mv2_data_readiness só pode conter arquivos leves.

Garante que o diretório público do MV2-12 contém apenas manifestos, índices,
relatórios, checksums, tabelas leves e README — e nenhum dado bruto pesado
(GeoTIFF/JP2/SAFE/NPZ/NPY/ZIP/GPKG/SHP) nem renderização binária pesada.

Também verifica que a tabela de download readiness não marca
publish_to_outputs_public=true para nenhuma fonte bruta.

Sai com código !=0 se encontrar violação (uso em CI/regressão).
"""

from __future__ import annotations

import sys

from mv2_12_data_readiness_common import (
    HEAVY_RAW_EXTENSIONS,
    OUTPUT_DIR,
    PROJECT_ROOT,
    read_csv_rows,
)

# Teto de tamanho para qualquer arquivo do diretório público (índice/tabela leve).
MAX_FILE_BYTES = 2_000_000  # 2 MB
ALLOWED_EXTENSIONS = {".md", ".csv", ".json", ".txt", ".sha256"}


def find_violations() -> list[str]:
    violations: list[str] = []
    if not OUTPUT_DIR.exists():
        return [f"diretório ausente: {OUTPUT_DIR}"]

    for path in sorted(OUTPUT_DIR.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        ext = path.suffix.lower()
        if ext in HEAVY_RAW_EXTENSIONS:
            violations.append(f"bruto pesado proibido em outputs_public: {rel}")
            continue
        if ext not in ALLOWED_EXTENSIONS:
            violations.append(f"extensão não-permitida no público: {rel} ({ext})")
        try:
            size = path.stat().st_size
        except OSError:
            size = -1
        if size > MAX_FILE_BYTES:
            violations.append(f"arquivo acima de {MAX_FILE_BYTES} bytes: {rel} ({size})")

    # Cross-check: download readiness não pode publicar bruto.
    readiness = OUTPUT_DIR / "mv2_12_download_readiness.csv"
    for row in read_csv_rows(readiness):
        if str(row.get("publish_to_outputs_public", "")).strip().lower() == "true":
            violations.append(
                f"download_readiness marca publish_to_outputs_public=true: {row.get('source_name')}"
            )
        target = str(row.get("download_target_dir", ""))
        if target.startswith("outputs_public"):
            violations.append(
                f"download_target_dir aponta para outputs_public: {row.get('source_name')}"
            )
    return violations


def main() -> int:
    violations = find_violations()
    if violations:
        print("[mv2_12_validate] VIOLACOES ENCONTRADAS:")
        for v in violations:
            print(f"  - {v}")
        return 1
    n = sum(1 for p in OUTPUT_DIR.rglob("*") if p.is_file())
    print(f"[mv2_12_validate] OK — {n} arquivos leves, 0 bruto pesado em outputs_public/mv2_data_readiness")
    return 0


if __name__ == "__main__":
    sys.exit(main())
