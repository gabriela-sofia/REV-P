"""MV2-DATA-03 Tarefa 9 — Matriz de impacto no corpus.

Explica formalmente que o canary OFICIAL valida apenas o mecanismo técnico (auth, probe,
caminho privado, checksum, bandas/SCL) e NÃO desbloqueia os 128 targets, nem o Dia 10, nem
o sandbox.

Saída: outputs_public/mv2_data_live_api_canary/mv2_data_03_corpus_impact_matrix.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_data_03_common import PROJECT_ROOT, PUBLIC_DIR, ensure_public_dir, write_csv

COLUMNS = [
    "impact_id",
    "component",
    "canary_validates",
    "affects_128_targets",
    "affects_day10",
    "affects_sandbox",
    "allowed_next_step",
    "forbidden_interpretation",
    "notes",
]

# (component, canary_validates, allowed_next_step, forbidden_interpretation)
ROWS = [
    ("API auth validated", "true",
     "reusar credencial via env var para probes de metadados",
     "tratar auth ok como lineage do corpus"),
    ("metadata probe validated", "true",
     "consultar metadados Sentinel dos candidates oficiais",
     "preencher scene_id dos 128 targets sem recuperacao real"),
    ("raster private path validated", "true",
     "baixar 1 raster canary em diretorio privado",
     "publicar raster ou baixar os 128 targets"),
    ("checksum validation", "true",
     "verificar integridade do raster privado por SHA256",
     "assumir corpus baixado/integro"),
    ("band/SCL validation", "true",
     "confirmar bandas B02..B12 + SCL no raster canary",
     "gerar feature espectral do corpus"),
    ("corpus lineage still missing", "false",
     "recuperar scene_id/datetime/tile/cloud dos 128 via DATA-01/02",
     "considerar lineage do corpus resolvido pelo canary"),
    ("Dia 10 still blocked", "false",
     "manter Dia 10 bloqueado ate lineage forte + raster do corpus validado",
     "desbloquear Dia 10 com base no canary oficial"),
    ("sandbox still blocked", "false",
     "manter sandbox bloqueado",
     "abrir sandbox/treino com base no canary"),
]


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i, (component, validates, nxt, forbidden) in enumerate(ROWS, start=1):
        corpus_affected = component in {
            "corpus lineage still missing", "Dia 10 still blocked", "sandbox still blocked"
        }
        rows.append(
            {
                "impact_id": f"MV2_DATA_03_IMP_{i:02d}",
                "component": component,
                "canary_validates": validates,
                # canary NUNCA afeta corpus/Dia10/sandbox
                "affects_128_targets": "false",
                "affects_day10": "false",
                "affects_sandbox": "false",
                "allowed_next_step": nxt,
                "forbidden_interpretation": forbidden,
                "notes": "canary oficial = prova tecnica, nao baseline do corpus"
                + (";gargalo do corpus permanece" if corpus_affected else ""),
            }
        )
    return rows


def main() -> None:
    ensure_public_dir()
    rows = build_rows()
    out = PUBLIC_DIR / "mv2_data_03_corpus_impact_matrix.csv"
    write_csv(out, COLUMNS, rows)
    print(
        f"[mv2_data_03_impact] {len(rows)} linhas | "
        f"saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
