"""MV2-DATA-06 temporal window human-input pack.

Builds an auditable, fill-by-human template for temporal windows. It never
infers dates from bbox, city, or "probable event": every window must come from a
traceable source. With no filled input the stage stays BLOCKED_NO_FILLED_TEMPLATE
and no metadata probe is opened.

This script only writes a blank template plus instructions; it does not call any
provider, does not download, and does not create raster or crop data.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs_public" / "mv2_data_temporal_window_human_pack"

PROMOTION_PATH = (
    PROJECT_ROOT
    / "outputs_public"
    / "mv2_data_temporal_window_promotion"
    / "mv2_data_06_temporal_window_promotion.csv"
)
SEED_PATH = (
    PROJECT_ROOT
    / "outputs_public"
    / "mv2_pre_unification_seed"
    / "revp_temporal_window_seed_10.csv"
)
INTAKE_PATH = (
    PROJECT_ROOT
    / "outputs_public"
    / "mv2_data_temporal_window_intake"
    / "mv2_data_05_temporal_window_correction_template.csv"
)

TEMPLATE_FIELDS = [
    "target_rank",
    "patch_id",
    "asset_id",
    "city",
    "bbox",
    "crs",
    "temporal_window_start",
    "temporal_window_end",
    "temporal_window_source",
    "source_ref",
    "source_type",
    "evidence_strength",
    "review_status",
    "reviewer_notes",
    "blocked_reason",
]

ACCEPTED_SOURCE_TYPES = [
    "BOLETIM_OFICIAL",
    "RELATORIO_OFICIAL",
    "CEMADEN",
    "DEFESA_CIVIL",
    "CEMS_COPERNICUS_EMS",
    "SGB_CPRM",
    "ANA",
    "PUBLICACAO_CIENTIFICA",
    "REGISTRO_INTERNO_AUDITAVEL",
]

REJECTED_SOURCE_TYPES = [
    "MEMORIA_HUMANA_SEM_REGISTRO",
    "ESTIMATIVA_VISUAL",
    "DATA_INVENTADA",
    "JANELA_ABERTA_SEM_JUSTIFICATIVA",
    "BBOX_ONLY_SEARCH",
]

DEFAULT_BLOCKED_REASON = "BLOCKED_NO_FILLED_TEMPLATE"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _seed_index(seed_rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {
        (row.get("patch_id", ""), row.get("asset_id", "")): row
        for row in seed_rows
    }


def build_template_rows(
    promotion_rows: list[dict[str, str]],
    seed_rows: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """One blank template row per known target. Dates are never filled here."""
    seed_by_key = _seed_index(seed_rows or [])
    rows: list[dict[str, Any]] = []
    for idx, target in enumerate(promotion_rows, 1):
        patch_id = target.get("patch_id", "")
        asset_id = target.get("asset_id", "")
        seed = seed_by_key.get((patch_id, asset_id), {})
        rows.append(
            {
                "target_rank": idx,
                "patch_id": patch_id,
                "asset_id": asset_id,
                "city": seed.get("city", ""),
                "bbox": seed.get("bbox", ""),
                "crs": seed.get("crs", ""),
                "temporal_window_start": "",
                "temporal_window_end": "",
                "temporal_window_source": "",
                "source_ref": "",
                "source_type": "",
                "evidence_strength": "",
                "review_status": "PENDING_HUMAN_FILL",
                "reviewer_notes": "",
                "blocked_reason": DEFAULT_BLOCKED_REASON,
            }
        )
    return rows


def assert_no_autofilled_dates(rows: list[dict[str, Any]]) -> None:
    """Guardrail: the generated template must not carry any inferred window."""
    for row in rows:
        start = str(row.get("temporal_window_start", "")).strip()
        end = str(row.get("temporal_window_end", "")).strip()
        if start or end:
            raise ValueError(
                f"template row {row.get('patch_id')} carries an auto-filled window; "
                "dates must come only from a human-filled traceable source"
            )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "stage": "DATA-06",
        "artifact": "temporal_window_human_pack",
        "targets_in_template": len(rows),
        "filled_windows": 0,
        "promotion_status": DEFAULT_BLOCKED_REASON,
        "auto_filled_dates": 0,
        "api_calls": 0,
        "downloads": 0,
        "rasters": 0,
        "crops": 0,
    }


def write_schema() -> None:
    write_json(
        PROJECT_ROOT / "datasets" / "schemas" / "schema_mv2_data_06_temporal_window_human_pack.json",
        {
            "schema_id": "schema_mv2_data_06_temporal_window_human_pack",
            "required_fields": [
                "patch_id",
                "asset_id",
                "temporal_window_start",
                "temporal_window_end",
                "temporal_window_source",
                "source_ref",
                "source_type",
                "review_status",
            ],
            "template_fields": TEMPLATE_FIELDS,
            "accepted_source_types": ACCEPTED_SOURCE_TYPES,
            "rejected_source_types": REJECTED_SOURCE_TYPES,
            "rules": [
                "datas nunca preenchidas automaticamente",
                "sem inferencia por bbox",
                "sem inferencia por cidade",
                "sem evento provavel sem fonte",
                "janela so com fonte rastreavel",
                "se vazio mantem BLOCKED_NO_FILLED_TEMPLATE",
            ],
            "default_blocked_reason": DEFAULT_BLOCKED_REASON,
        },
    )


def _instructions_md() -> str:
    accepted = "\n".join(f"- {item}" for item in ACCEPTED_SOURCE_TYPES)
    rejected = "\n".join(f"- {item}" for item in REJECTED_SOURCE_TYPES)
    return f"""# DATA-06 - Instrucoes de preenchimento da janela temporal

Este pacote e de entrada humana rastreavel. Nenhuma data e preenchida
automaticamente pelo script.

## Como preencher
1. Abra `mv2_data_06_temporal_window_human_template.csv`.
2. Para cada `patch_id`/`asset_id`, localize a janela temporal **apenas** em uma
   fonte oficial ou registro auditavel.
3. Preencha `temporal_window_start` e `temporal_window_end` no formato
   ISO `AAAA-MM-DD`.
4. Preencha `temporal_window_source` (nome da fonte), `source_ref` (link, DOI,
   protocolo ou caminho do registro), `source_type` e `evidence_strength`.
5. So mude `review_status` para `APPROVED`/`REVIEWED`/`CONFIRMED` depois de
   conferir a fonte. Enquanto estiver vazio, mantenha `PENDING_HUMAN_FILL`.
6. Salve o arquivo preenchido em uma pasta local (nao versionada), por exemplo
   `local_only/mv2_data_temporal_window/mv2_data_06_temporal_window_filled.csv`,
   e rode `scripts/mv2_data_06_temporal_window_promotion.py --filled-template <caminho>`.

## Regras inviolaveis
- Datas nunca sao inferidas por bbox.
- Datas nunca sao inferidas por cidade.
- "Evento provavel" sem fonte nao vale.
- Janela so pode existir com fonte rastreavel.
- Sem preenchimento, o estagio permanece `{DEFAULT_BLOCKED_REASON}`.

## Fontes aceitas
{accepted}

## Fontes nao aceitas
{rejected}
"""


def _source_policy_md() -> str:
    return """# DATA-06 - Politica de fontes da janela temporal

## Aceitas
- Boletim ou relatorio oficial (orgao publico).
- CEMADEN.
- Defesa Civil (estadual ou municipal).
- CEMS / Copernicus EMS.
- SGB / CPRM.
- ANA.
- Publicacao cientifica com referencia verificavel.
- Registro interno auditavel, desde que com referencia rastreavel.

## Nao aceitas
- Memoria humana sem registro.
- Estimativa visual.
- Data inventada.
- Janela aberta demais sem justificativa.
- Busca apenas por bbox (bbox-only search).

## Principio
Toda janela temporal precisa de `source_ref` rastreavel. Sem isso, o registro
fica bloqueado e nao entra na sondagem de metadados.
"""


def _examples_md() -> str:
    return """# DATA-06 - Exemplos (ilustrativos, nao preencher o template com estes valores)

Os exemplos abaixo mostram o formato esperado. Eles NAO devem ser copiados para
o template real: cada linha precisa da sua propria fonte verificada.

## Exemplo valido (promove)
- temporal_window_start: 2022-05-24
- temporal_window_end: 2022-05-31
- temporal_window_source: Defesa Civil PE - boletim de ocorrencia
- source_ref: protocolo-interno-REC-2022-05-24 (registro auditavel)
- source_type: DEFESA_CIVIL
- evidence_strength: STRONG
- review_status: APPROVED

## Exemplo bloqueado (sem fonte)
- temporal_window_start: 2022-05-24
- temporal_window_end: 2022-05-31
- temporal_window_source: (vazio)
- source_ref: (vazio)
- review_status: PENDING_HUMAN_FILL
- Resultado: BLOCKED_NO_SOURCE.

## Exemplo recusado (inferencia proibida)
- Preencher data so porque "choveu muito naquele mes" sem boletim: proibido.
- Inferir janela a partir do bbox: proibido.
"""


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser().parse_args(argv)

    promotion_rows = read_csv(PROMOTION_PATH)
    seed_rows = read_csv(SEED_PATH)
    # intake is read only to confirm presence; it never auto-fills windows.
    intake_present = INTAKE_PATH.exists()

    rows = build_template_rows(promotion_rows, seed_rows)
    assert_no_autofilled_dates(rows)

    write_schema()
    write_csv(OUT_DIR / "mv2_data_06_temporal_window_human_template.csv", TEMPLATE_FIELDS, rows)
    write_text(OUT_DIR / "mv2_data_06_temporal_window_human_instructions.md", _instructions_md())
    write_text(OUT_DIR / "mv2_data_06_temporal_window_source_policy.md", _source_policy_md())
    write_text(OUT_DIR / "mv2_data_06_temporal_window_examples.md", _examples_md())

    summary = summarize(rows)
    summary["promotion_template_found"] = PROMOTION_PATH.exists()
    summary["seed_found"] = SEED_PATH.exists()
    summary["intake_found"] = intake_present
    write_json(OUT_DIR / "mv2_data_06_temporal_window_human_pack_summary.json", summary)
    write_text(
        OUT_DIR / "commands.txt",
        "python scripts/mv2_data_06_prepare_temporal_window_human_pack.py\n"
        "# depois de preencher localmente:\n"
        "python scripts/mv2_data_06_temporal_window_promotion.py --filled-template <caminho_local>",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
