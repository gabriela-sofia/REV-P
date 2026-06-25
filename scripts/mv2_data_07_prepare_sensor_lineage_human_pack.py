"""MV2-DATA-07 source sensor lineage human-input pack.

Builds an auditable, fill-by-human template for sensor family recovery. Sensor
family is never inferred from visual names: only a traceable `sensor_source_ref`
promotes a lineage. Only SENTINEL_2 can be spectral-eligible; SENTINEL_1 is
support-only; DINO/PNG/NPZ/UNKNOWN/CONFLICT block the optical spectral baseline.

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
OUT_DIR = PROJECT_ROOT / "outputs_public" / "mv2_data_sensor_lineage_human_pack"

PROMOTION_PATH = (
    PROJECT_ROOT
    / "outputs_public"
    / "mv2_data_source_sensor_lineage_promotion"
    / "mv2_data_07_sensor_lineage_promotion.csv"
)
SEED_PATH = (
    PROJECT_ROOT
    / "outputs_public"
    / "mv2_pre_unification_seed"
    / "revp_source_sensor_lineage_seed_10.csv"
)

TEMPLATE_FIELDS = [
    "target_rank",
    "patch_id",
    "asset_id",
    "slot_id",
    "evidence_id",
    "asset_ref",
    "source_asset_ref",
    "source_asset_type",
    "sensor_family",
    "sensor_source_ref",
    "spectral_eligible",
    "support_only",
    "review_status",
    "reviewer_notes",
    "blocked_reason",
]

ALLOWED_SENSOR_FAMILIES = [
    "SENTINEL_2",
    "SENTINEL_1",
    "DINO_DERIVED",
    "PNG_RENDER",
    "NPZ_EMBEDDING",
    "UNKNOWN",
    "CONFLICT",
]

DEFAULT_BLOCKED_REASON = "UNKNOWN_BLOCKED"


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


def eligibility_for_family(sensor_family: str) -> tuple[bool, bool]:
    """Return (spectral_eligible, support_only) for an already-validated family.

    Encodes the same policy the promotion stage enforces; used in the template's
    documentation and validated by tests. It does not decide a family by itself.
    """
    family = (sensor_family or "UNKNOWN").strip().upper()
    if family == "SENTINEL_2":
        return True, False
    if family == "SENTINEL_1":
        return False, True
    return False, False


def build_template_rows(
    promotion_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """One blank template row per known target. Sensor family is never inferred."""
    rows: list[dict[str, Any]] = []
    for idx, target in enumerate(promotion_rows, 1):
        rows.append(
            {
                "target_rank": idx,
                "patch_id": target.get("patch_id", ""),
                "asset_id": target.get("asset_id", ""),
                "slot_id": target.get("slot_id", ""),
                "evidence_id": target.get("evidence_id", ""),
                "asset_ref": target.get("asset_ref", ""),
                "source_asset_ref": "",
                "source_asset_type": "",
                "sensor_family": "UNKNOWN",
                "sensor_source_ref": "",
                "spectral_eligible": "false",
                "support_only": "false",
                "review_status": "PENDING_HUMAN_FILL",
                "reviewer_notes": "",
                "blocked_reason": DEFAULT_BLOCKED_REASON,
            }
        )
    return rows


def assert_no_inferred_sensor(rows: list[dict[str, Any]]) -> None:
    """Guardrail: the template ships with UNKNOWN/blocked rows, never eligible."""
    for row in rows:
        family = str(row.get("sensor_family", "")).strip().upper()
        eligible = str(row.get("spectral_eligible", "")).strip().lower()
        ref = str(row.get("sensor_source_ref", "")).strip()
        if eligible == "true" and (family != "SENTINEL_2" or not ref):
            raise ValueError(
                f"template row {row.get('patch_id')} is spectral_eligible without "
                "SENTINEL_2 + sensor_source_ref; lineage must not be inferred"
            )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "stage": "DATA-07",
        "artifact": "sensor_lineage_human_pack",
        "targets_in_template": len(rows),
        "spectral_eligible": 0,
        "promotion_status": DEFAULT_BLOCKED_REASON,
        "inferred_sensors": 0,
        "api_calls": 0,
        "downloads": 0,
        "rasters": 0,
        "crops": 0,
    }


def write_schema() -> None:
    write_json(
        PROJECT_ROOT / "datasets" / "schemas" / "schema_mv2_data_07_sensor_lineage_human_pack.json",
        {
            "schema_id": "schema_mv2_data_07_sensor_lineage_human_pack",
            "required_fields": [
                "patch_id",
                "asset_id",
                "sensor_family",
                "sensor_source_ref",
                "review_status",
            ],
            "template_fields": TEMPLATE_FIELDS,
            "allowed_sensor_families": ALLOWED_SENSOR_FAMILIES,
            "rules": [
                "so SENTINEL_2 pode ser spectral_eligible=true",
                "SENTINEL_1 e support_only=true",
                "DINO_DERIVED/PNG_RENDER/NPZ_EMBEDDING/UNKNOWN/CONFLICT bloqueiam baseline espectral",
                "sensor nunca inferido por nome visual",
                "sensor_source_ref obrigatorio para promover",
            ],
            "default_blocked_reason": DEFAULT_BLOCKED_REASON,
        },
    )


def _instructions_md() -> str:
    families = "\n".join(f"- {item}" for item in ALLOWED_SENSOR_FAMILIES)
    return f"""# DATA-07 - Instrucoes de recuperacao de sensor family

Este pacote e de entrada humana rastreavel. O sensor nunca e inferido pelo nome
visual do arquivo ou do asset.

## Como preencher
1. Abra `mv2_data_07_sensor_lineage_human_template.csv`.
2. Para cada `patch_id`/`asset_id`, identifique a origem real do asset e preencha
   `source_asset_ref` (referencia da cena/produto de origem) e `source_asset_type`.
3. Preencha `sensor_family` com um dos valores permitidos.
4. Preencha `sensor_source_ref` com a evidencia rastreavel que comprova a familia
   do sensor (metadado de produto, manifesto, registro auditavel). Sem isso, o
   registro permanece bloqueado.
5. `spectral_eligible` so pode ser `true` para `SENTINEL_2` com `sensor_source_ref`.
   `SENTINEL_1` recebe `support_only=true`. As demais familias bloqueiam o baseline.
6. Atualize `review_status` apenas apos conferencia.

## Valores permitidos de sensor_family
{families}

## Regras inviolaveis
- So `SENTINEL_2` pode ser `spectral_eligible=true`.
- `SENTINEL_1` e `support_only=true` (suporte SAR, nao baseline optico S2).
- `DINO_DERIVED`, `PNG_RENDER`, `NPZ_EMBEDDING`, `UNKNOWN` e `CONFLICT` bloqueiam
  o baseline espectral.
- Sensor nunca e inferido por nome visual.
- `sensor_source_ref` e obrigatorio para promover.
- Sem preenchimento, o estagio permanece `{DEFAULT_BLOCKED_REASON}`.
"""


def _source_policy_md() -> str:
    return """# DATA-07 - Politica de fontes de sensor lineage

## Evidencia aceita (sensor_source_ref)
- Metadado de produto Sentinel (ex.: product_id, datatake, MGRS tile).
- Manifesto de aquisicao oficial (CDSE, GEE) com referencia verificavel.
- Registro interno auditavel que vincule o asset a um produto de origem.

## Nao aceito
- Inferir Sentinel-2 so porque o PNG "parece" optico.
- Tratar embedding NPZ ou render DINO como raster espectral.
- Deduzir familia pelo nome do arquivo.

## Principio
A familia do sensor so e promovida com `sensor_source_ref` rastreavel. Sem cadeia
de origem, o registro fica `UNKNOWN_BLOCKED` e nao gera elegibilidade espectral.
"""


def _examples_md() -> str:
    return """# DATA-07 - Exemplos (ilustrativos, nao preencher o template com estes valores)

## Exemplo elegivel espectral (promove)
- sensor_family: SENTINEL_2
- source_asset_ref: S2A_MSIL2A_..._T24MTV (produto de origem)
- source_asset_type: SENTINEL_2_L2A
- sensor_source_ref: manifesto de aquisicao CDSE (registro auditavel)
- spectral_eligible: true
- support_only: false

## Exemplo suporte (nao baseline)
- sensor_family: SENTINEL_1
- sensor_source_ref: metadado GRD verificavel
- spectral_eligible: false
- support_only: true

## Exemplos bloqueados
- sensor_family: DINO_DERIVED -> bloqueado (nao e raster espectral).
- sensor_family: PNG_RENDER -> bloqueado (render nao e raster espectral).
- sensor_family: NPZ_EMBEDDING -> bloqueado (embedding nao e raster espectral).
- sensor_family: UNKNOWN ou CONFLICT -> bloqueado.
- SENTINEL_2 sem sensor_source_ref -> bloqueado (sem cadeia rastreavel).
"""


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser().parse_args(argv)

    promotion_rows = read_csv(PROMOTION_PATH)
    rows = build_template_rows(promotion_rows)
    assert_no_inferred_sensor(rows)

    write_schema()
    write_csv(OUT_DIR / "mv2_data_07_sensor_lineage_human_template.csv", TEMPLATE_FIELDS, rows)
    write_text(OUT_DIR / "mv2_data_07_sensor_lineage_instructions.md", _instructions_md())
    write_text(OUT_DIR / "mv2_data_07_sensor_lineage_source_policy.md", _source_policy_md())
    write_text(OUT_DIR / "mv2_data_07_sensor_lineage_examples.md", _examples_md())

    summary = summarize(rows)
    summary["promotion_template_found"] = PROMOTION_PATH.exists()
    summary["seed_found"] = SEED_PATH.exists()
    write_json(OUT_DIR / "mv2_data_07_sensor_lineage_human_pack_summary.json", summary)
    write_text(
        OUT_DIR / "commands.txt",
        "python scripts/mv2_data_07_prepare_sensor_lineage_human_pack.py\n"
        "# depois de preencher localmente:\n"
        "python scripts/mv2_data_07_source_sensor_lineage_promotion.py --input <caminho_local>",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
