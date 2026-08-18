"""MV2-DATA-06/07/08 metadata-only readiness orchestrator (dry-run).

Runs the human-input packs, reinforces the DATA-08 preflight, and recomputes the
consolidated readiness for the metadata-only unblock front. It is strictly
fail-closed by default:

  - no network
  - no downloads
  - no crops
  - no raster

The promotion / preflight / MV2-16 portions are recomputed from the committed
canonical outputs using the existing modules' pure functions. The orchestrator
never overwrites those canonical files: it only writes the human packs, the
DATA-08 checklist docs, the consolidation outputs, and the execution report.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

import sys

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import mv2_data_06_prepare_temporal_window_human_pack as d06pack
import mv2_data_07_prepare_sensor_lineage_human_pack as d07pack
import mv2_data_06_temporal_window_promotion as d06prom
import mv2_data_07_source_sensor_lineage_promotion as d07prom
import mv2_data_08_metadata_only_probe_runner as d08probe
import mv2_16_unified_sentinel_execution_core as core

OUT_DIR = PROJECT_ROOT / "outputs_public" / "mv2_data_06_08_metadata_only_readiness"
EXEC_REPORTS = PROJECT_ROOT / "outputs_public" / "execution_reports"
PROBE_DIR = PROJECT_ROOT / "outputs_public" / "mv2_data_metadata_only_probe"
CONFIG_EXAMPLE = PROJECT_ROOT / "configs" / "api_config.metadata_only.example.json"

DATA06_PROMOTION_CSV = (
    PROJECT_ROOT
    / "outputs_public"
    / "mv2_data_temporal_window_promotion"
    / "mv2_data_06_temporal_window_promotion.csv"
)
DATA07_PROMOTION_CSV = (
    PROJECT_ROOT
    / "outputs_public"
    / "mv2_data_source_sensor_lineage_promotion"
    / "mv2_data_07_sensor_lineage_promotion.csv"
)
MV2_16_MATRIX_CSV = (
    PROJECT_ROOT
    / "outputs_public"
    / "mv2_16_unified_sentinel_execution_core"
    / "mv2_16_unified_gate_matrix.csv"
)

DEFAULT_TIMESTAMP = "20260623T213111"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def evaluate_data06() -> dict[str, Any]:
    """Recompute DATA-06 promotion status from committed rows, no overwrite."""
    rows = read_csv(DATA06_PROMOTION_CSV)
    template = d06prom.find_filled_template()
    template_found = template is not None
    statuses = [
        d06prom.classify_temporal_row(row, template_found)[0] for row in rows
    ]
    promoted = sum(1 for status in statuses if status == "PROMOTED_METADATA_READY")
    if not rows:
        consolidated = "BLOCKED_NO_FILLED_TEMPLATE"
    elif not template_found:
        consolidated = "BLOCKED_NO_FILLED_TEMPLATE"
    elif promoted > 0:
        consolidated = "PROMOTED_METADATA_READY"
    else:
        consolidated = "BLOCKED_NO_FILLED_TEMPLATE"
    return {
        "stage": "DATA-06",
        "filled_template_found": template_found,
        "targets": len(rows),
        "promoted_metadata_ready": promoted,
        "status": consolidated,
    }


def evaluate_data07() -> dict[str, Any]:
    """Recompute DATA-07 lineage status from committed rows, no overwrite."""
    rows = read_csv(DATA07_PROMOTION_CSV)
    statuses = [
        d07prom.classify_sensor_lineage(
            row.get("sensor_family", ""), row.get("source_asset_ref", "")
        )[0]
        for row in rows
    ]
    eligible = sum(1 for status in statuses if status == "SENTINEL_2_ELIGIBLE")
    if not rows:
        consolidated = "UNKNOWN_BLOCKED"
    elif eligible > 0:
        consolidated = "SENTINEL_2_ELIGIBLE"
    else:
        consolidated = "UNKNOWN_BLOCKED"
    return {
        "stage": "DATA-07",
        "targets": len(rows),
        "sentinel_2_eligible": eligible,
        "status": consolidated,
    }


def evaluate_data08() -> dict[str, Any]:
    """Recompute DATA-08 preflight from local config (absent => BLOCKED_NO_CONFIG)."""
    preflight = d08probe.metadata_preflight(d08probe.DEFAULT_CONFIG)
    status = preflight.get("status", "BLOCKED_NO_CONFIG")
    if status == "READY_METADATA_ONLY":
        consolidated = "READY_METADATA_ONLY"
    elif status == "BLOCKED_BY_FLAGS":
        consolidated = "BLOCKED_BY_FLAGS"
    else:
        consolidated = "BLOCKED_NO_CONFIG"
    return {
        "stage": "DATA-08",
        "config_present": preflight.get("config_present", False),
        "calls_allowed": preflight.get("calls_allowed", False),
        "status": consolidated,
    }


def evaluate_mv2_16() -> dict[str, Any]:
    """Recompute MV2-16 readiness and Day 10 from committed gate matrix."""
    rows = read_csv(MV2_16_MATRIX_CSV)
    readiness = core.compute_mv2_16_readiness(rows)
    day10_states = {row.get("day10_status", "BLOCKED") for row in rows}
    if not rows:
        day10 = "BLOCKED"
    elif day10_states == {"READY_REVIEW_ONLY"}:
        day10 = "READY_REVIEW_ONLY"
    else:
        day10 = "BLOCKED"
    return {
        "stage": "MV2-16",
        "targets": len(rows),
        "readiness": readiness,
        "day10_status": day10,
    }


def ensure_metadata_only_example_config() -> None:
    write_json(
        CONFIG_EXAMPLE,
        {
            "_comment": (
                "Exemplo publico metadata-only. Default seguro: tudo desligado. Para sair de "
                "dry-run, copie para configs/api_config.local.json (NAO versionado) e habilite "
                "apenas allow_network e allow_metadata_calls. Nunca habilite raster/canary aqui. "
                "Segredos vao em variaveis de ambiente."
            ),
            "allow_network": False,
            "allow_metadata_calls": False,
            "allow_raster_download": False,
            "allow_canary_download": False,
            "providers": {
                "GEE": {"enabled": False, "project_id_env": "REV_P_GEE_PROJECT_ID"},
                "CDSE_STAC": {"enabled": False, "base_url": "https://stac.dataspace.copernicus.eu/v1"},
                "CDSE_ODATA": {"enabled": False, "base_url": "https://catalogue.dataspace.copernicus.eu/odata/v1"},
            },
        },
    )


def write_data08_checklist() -> None:
    write_text(
        PROBE_DIR / "mv2_data_08_metadata_only_config_instructions.md",
        """# DATA-08 - Instrucoes de config metadata-only

## Estado padrao (dry-run)
Sem `configs/api_config.local.json`, o preflight retorna `BLOCKED_NO_CONFIG` e
nenhuma chamada e feita.

## Exemplo seguro versionado
`configs/api_config.metadata_only.example.json` mantem todos os flags em `false`.

## Para sair de dry-run (acao manual humana)
Crie LOCALMENTE (nunca versione) `configs/api_config.local.json` com:

```json
{
  "allow_network": true,
  "allow_metadata_calls": true,
  "allow_raster_download": false,
  "allow_canary_download": false
}
```

E exporte as variaveis de ambiente necessarias (sem versionar segredo), por exemplo:

```
REV_P_GEE_PROJECT_ID=<seu_project_id>
```

## Nunca criar/versionar
- `configs/api_config.local.json`
- `.env`
- qualquer token ou credencial

Mesmo com `allow_metadata_calls=true`, `allow_raster_download` e
`allow_canary_download` permanecem `false`: a frente e metadata-only.
""",
    )
    write_text(
        PROBE_DIR / "mv2_data_08_metadata_only_preflight_checklist.md",
        """# DATA-08 - Checklist de preflight metadata-only

- [ ] DATA-06 promovido: janela temporal com fonte rastreavel (`PROMOTED_METADATA_READY`).
- [ ] DATA-07 promovido: sensor `SENTINEL_2` elegivel com `sensor_source_ref`.
- [ ] `configs/api_config.local.json` criado localmente (nao versionado).
- [ ] `allow_network=true` e `allow_metadata_calls=true`.
- [ ] `allow_raster_download=false` e `allow_canary_download=false`.
- [ ] Variaveis de ambiente de provedor exportadas (sem segredo no repo).
- [ ] Nenhum `.env`/token/credencial versionado.

Enquanto qualquer item acima estiver pendente, o preflight permanece
`BLOCKED_NO_CONFIG`/`BLOCKED_BY_FLAGS` e o Dia 10 segue `BLOCKED`.
""",
    )


def build_consolidation(
    d06: dict[str, Any],
    d07: dict[str, Any],
    d08: dict[str, Any],
    mv216: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stage": "DATA-06/07/08 metadata-only readiness",
        "mode": "dry-run",
        "fail_closed": True,
        "data_06_status": d06["status"],
        "data_07_status": d07["status"],
        "data_08_status": d08["status"],
        "mv2_16_readiness": mv216["readiness"],
        "day10_status": mv216["day10_status"],
        "api_calls": 0,
        "downloads": 0,
        "rasters": 0,
        "crops": 0,
        "details": {
            "data_06": d06,
            "data_07": d07,
            "data_08": d08,
            "mv2_16": mv216,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", default=DEFAULT_TIMESTAMP)
    args = parser.parse_args(argv)
    stamp = args.timestamp

    # 1-2. Human packs (additive; new output dirs only).
    d06pack.main([])
    d07pack.main([])

    # 3. DATA-08 config checklist + safe example config.
    ensure_metadata_only_example_config()
    write_data08_checklist()

    # 4-7. Recompute statuses from committed canonical outputs (no overwrite).
    d06 = evaluate_data06()
    d07 = evaluate_data07()
    d08 = evaluate_data08()
    mv216 = evaluate_mv2_16()

    consolidation = build_consolidation(d06, d07, d08, mv216)
    write_json(OUT_DIR / "mv2_data_06_08_readiness_summary.json", consolidation)
    write_text(
        OUT_DIR / "mv2_data_06_08_readiness_report.md",
        f"""# DATA-06/07/08 metadata-only readiness (dry-run)

- DATA-06: {consolidation['data_06_status']}
- DATA-07: {consolidation['data_07_status']}
- DATA-08: {consolidation['data_08_status']}
- MV2-16: {consolidation['mv2_16_readiness']}
- Dia 10: {consolidation['day10_status']}
- chamadas/downloads/rasters/crops: 0/0/0/0
""",
    )
    write_text(
        OUT_DIR / "commands.txt",
        "python scripts/mv2_data_06_08_metadata_only_readiness_orchestrator.py",
    )

    # 6. Final execution report (timestamped).
    write_execution_report(stamp, consolidation)

    print(
        "[mv2_data_06_08_orchestrator] "
        f"DATA-06={d06['status']} DATA-07={d07['status']} "
        f"DATA-08={d08['status']} MV2-16={mv216['readiness']} "
        f"Dia10={mv216['day10_status']}"
    )
    return 0


def write_execution_report(stamp: str, consolidation: dict[str, Any]) -> None:
    branch = "dados/desbloqueio-metadata-only-data-06-08"
    json_out = EXEC_REPORTS / f"revp_data_06_08_metadata_only_readiness_summary_{stamp}.json"
    write_json(
        json_out,
        {
            "branch": branch,
            "worktree": "REV-P-dados-metadata-only-data-06-08",
            **consolidation,
        },
    )
    md_out = EXEC_REPORTS / f"revp_data_06_08_metadata_only_readiness_report_{stamp}.md"
    write_text(
        md_out,
        f"""# Relatorio de execucao - desbloqueio metadata-only DATA-06/07/08

## 1. Branch / worktree
- branch: {branch}
- worktree: REV-P-dados-metadata-only-data-06-08
- base: marco/pre-unificacao-gates-mv1 (HEAD 1c5744b)

## 2. Estado DATA-06
- {consolidation['data_06_status']}
- targets no template de promocao: {consolidation['details']['data_06']['targets']}
- janela com fonte rastreavel: nenhuma preenchida (template humano pendente)

## 3. Estado DATA-07
- {consolidation['data_07_status']}
- targets: {consolidation['details']['data_07']['targets']}
- Sentinel-2 elegivel: {consolidation['details']['data_07']['sentinel_2_eligible']}

## 4. Estado DATA-08
- {consolidation['data_08_status']}
- config local presente: {consolidation['details']['data_08']['config_present']}

## 5. Templates criados
- outputs_public/mv2_data_temporal_window_human_pack/ (DATA-06)
- outputs_public/mv2_data_sensor_lineage_human_pack/ (DATA-07)
- outputs_public/mv2_data_metadata_only_probe/mv2_data_08_*.md (DATA-08)
- configs/api_config.metadata_only.example.json

## 6. Proximos inputs humanos
- DATA-06: preencher janela temporal com fonte rastreavel e rodar a promocao.
- DATA-07: preencher sensor_family + sensor_source_ref e rodar a promocao.
- DATA-08: criar configs/api_config.local.json local (nao versionado) para sair de dry-run.

## 7. Chamadas / downloads / rasters / crops
- 0 / 0 / 0 / 0

## 8. Dia 10
- {consolidation['day10_status']}

## 9. Criterio para sair de dry-run
- DATA-06 PROMOTED_METADATA_READY + DATA-07 SENTINEL_2_ELIGIBLE +
  DATA-08 READY_METADATA_ONLY (config local habilitada metadata-only).

## 10. Proximo comando recomendado
- python scripts/mv2_data_06_08_metadata_only_readiness_orchestrator.py
""",
    )


if __name__ == "__main__":
    raise SystemExit(main())
