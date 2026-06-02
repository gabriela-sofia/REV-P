"""REV-P v1sq — Official download command pack.

Emits a reproducible, fail-closed command pack for the v1sg-v1sp official data
acquisition chain: a documented CSV of steps, a methodology doc, and a safe
PowerShell script that starts with downloads disabled and only enables real
downloads behind an explicit manual block. Never downloads anything itself.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

from revp_v1sg_v1sz_official_download_common import (
    DATASETS, DOCS, _p, guardrail_row, write_csv_with_header,
    write_doc, write_schema_for, forbidden_guardrail_scan,
    max_files, max_bytes_per_file, connect_timeout_sec, read_timeout_sec,
    retries,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"

OUT_PACK = _p("REVP_V1SQ_OUT_PACK", DATASETS / "protocol_c_official_download_command_pack_v1sq.csv")
OUT_SUMMARY = _p("REVP_V1SQ_OUT_SUMMARY", DATASETS / "protocol_c_official_download_command_pack_summary_v1sq.csv")
SCHEMA_P = _p("REVP_V1SQ_SCHEMA_P", DATASETS / "schemas" / "protocol_c_official_download_command_pack_v1sq_schema.csv")
SCHEMA_S = _p("REVP_V1SQ_SCHEMA_S", DATASETS / "schemas" / "protocol_c_official_download_command_pack_summary_v1sq_schema.csv")
DOC = _p("REVP_V1SQ_DOC", DOCS / "revp_v1sq_official_download_command_pack.md")
PS1 = _p("REVP_V1SQ_PS1", CONFIGS / "revp_official_download_commands_v1sq.ps1")

PACK_FIELDS = [
    "step_id", "stage", "command", "purpose", "downloads_enabled",
    "produces", "safety_note", "review_only", "can_create_operational_label",
    "can_train_model", "target_created", "ground_truth_operational",
    "formal_negative", "notes",
]
SUM_FIELDS = ["stat_key", "stat_value"]

# Ordered acquisition chain. v1si runs after v1sh so any downloaded INMET data
# can be extracted; consolidation stages (v1sl-v1sp) re-read existing outputs.
_CHAIN = [
    ("v1sg", "revp_v1sg_official_source_endpoint_registry.py", "Registrar endpoints oficiais (nao baixa)", "endpoint registry"),
    ("v1sh", "revp_v1sh_inmet_historical_data_downloader.py", "Baixar/auditar ZIPs historicos INMET", "inmet manifest+queue"),
    ("v1si", "revp_v1si_inmet_station_precipitation_extractor.py", "Extrair estacoes/precipitacao dos ZIPs INMET", "station+precip review"),
    ("v1sj", "revp_v1sj_ana_hidroweb_acquisition.py", "Planejar/baixar ANA HidroWeb/Telemetria", "ana manifest+queue"),
    ("v1sk", "revp_v1sk_institutional_document_discovery_queue.py", "Fila de descoberta institucional (manual)", "institutional queue"),
    ("v1sl", "revp_v1sl_official_download_orchestrator.py", "Consolidar manifests de download", "orchestrator manifest"),
    ("v1sm", "revp_v1sm_downloaded_external_document_intake_adapter.py", "Gerar drafts de intake review-only", "intake draft"),
    ("v1sn", "revp_v1sn_official_data_provenance_license_audit.py", "Auditar proveniencia e licenca", "provenance audit"),
    ("v1so", "revp_v1so_official_evidence_readiness_gate.py", "Gate de prontidao de evidencia", "readiness gate"),
    ("v1sp", "revp_v1sp_official_acquisition_bundle.py", "Bundle final + tabela TCC", "acquisition bundle"),
]


def _build_ps1() -> str:
    chain_lines = "\n".join(f'  "{cmd}",' for _, cmd, _, _ in _CHAIN).rstrip(",")
    return f"""# REV-P Official Download Command Pack (v1sq)
# Aquisicao review-only de dados publicos oficiais (INMET, ANA, CEMADEN, SGB, etc).
# IMPORTANTE: arquivos brutos vao para data/external_raw/ (git-ignored).
#             NUNCA commitar dados brutos. Todo output e review-only.
#
# Passo 1 roda com downloads DESABILITADOS (queue-only) — default seguro.
# Passo 2 (bloco manual) habilita downloads reais apenas quando voce decidir.

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)   # raiz do repositorio

# --- DEFAULT SEGURO: queue-only, sem rede ---
$env:REVP_ENABLE_OFFICIAL_DOWNLOADS = "false"
$env:REVP_DOWNLOAD_FORCE_QUEUE_ONLY = "true"

# --- Limites de download (aplicados so quando voce habilitar abaixo) ---
$env:REVP_DOWNLOAD_MAX_FILES = "{max_files()}"
$env:REVP_DOWNLOAD_MAX_BYTES_PER_FILE = "{max_bytes_per_file()}"   # bytes por arquivo
$env:REVP_DOWNLOAD_MAX_GB = "2"                                    # orcamento total aproximado (GB)
$env:REVP_DOWNLOAD_CONNECT_TIMEOUT_SECONDS = "{connect_timeout_sec()}"
$env:REVP_DOWNLOAD_READ_TIMEOUT_SECONDS = "{read_timeout_sec()}"
$env:REVP_DOWNLOAD_RETRIES = "{retries()}"
$env:REVP_DOWNLOAD_RATE_LIMIT_SECONDS = "2"

$scripts = "scripts/protocolo_c"
$chain = @(
{chain_lines}
)

function Invoke-RevpChain {{
  foreach ($s in $chain) {{
    Write-Host "[v1sq] running $s"
    python "$scripts/$s"
    if ($LASTEXITCODE -ne 0) {{ throw "Falhou: $s" }}
  }}
}}

Write-Host "[v1sq] Passo 1: cadeia queue-only (downloads desabilitados)..."
Invoke-RevpChain
Write-Host "[v1sq] Passo 1 concluido. Outputs review-only em datasets/."

# =====================================================================
# PASSO 2 — MANUAL: habilitar downloads reais
# Descomente as linhas abaixo SOMENTE quando quiser baixar de verdade.
# Os brutos vao para data/external_raw/ (git-ignored). NAO commitar brutos.
# Reveja datasets/protocol_c_official_acquisition_* depois de rodar.
# =====================================================================
# $env:REVP_ENABLE_OFFICIAL_DOWNLOADS = "true"
# $env:REVP_DOWNLOAD_FORCE_QUEUE_ONLY = "false"
# Write-Host "[v1sq] Passo 2: downloads habilitados (review-only, fail-closed)..."
# Invoke-RevpChain
# Write-Host "[v1sq] Passo 2 concluido. LEMBRE: nao commitar data/external_raw/."
"""


def run(datasets: Path | None = None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for i, (stage, cmd, purpose, produces) in enumerate(_CHAIN):
        row = {
            "step_id": f"V1SQ_S{i:02d}", "stage": stage,
            "command": f"python scripts/protocolo_c/{cmd}",
            "purpose": purpose, "downloads_enabled": "false",
            "produces": produces,
            "safety_note": "queue-only por default; raw git-ignored; nao commitar bruto",
            "notes": "",
        }
        row.update(guardrail_row())
        rows.append(row)

    forbidden_guardrail_scan(rows, "v1sq_pack")
    write_csv_with_header(OUT_PACK, rows, PACK_FIELDS)
    write_schema_for(SCHEMA_P, PACK_FIELDS, "v1sq_pack")

    ps1_text = _build_ps1()
    PS1.parent.mkdir(parents=True, exist_ok=True)
    PS1.write_text(ps1_text, encoding="utf-8")

    summary = [
        {"stat_key": "chain_steps", "stat_value": str(len(rows))},
        {"stat_key": "default_downloads_enabled", "stat_value": "false"},
        {"stat_key": "powershell_emitted", "stat_value": "true"},
        {"stat_key": "max_files", "stat_value": str(max_files())},
        {"stat_key": "max_bytes_per_file", "stat_value": str(max_bytes_per_file())},
        {"stat_key": "stage", "stat_value": "v1sq"},
    ]
    write_csv_with_header(OUT_SUMMARY, summary, SUM_FIELDS)
    write_schema_for(SCHEMA_S, SUM_FIELDS, "v1sq_summary")

    write_doc(DOC, "v1sq — Official Download Command Pack", [
        "## Objetivo",
        "Empacotar a cadeia v1sg-v1sp de aquisicao oficial num command pack "
        "reproduzivel e fail-closed: CSV documentado, doc e um PowerShell seguro.",
        "## Uso",
        "Rode `configs/revp_official_download_commands_v1sq.ps1`. Por padrao a "
        "cadeia roda em modo queue-only (sem rede). Para baixar de verdade, "
        "descomente o bloco manual (Passo 2) e reexecute.",
        "## Limites e seguranca",
        f"max_files={max_files()}, max_bytes_per_file={max_bytes_per_file()}, "
        f"connect_timeout={connect_timeout_sec()}s, read_timeout={read_timeout_sec()}s, "
        f"retries={retries()}. Downloads so para dominios .gov.br da allowlist; "
        "redirects para fora da allowlist sao bloqueados.",
        "## Limitacoes",
        "Os brutos vao para `data/external_raw/` (git-ignored) e nao devem ser "
        "commitados. Todos os outputs sao review-only: nao criam rotulos, targets, "
        "ground truth operacional nem negativos formais.",
    ])
    print(f"[v1sq] steps={len(rows)} ps1=emitted downloads_default=false")
    return {"steps": len(rows), "ps1": str(PS1.name)}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1sq official download command pack").parse_args()
    run()
