# REV-P Official Download Command Pack (v1sq)
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
$env:REVP_DOWNLOAD_MAX_FILES = "20"
$env:REVP_DOWNLOAD_MAX_BYTES_PER_FILE = "262144000"   # bytes por arquivo
$env:REVP_DOWNLOAD_MAX_GB = "2"                                    # orcamento total aproximado (GB)
$env:REVP_DOWNLOAD_CONNECT_TIMEOUT_SECONDS = "15"
$env:REVP_DOWNLOAD_READ_TIMEOUT_SECONDS = "60"
$env:REVP_DOWNLOAD_RETRIES = "2"
$env:REVP_DOWNLOAD_RATE_LIMIT_SECONDS = "2"

$scripts = "scripts/protocolo_c"
$chain = @(
  "revp_v1sg_official_source_endpoint_registry.py",
  "revp_v1sh_inmet_historical_data_downloader.py",
  "revp_v1si_inmet_station_precipitation_extractor.py",
  "revp_v1sj_ana_hidroweb_acquisition.py",
  "revp_v1sk_institutional_document_discovery_queue.py",
  "revp_v1sl_official_download_orchestrator.py",
  "revp_v1sm_downloaded_external_document_intake_adapter.py",
  "revp_v1sn_official_data_provenance_license_audit.py",
  "revp_v1so_official_evidence_readiness_gate.py",
  "revp_v1sp_official_acquisition_bundle.py"
)

function Invoke-RevpChain {
  foreach ($s in $chain) {
    Write-Host "[v1sq] running $s"
    python "$scripts/$s"
    if ($LASTEXITCODE -ne 0) { throw "Falhou: $s" }
  }
}

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
