"""MV2-DATA-02 Tarefa 11 — Validador geral do harness de API/raster.

Sai com código != 0 se qualquer trava fail-closed for violada: segredo/raster/caminho
privado em outputs_public; download com config bloqueada; canary sem lineage forte; bulk
download (sempre proibido); STAC real em lote; GEE/HTTP sem config; Dia 10 desbloqueado sem
raster privado validado; label/silver/gold/negativo/treino/sandbox != zero; summary
incoerente.
"""

from __future__ import annotations

import json
import re
import sys

from mv2_data_02_api_config import load_config
from mv2_data_02_common import (
    PROJECT_ROOT,
    PUBLIC_DIR,
    RISK_TYPES,
    clean,
    read_csv_rows,
)

RASTER_EXT = {".tif", ".tiff", ".jp2", ".safe", ".img", ".cog", ".npy", ".npz",
              ".nc", ".hdf", ".png", ".jpg", ".jpeg", ".zip"}
ALLOWED_EXT = {".csv", ".json", ".md", ".txt"}
MAX_BYTES = 2_000_000

# Padrões de segredo plausível (token/chave) — nunca devem aparecer em outputs_public.
SECRET_PATTERNS = [
    re.compile(r"eyJ[A-Za-z0-9_-]{20,}"),               # JWT
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),  # PEM
    re.compile(r"(?i)\b(secret|password|access_token|api_key)\b\s*[:=]\s*['\"][^'\"]{8,}"),
]


def find_violations() -> list[str]:
    v: list[str] = []
    if not PUBLIC_DIR.exists():
        return [f"diretório ausente: {PUBLIC_DIR}"]
    cfg = load_config()

    # outputs_public: só leve; sem raster; sem caminho privado; sem segredo.
    for p in sorted(PUBLIC_DIR.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(PROJECT_ROOT).as_posix()
        ext = p.suffix.lower()
        if ext in RASTER_EXT:
            v.append(f"raster/binário em outputs_public: {rel}")
        elif ext not in ALLOWED_EXT:
            v.append(f"extensão não permitida: {rel} ({ext})")
        if p.stat().st_size > MAX_BYTES:
            v.append(f"arquivo acima de {MAX_BYTES} bytes: {rel}")
        if ext in ALLOWED_EXT:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            if "C:/Users" in txt or "C:\\Users" in txt:
                v.append(f"caminho absoluto privado em {p.name}")
            for pat in SECRET_PATTERNS:
                if pat.search(txt):
                    v.append(f"possível segredo em {p.name}")
                    break

    # Provider registry: nenhum segredo, só nome de env var.
    providers = read_csv_rows(PUBLIC_DIR / "mv2_data_02_api_provider_registry.csv")
    if not providers:
        v.append("provider registry ausente/vazio")

    metadata_blocked = not (cfg.get("allow_network") and cfg.get("allow_metadata_calls"))

    # STAC/GEE: se config bloqueia metadados, nenhuma chamada pode ter executado.
    for name, col in [
        ("mv2_data_02_cdse_stac_metadata_results.csv", "executed"),
        ("mv2_data_02_gee_metadata_results.csv", "executed"),
        ("mv2_data_02_cdse_odata_metadata_results.csv", "executed"),
    ]:
        rows = read_csv_rows(PUBLIC_DIR / name)
        if metadata_blocked and any(clean(r.get(col)) == "true" for r in rows):
            v.append(f"chamada executada com config bloqueada: {name}")

    # OData: nenhum download por padrão.
    odata = read_csv_rows(PUBLIC_DIR / "mv2_data_02_cdse_odata_metadata_results.csv")
    if not cfg.get("allow_raster_download"):
        for r in odata:
            if clean(r.get("download_executed")) == "true":
                v.append("OData baixou com allow_raster_download=false")

    # Lineage resolution: bulk download sempre false; canary só com lineage forte.
    lineage = read_csv_rows(PUBLIC_DIR / "mv2_data_02_api_lineage_resolution.csv")
    for r in lineage:
        if clean(r.get("can_promote_to_bulk_download")) != "false":
            v.append(f"bulk download habilitado: {clean(r.get('resolution_id'))}")
        if clean(r.get("can_promote_to_raster_canary")) == "true" and \
           clean(r.get("lineage_status")) != "API_LINEAGE_RESOLVED_STRONG":
            v.append(f"canary sem lineage forte: {clean(r.get('resolution_id'))}")

    # Canary plan: download/crop nunca true por padrão; data_ok exige lineage forte.
    plan = read_csv_rows(PUBLIC_DIR / "mv2_data_02_raster_canary_plan.csv")
    for r in plan:
        if clean(r.get("download_allowed_now")) == "true" and not cfg.get("allow_canary_download"):
            v.append(f"download_allowed_now=true sem config: {clean(r.get('canary_id'))}")
        if clean(r.get("crop_allowed_now")) == "true" and not cfg.get("allow_canary_download"):
            v.append(f"crop_allowed_now=true sem config: {clean(r.get('canary_id'))}")
        if not clean(r.get("private_output_dir")):
            v.append(f"canary sem private_output_dir: {clean(r.get('canary_id'))}")

    # Canary executor: sem config/env, nenhuma execução.
    manifest = read_csv_rows(PUBLIC_DIR / "mv2_data_02_raster_canary_execution_manifest.csv")
    canary_ok = bool(cfg.get("allow_canary_download")) and bool(cfg.get("allow_raster_download"))
    for r in manifest:
        if clean(r.get("executed")) == "true" and not canary_ok:
            v.append(f"canary executado sem config: {clean(r.get('canary_id'))}")

    # Private raster validation: nenhum vazamento público.
    private_val = read_csv_rows(PUBLIC_DIR / "mv2_data_02_private_raster_validation.csv")
    for r in private_val:
        if clean(r.get("public_leak_detected")) == "true":
            v.append(f"vazamento público de raster detectado: {clean(r.get('validation_id'))}")

    # Risk matrix completa.
    present = {clean(r.get("risk_type")) for r in read_csv_rows(PUBLIC_DIR / "mv2_data_02_api_raster_risk_matrix.csv")}
    for rt in RISK_TYPES:
        if rt not in present:
            v.append(f"risk_type obrigatório ausente: {rt}")

    # Summary coerente e fail-closed.
    summary_path = PUBLIC_DIR / "mv2_data_02_summary.json"
    if not summary_path.exists():
        return v + ["summary JSON ausente"]
    s = json.loads(summary_path.read_text(encoding="utf-8"))
    for key in ["crops_created", "native_rasters_created", "spectral_features_created",
                "labels_created", "silver_created", "gold_created", "negatives_created",
                "public_raster_leaks"]:
        if s.get(key, -1) != 0:
            v.append(f"guardrail violado: {key}={s.get(key)} (deve ser 0)")
    if s.get("can_train") is not False:
        v.append("guardrail violado: can_train deve ser false")
    if s.get("sandbox_status") != "bloqueado":
        v.append("guardrail violado: sandbox_status deve ser 'bloqueado'")
    # Dia 10 só pode desbloquear com raster privado validado.
    if s.get("can_unlock_day10_now") and s.get("private_rasters_validated", 0) <= 0:
        v.append("guardrail violado: Dia 10 desbloqueado sem raster privado validado")
    if not cfg.get("allow_raster_download") and s.get("downloads_executed", 0) != 0:
        v.append("guardrail violado: downloads_executed != 0 com config bloqueada")
    # Coerência de contagens.
    if s.get("total_targets") != len(read_csv_rows(PUBLIC_DIR / "mv2_data_02_api_lineage_resolution.csv")):
        v.append("summary incoerente: total_targets != linhas de lineage resolution")
    if s.get("raster_canary_candidates") != len(plan):
        v.append("summary incoerente: raster_canary_candidates")

    return v


def main() -> int:
    v = find_violations()
    if v:
        print("[mv2_data_02_validate] VIOLACOES:")
        for x in v:
            print(f"  - {x}")
        return 1
    n = sum(1 for p in PUBLIC_DIR.rglob("*") if p.is_file())
    print(f"[mv2_data_02_validate] OK — {n} arquivos públicos leves; fail-closed preservado")
    return 0


if __name__ == "__main__":
    sys.exit(main())
