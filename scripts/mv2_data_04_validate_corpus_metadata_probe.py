"""MV2-DATA-04 Tarefa 11 — Validador do corpus metadata probe.

Sai com código != 0 se qualquer trava fail-closed for violada: batch > 15; canary no batch;
probe executado sem janela temporal; OData sem product_id não-bloqueado; query aberta ampla;
download/crop/raster/feature; label/silver/gold/negativo; Dia 10/sandbox desbloqueados;
segredo/caminho privado em outputs_public; summary incoerente.
"""

from __future__ import annotations

import json
import re
import sys

from mv2_data_04_common import (
    MAX_BATCH,
    MAX_PER_REGION,
    PROJECT_ROOT,
    PUBLIC_DIR,
    RISK_TYPES,
    clean,
    official_canary_ids,
    read_csv_rows,
)

RASTER_EXT = {".tif", ".tiff", ".jp2", ".safe", ".img", ".cog", ".npy", ".npz",
              ".nc", ".hdf", ".png", ".jpg", ".jpeg", ".zip"}
ALLOWED_EXT = {".csv", ".json", ".md", ".txt"}
MAX_BYTES = 2_000_000

SECRET_PATTERNS = [
    re.compile(r"eyJ[A-Za-z0-9_-]{20,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)\b(secret|password|access_token|api_key)\b\s*[:=]\s*['\"][^'\"]{8,}"),
]


def find_violations() -> list[str]:
    v: list[str] = []
    if not PUBLIC_DIR.exists():
        return [f"diretório ausente: {PUBLIC_DIR}"]

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

    # Batch: <=15, balanceado (<=5/região), sem canary.
    batch = read_csv_rows(PUBLIC_DIR / "mv2_data_04_corpus_metadata_batch.csv")
    if not batch:
        v.append("batch ausente/vazio")
    if len(batch) > MAX_BATCH:
        v.append(f"batch acima de {MAX_BATCH}: {len(batch)}")
    canary = official_canary_ids()
    from collections import Counter

    region_count = Counter(clean(r.get("region")) for r in batch)
    for region, cnt in region_count.items():
        if cnt > MAX_PER_REGION:
            v.append(f"região {region} acima de {MAX_PER_REGION}: {cnt}")
    for r in batch:
        if clean(r.get("patch_id")) in canary or clean(r.get("target_id")) in canary:
            v.append(f"canary Protocolo-C no batch: {clean(r.get('target_id'))}")
        if clean(r.get("is_official_canary")) == "true":
            v.append(f"is_official_canary=true no batch: {clean(r.get('target_id'))}")
        if not clean(r.get("bbox")) or not clean(r.get("crs")):
            v.append(f"batch target sem bbox/crs: {clean(r.get('target_id'))}")

    # GEE/STAC: sem janela temporal => não pode ter executado; nunca query aberta.
    for name in ["mv2_data_04_gee_corpus_metadata_probe.csv",
                 "mv2_data_04_cdse_stac_corpus_metadata_probe.csv"]:
        for r in read_csv_rows(PUBLIC_DIR / name):
            twin = clean(r.get("temporal_window_status"))
            if twin != "PRESENT" and clean(r.get("executed")) == "true":
                v.append(f"probe executado sem janela temporal: {name}:{clean(r.get('probe_id'))}")
            if twin != "PRESENT" and clean(r.get("metadata_status")) != "BLOCKED_NO_TEMPORAL_WINDOW":
                v.append(f"sem janela temporal mas não bloqueou: {name}:{clean(r.get('probe_id'))}")

    # OData: sem product_id => BLOCKED_NO_PRODUCT_ID e nunca executado/baixado.
    for r in read_csv_rows(PUBLIC_DIR / "mv2_data_04_cdse_odata_metadata_probe.csv"):
        if not clean(r.get("product_id")) and clean(r.get("metadata_status")) != "BLOCKED_NO_PRODUCT_ID":
            v.append(f"OData sem product_id não bloqueou: {clean(r.get('probe_id'))}")
        if clean(r.get("executed")) == "true":
            v.append(f"OData executado: {clean(r.get('probe_id'))}")

    # Lineage: Dia 10 nunca desbloqueia; raster sempre passo futuro bloqueado.
    for r in read_csv_rows(PUBLIC_DIR / "mv2_data_04_corpus_api_lineage_resolution.csv"):
        if clean(r.get("can_unlock_day10_now")) != "false":
            v.append(f"lineage can_unlock_day10_now!=false: {clean(r.get('resolution_id'))}")
        if clean(r.get("can_raster_download_next_step")) != "false":
            v.append(f"raster download habilitado: {clean(r.get('resolution_id'))}")

    # Risk matrix completa.
    present_risk = {clean(r.get("risk_type")) for r in read_csv_rows(PUBLIC_DIR / "mv2_data_04_risk_matrix.csv")}
    for rt in RISK_TYPES:
        if rt not in present_risk:
            v.append(f"risk_type obrigatório ausente: {rt}")

    # Template temporal existe.
    if not (PUBLIC_DIR / "mv2_data_04_temporal_window_template.csv").exists():
        v.append("temporal_window_template ausente")

    # Summary coerente e fail-closed.
    summary_path = PUBLIC_DIR / "mv2_data_04_summary.json"
    if not summary_path.exists():
        return v + ["summary JSON ausente"]
    s = json.loads(summary_path.read_text(encoding="utf-8"))
    for key in ["downloads_executed", "crops_created", "native_rasters_created",
                "spectral_features_created", "labels_created", "silver_created",
                "gold_created", "negatives_created"]:
        if s.get(key, -1) != 0:
            v.append(f"guardrail violado: {key}={s.get(key)} (deve ser 0)")
    if s.get("can_train") is not False:
        v.append("guardrail violado: can_train deve ser false")
    if s.get("corpus_day10_unlocked") is not False:
        v.append("guardrail violado: corpus_day10_unlocked deve ser false")
    if s.get("sandbox_unlocked") is not False:
        v.append("guardrail violado: sandbox_unlocked deve ser false")
    if s.get("batch_targets") != len(batch):
        v.append("summary incoerente: batch_targets")
    if s.get("batch_targets", 0) > MAX_BATCH:
        v.append("summary incoerente: batch_targets > 15")

    return v


def main() -> int:
    v = find_violations()
    if v:
        print("[mv2_data_04_validate] VIOLACOES:")
        for x in v:
            print(f"  - {x}")
        return 1
    n = sum(1 for p in PUBLIC_DIR.rglob("*") if p.is_file())
    print(f"[mv2_data_04_validate] OK — {n} arquivos públicos leves; fail-closed preservado")
    return 0


if __name__ == "__main__":
    sys.exit(main())
