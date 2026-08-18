"""MV2-DATA-05 Tarefa 9 — Validador do intake de janela temporal.

Sai com código != 0 se qualquer trava fail-closed for violada: promover janela vazia/sem
fonte/sem review/futura/start>end/ampla demais; promover target inexistente ou asset/patch
mismatch; habilitar GEE/STAC sem promoção; habilitar OData; executar API; download/crop/
raster/feature; label/silver/gold/negativo; Dia 10/sandbox; segredo/caminho privado; summary
incoerente.
"""

from __future__ import annotations

import json
import re
import sys

from mv2_data_05_common import (
    APPROVED_REVIEW,
    PROJECT_ROOT,
    PUBLIC_DIR,
    RISK_TYPES,
    clean,
    corpus_index,
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

PROMOTED = {"TEMPORAL_WINDOW_PROMOTED_STRONG", "TEMPORAL_WINDOW_PROMOTED_PARTIAL"}


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

    corpus = corpus_index()
    evidence = {clean(e.get("target_id")): e
                for e in read_csv_rows(PUBLIC_DIR / "mv2_data_05_temporal_evidence_validation.csv")}
    gate = read_csv_rows(PUBLIC_DIR / "mv2_data_05_temporal_promotion_gate.csv")
    if not gate:
        v.append("promotion gate ausente/vazio")

    for g in gate:
        tid = clean(g.get("target_id"))
        promotion = clean(g.get("promotion_status"))
        ev = evidence.get(tid, {})
        ev_status = clean(ev.get("evidence_status"))
        promoted = promotion in PROMOTED

        if promoted:
            # target precisa existir, asset/patch bater, datas válidas/nao futuras, fonte+review p/ strong.
            if tid not in corpus:
                v.append(f"promovido com target inexistente: {tid}")
            if clean(ev.get("asset_patch_match")) != "true":
                v.append(f"promovido com asset/patch mismatch: {tid}")
            if clean(ev.get("future_date")) == "true":
                v.append(f"promovido com data futura: {tid}")
            if clean(ev.get("start_after_end")) == "true":
                v.append(f"promovido com start>end: {tid}")
            if clean(ev.get("too_broad")) == "true":
                v.append(f"promovido com janela ampla demais: {tid}")
            if not clean(g.get("source_ref")):
                v.append(f"promovido sem source_ref: {tid}")
            if promotion == "TEMPORAL_WINDOW_PROMOTED_STRONG":
                if not (clean(ev.get("source_present")) == "true"):
                    v.append(f"strong sem fonte completa: {tid}")
                if clean(g.get("review_status")).upper() not in APPROVED_REVIEW:
                    v.append(f"strong sem review aprovado: {tid}")
            if ev_status not in {"TEMPORAL_EVIDENCE_VALID_STRONG", "TEMPORAL_EVIDENCE_VALID_PARTIAL"}:
                v.append(f"promovido sem evidência válida: {tid}")

        # GEE/STAC só habilitam com promoção; OData nunca habilita aqui.
        if clean(g.get("can_enable_gee_metadata_probe")) == "true" and not promoted:
            v.append(f"GEE habilitado sem promoção: {tid}")
        if clean(g.get("can_enable_stac_metadata_probe")) == "true" and not promoted:
            v.append(f"STAC habilitado sem promoção: {tid}")
        if clean(g.get("can_enable_odata_probe")) == "true":
            v.append(f"OData habilitado (proibido sem product_id): {tid}")
        if clean(g.get("can_unlock_day10_now")) != "false":
            v.append(f"can_unlock_day10_now!=false: {tid}")

    # probe-ready batch coerente com o gate.
    pr = read_csv_rows(PUBLIC_DIR / "mv2_data_05_probe_ready_batch.csv")
    for r in pr:
        if clean(r.get("can_probe_odata_metadata")) == "true":
            v.append(f"probe-ready OData habilitado: {clean(r.get('target_id'))}")

    # Risk matrix completa.
    present_risk = {clean(r.get("risk_type")) for r in read_csv_rows(PUBLIC_DIR / "mv2_data_05_risk_matrix.csv")}
    for rt in RISK_TYPES:
        if rt not in present_risk:
            v.append(f"risk_type obrigatório ausente: {rt}")

    # Correction template existe.
    if not (PUBLIC_DIR / "mv2_data_05_temporal_window_correction_template.csv").exists():
        v.append("correction template ausente")

    # Summary coerente e fail-closed.
    summary_path = PUBLIC_DIR / "mv2_data_05_summary.json"
    if not summary_path.exists():
        return v + ["summary JSON ausente"]
    s = json.loads(summary_path.read_text(encoding="utf-8"))
    for key in ["gee_calls_executed", "stac_calls_executed", "odata_calls_executed",
                "downloads_executed", "crops_created", "native_rasters_created",
                "spectral_features_created", "labels_created", "silver_created",
                "gold_created", "negatives_created", "probe_ready_odata"]:
        if s.get(key, -1) != 0:
            v.append(f"guardrail violado: {key}={s.get(key)} (deve ser 0)")
    if s.get("can_train") is not False:
        v.append("guardrail violado: can_train deve ser false")
    if s.get("corpus_day10_unlocked") is not False:
        v.append("guardrail violado: corpus_day10_unlocked deve ser false")
    if s.get("sandbox_unlocked") is not False:
        v.append("guardrail violado: sandbox_unlocked deve ser false")
    # Sem template preenchido => 0 promovidas.
    if s.get("filled_template_found") is False:
        if (s.get("temporal_promoted_strong", 0) + s.get("temporal_promoted_partial", 0)) != 0:
            v.append("template vazio mas promoveu janela")
    if s.get("total_input_rows") != len(read_csv_rows(PUBLIC_DIR / "mv2_data_05_normalized_temporal_windows.csv")):
        v.append("summary incoerente: total_input_rows")

    return v


def main() -> int:
    v = find_violations()
    if v:
        print("[mv2_data_05_validate] VIOLACOES:")
        for x in v:
            print(f"  - {x}")
        return 1
    n = sum(1 for p in PUBLIC_DIR.rglob("*") if p.is_file())
    print(f"[mv2_data_05_validate] OK — {n} arquivos públicos leves; fail-closed preservado")
    return 0


if __name__ == "__main__":
    sys.exit(main())
