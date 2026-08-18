"""MV2-DATA-03 Tarefa 12 — Validador geral do live API/canary.

Sai com código != 0 se qualquer trava fail-closed for violada: segredo/raster/caminho
privado em outputs_public; mais de 1 canary executado; bulk download; algum dos 128 corpus
targets baixado; canary desbloqueando Dia 10/sandbox/feature/label; can_use_for_corpus_day10
!= false; raster baixado fora de local privado; summary incoerente.
"""

from __future__ import annotations

import json
import re
import sys

from mv2_data_03_common import (
    MAX_CANARY_EXECUTIONS,
    PROJECT_ROOT,
    PUBLIC_DIR,
    RISK_TYPES,
    clean,
    corpus_patch_ids,
    is_private_path,
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
# token=valor numa URL — exceto o marcador de redação.
URL_TOKEN = re.compile(r"(?i)(token|access_token|api_key|password)=(?!__REDACTED__)[^&\s]+")

IMPACT_REQUIRED = [
    "API auth validated", "metadata probe validated", "raster private path validated",
    "checksum validation", "band/SCL validation", "corpus lineage still missing",
    "Dia 10 still blocked", "sandbox still blocked",
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
            if URL_TOKEN.search(txt):
                v.append(f"token em URL em {p.name}")

    candidates = read_csv_rows(PUBLIC_DIR / "mv2_data_03_official_canary_candidates.csv")
    corpus = corpus_patch_ids()
    if not candidates:
        v.append("official canary candidates ausentes/vazios")
    for c in candidates:
        if clean(c.get("can_use_for_corpus_day10")) != "false":
            v.append(f"candidate can_use_for_corpus_day10!=false: {clean(c.get('canary_candidate_id'))}")
        if clean(c.get("canary_role")) != "TECHNICAL_CANARY_ONLY":
            v.append(f"candidate sem TECHNICAL_CANARY_ONLY: {clean(c.get('canary_candidate_id'))}")
        if clean(c.get("official_refpatch_id")) in corpus:
            v.append(f"candidate é corpus target: {clean(c.get('canary_candidate_id'))}")
        if clean(c.get("corpus_patch_id")):
            v.append(f"candidate vinculado a corpus_patch_id: {clean(c.get('canary_candidate_id'))}")

    # Consenso: can_use_for_corpus_day10=false sempre.
    consensus = read_csv_rows(PUBLIC_DIR / "mv2_data_03_canary_metadata_consensus.csv")
    for r in consensus:
        if clean(r.get("can_use_for_corpus_day10")) != "false":
            v.append(f"consenso can_use_for_corpus_day10!=false: {clean(r.get('consensus_id'))}")

    # Manifest: no máximo 1 canary; se executado, caminho privado; nunca público.
    manifest = read_csv_rows(PUBLIC_DIR / "mv2_data_03_private_raster_canary_execution_manifest.csv")
    if len(manifest) > MAX_CANARY_EXECUTIONS:
        v.append(f"mais de {MAX_CANARY_EXECUTIONS} canary no manifest: {len(manifest)}")
    executed_count = 0
    for m in manifest:
        if clean(m.get("executed")) == "true":
            executed_count += 1
            ref = clean(m.get("private_output_ref"))
            if not is_private_path(ref):
                v.append(f"canary executado fora de diretório privado: {clean(m.get('execution_id'))}")
            if clean(m.get("public_output_created")) == "true":
                v.append(f"canary criou saída pública: {clean(m.get('execution_id'))}")
        # canary candidate nunca pode ser corpus
        cid = clean(m.get("canary_candidate_id"))
        cand = next((c for c in candidates if clean(c.get("canary_candidate_id")) == cid), {})
        if clean(cand.get("official_refpatch_id")) in corpus:
            v.append(f"manifest aponta para corpus target: {clean(m.get('execution_id'))}")
    if executed_count > MAX_CANARY_EXECUTIONS:
        v.append(f"mais de {MAX_CANARY_EXECUTIONS} canary executado(s): {executed_count}")

    # Validação privada: nenhum vazamento público.
    validation = read_csv_rows(PUBLIC_DIR / "mv2_data_03_private_raster_canary_validation.csv")
    for r in validation:
        if clean(r.get("public_leak_detected")) == "true":
            v.append(f"vazamento público de raster: {clean(r.get('validation_id'))}")

    # Risk matrix completa.
    present_risk = {clean(r.get("risk_type")) for r in read_csv_rows(PUBLIC_DIR / "mv2_data_03_live_api_canary_risk_matrix.csv")}
    for rt in RISK_TYPES:
        if rt not in present_risk:
            v.append(f"risk_type obrigatório ausente: {rt}")

    # Corpus impact matrix: linhas obrigatórias + nunca afeta corpus.
    impact = read_csv_rows(PUBLIC_DIR / "mv2_data_03_corpus_impact_matrix.csv")
    comps = {clean(r.get("component")) for r in impact}
    for comp in IMPACT_REQUIRED:
        if comp not in comps:
            v.append(f"linha obrigatória ausente na corpus impact matrix: {comp}")
    for r in impact:
        if clean(r.get("affects_128_targets")) != "false" or clean(r.get("affects_day10")) != "false" \
           or clean(r.get("affects_sandbox")) != "false":
            v.append(f"impact matrix afeta corpus/day10/sandbox: {clean(r.get('impact_id'))}")

    # Summary coerente e fail-closed.
    summary_path = PUBLIC_DIR / "mv2_data_03_summary.json"
    if not summary_path.exists():
        return v + ["summary JSON ausente"]
    s = json.loads(summary_path.read_text(encoding="utf-8"))
    for key in ["crops_created", "native_rasters_created", "spectral_features_created",
                "labels_created", "silver_created", "gold_created", "negatives_created",
                "public_raster_leaks", "corpus_targets_downloaded"]:
        if s.get(key, -1) != 0:
            v.append(f"guardrail violado: {key}={s.get(key)} (deve ser 0)")
    if s.get("can_train") is not False:
        v.append("guardrail violado: can_train deve ser false")
    if s.get("corpus_day10_unlocked") is not False:
        v.append("guardrail violado: corpus_day10_unlocked deve ser false")
    if s.get("sandbox_unlocked") is not False:
        v.append("guardrail violado: sandbox_unlocked deve ser false")
    if s.get("private_raster_canary_executed", 0) > MAX_CANARY_EXECUTIONS:
        v.append("guardrail violado: mais de 1 canary executado")
    if s.get("official_canary_candidates") != len(candidates):
        v.append("summary incoerente: official_canary_candidates")
    if s.get("conflicts") != sum(1 for r in consensus if clean(r.get("consensus_status")) == "CONFLICT"):
        v.append("summary incoerente: conflicts")

    return v


def main() -> int:
    v = find_violations()
    if v:
        print("[mv2_data_03_validate] VIOLACOES:")
        for x in v:
            print(f"  - {x}")
        return 1
    n = sum(1 for p in PUBLIC_DIR.rglob("*") if p.is_file())
    print(f"[mv2_data_03_validate] OK — {n} arquivos públicos leves; fail-closed preservado")
    return 0


if __name__ == "__main__":
    sys.exit(main())
