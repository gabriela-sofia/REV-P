"""MV2-DATA-03 Tarefa 7 — Executor opcional de raster canary privado (máximo 1).

Por padrão NÃO executa. Executa no máximo 1 canary somente se TODAS as condições forem
verdadeiras: config local existe; allow_network; allow_raster_download; allow_canary
(REV_P_ALLOW_RASTER_CANARY=YES); consenso forte (API_CONFIRMED_STRONG/REGISTRY_ONLY_STRONG);
private_output_dir válido e fora de outputs_public; tamanho estimado <= max_download_mb;
o alvo é um canary oficial (não um dos 128 do corpus). Senão -> NOT_EXECUTED_GUARDRAIL.

Quando executa: baixa só em diretório privado, calcula SHA256, e grava manifest público
leve (sem caminho absoluto privado, sem token). Nunca publica raster.

Saída pública: outputs_public/mv2_data_live_api_canary/mv2_data_03_private_raster_canary_execution_manifest.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_data_03_common import (
    CANARY_BANDS,
    CANARY_SCL,
    MAX_CANARY_EXECUTIONS,
    PROJECT_ROOT,
    PUBLIC_DIR,
    clean,
    corpus_patch_ids,
    effective_flags,
    ensure_public_dir,
    is_private_path,
    lib_available,
    read_csv_rows,
    write_csv,
)

ESTIMATED_CROP_MB = 18

COLUMNS = [
    "execution_id",
    "canary_candidate_id",
    "executed",
    "provider",
    "scene_id",
    "product_id",
    "private_output_ref",
    "public_output_created",
    "bands_requested",
    "scl_requested",
    "file_size_bytes",
    "sha256",
    "download_started_at",
    "download_finished_at",
    "execution_status",
    "blocking_reason",
    "notes",
]


def _candidates_index() -> dict[str, dict[str, str]]:
    return {clean(r.get("canary_candidate_id")): r
            for r in read_csv_rows(PUBLIC_DIR / "mv2_data_03_official_canary_candidates.csv")}


def _unmet(consensus_row: dict[str, str], candidate: dict[str, str], flags: dict[str, object]) -> list[str]:
    unmet: list[str] = []
    if not flags.get("config_found"):
        unmet.append("config_local_ausente")
    if not flags.get("allow_network"):
        unmet.append("allow_network=false")
    if not flags.get("allow_raster_download"):
        unmet.append("allow_raster_download=false")
    if not flags.get("allow_canary"):
        unmet.append("REV_P_ALLOW_RASTER_CANARY!=YES")
    if clean(consensus_row.get("consensus_status")) not in {"API_CONFIRMED_STRONG", "REGISTRY_ONLY_STRONG"}:
        unmet.append("consenso_nao_forte")
    priv = str(flags.get("private_output_dir"))
    if not is_private_path(priv):
        unmet.append("private_output_dir_invalido")
    if ESTIMATED_CROP_MB > int(str(flags.get("max_download_mb", 0))):
        unmet.append("tamanho_estimado_acima_de_max_download_mb")
    # alvo precisa ser canary oficial (não corpus)
    if clean(candidate.get("canary_role")) != "TECHNICAL_CANARY_ONLY":
        unmet.append("nao_e_canary_oficial")
    if clean(candidate.get("official_refpatch_id")) in corpus_patch_ids():
        unmet.append("alvo_e_corpus_target")
    return unmet


def _attempt_download(candidate: dict[str, str], flags: dict[str, object]) -> dict[str, object]:
    """Caminho real — só alcançado com todos os gates. Conservador e fail-closed."""
    import datetime as _dt
    import hashlib

    odata = {clean(r.get("canary_candidate_id")): r
             for r in read_csv_rows(PUBLIC_DIR / "mv2_data_03_live_cdse_odata_metadata_probe.csv")}
    cid = clean(candidate.get("canary_candidate_id"))
    url = clean(odata.get(cid, {}).get("download_url_resolved"))
    if not url or "__REDACTED__" in url:
        return {"executed": "false", "execution_status": "NOT_EXECUTED_NO_DOWNLOAD_URL",
                "blocking_reason": "OData_nao_resolveu_url_de_download", "notes": "endpoint pendente"}
    if not lib_available("requests"):
        return {"executed": "false", "execution_status": "NOT_EXECUTED_FETCH_DEP_MISSING",
                "blocking_reason": "requests_indisponivel", "notes": ""}
    private_root = PROJECT_ROOT / str(flags["private_output_dir"])
    private_root.mkdir(parents=True, exist_ok=True)
    dest = private_root / (cid + ".bin")
    started = _dt.datetime.now(_dt.timezone.utc).isoformat()
    try:
        import requests

        h = hashlib.sha256()
        size = 0
        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with dest.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    fh.write(chunk)
                    h.update(chunk)
                    size += len(chunk)
        return {
            "executed": "true",
            "execution_status": "EXECUTED_PRIVATE_ONLY",
            "private_output_ref": dest.relative_to(PROJECT_ROOT).as_posix(),
            "file_size_bytes": size,
            "sha256": h.hexdigest(),
            "download_started_at": started,
            "download_finished_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "blocking_reason": "",
            "notes": "raster apenas em diretorio privado git-ignored",
        }
    except Exception as exc:
        return {"executed": "false", "execution_status": "NOT_EXECUTED_FETCH_ERROR",
                "blocking_reason": f"erro:{type(exc).__name__}", "notes": ""}


def build_rows() -> list[dict[str, object]]:
    flags = effective_flags()
    consensus = read_csv_rows(PUBLIC_DIR / "mv2_data_03_canary_metadata_consensus.csv")
    candidates = _candidates_index()

    # Seleciona no máximo 1: o primeiro com consenso forte (data_ready).
    chosen = None
    for cons in consensus:
        if clean(cons.get("canary_download_data_ready")) == "true":
            chosen = cons
            break
    if chosen is None and consensus:
        chosen = consensus[0]
    if chosen is None:
        return []

    cid = clean(chosen.get("canary_candidate_id"))
    candidate = candidates.get(cid, {})
    unmet = _unmet(chosen, candidate, flags)

    base: dict[str, object] = {
        "execution_id": "MV2_DATA_03_EXEC_01",
        "canary_candidate_id": cid,
        "provider": "CDSE_ODATA",
        "scene_id": clean(chosen.get("scene_id")),
        "product_id": clean(chosen.get("product_id")),
        "private_output_ref": "",
        "public_output_created": "false",  # NUNCA cria raster público
        "bands_requested": ",".join(CANARY_BANDS),
        "scl_requested": CANARY_SCL,
        "file_size_bytes": "",
        "sha256": "",
        "download_started_at": "",
        "download_finished_at": "",
    }
    if unmet:
        base.update({
            "executed": "false",
            "execution_status": "NOT_EXECUTED_GUARDRAIL",
            "blocking_reason": "|".join(unmet),
            "notes": "nenhum raster baixado; nenhuma chamada de rede; max 1 canary",
        })
    else:
        base.update(_attempt_download(candidate, flags))
    return [base]  # garante no máximo MAX_CANARY_EXECUTIONS (=1) entradas


def main() -> None:
    ensure_public_dir()
    rows = build_rows()
    assert len(rows) <= MAX_CANARY_EXECUTIONS, "mais de 1 canary no manifest"
    out = PUBLIC_DIR / "mv2_data_03_private_raster_canary_execution_manifest.csv"
    write_csv(out, COLUMNS, rows)
    executed = sum(1 for r in rows if r["executed"] == "true")
    print(
        f"[mv2_data_03_canary_exec] {len(rows)} entrada(s) | executados={executed} | "
        f"saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
