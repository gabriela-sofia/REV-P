"""MV2-DATA-04 Tarefa 9 — Matriz de risco do corpus metadata probe.

Registra os riscos de mirar o corpus em modo metadata-only, com a mitigação fail-closed
correspondente. Garante todos os RISK_TYPES obrigatórios.

Saída: outputs_public/mv2_data_corpus_metadata_probe/mv2_data_04_risk_matrix.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_data_04_common import (
    PROJECT_ROOT,
    PUBLIC_DIR,
    RISK_TYPES,
    ensure_public_dir,
    write_csv,
)

COLUMNS = [
    "risk_id",
    "target_id",
    "severity",
    "risk_type",
    "evidence",
    "mitigation",
    "status",
    "blocks_next_step",
]

RISK_SPEC = {
    "RISK_METADATA_PROBE_WITHOUT_TEMPORAL_WINDOW": (
        "ALTA", "executar probe GEE/STAC sem janela temporal por target",
        "sem janela -> BLOCKED_NO_TEMPORAL_WINDOW; probe nunca executa sem filterDate"),
    "RISK_OPEN_ENDED_STAC_QUERY": (
        "ALTA", "STAC Item Search com datetime aberto/amplo",
        "datetime fechado obrigatorio; query aberta ampla nunca emitida"),
    "RISK_CITY_TILE_MATCH": (
        "ALTA", "casar cena por cidade/tile/regiao em vez de lineage real",
        "cidade/tile/regiao nunca e vinculo forte; so scene_id/datetime documentado"),
    "RISK_CANARY_AS_CORPUS": (
        "ALTA", "usar candidate canary Protocolo-C como target do corpus",
        "batch exclui official canary; validador rejeita refpatch/anchor no batch"),
    "RISK_ODATA_WITHOUT_PRODUCT_ID": (
        "ALTA", "tentar resolver OData sem product_id/scene_id",
        "sem product_id -> BLOCKED_NO_PRODUCT_ID; OData nunca itera o corpus"),
    "RISK_SECRET_LEAK": (
        "ALTA", "token/credencial gravado em outputs_public",
        "credencial so por env var; secret_value_logged=false; validador busca segredo"),
    "RISK_PRIVATE_PATH_LEAK": (
        "MEDIA", "caminho absoluto privado em outputs_public",
        "validador rejeita prefixo de home de usuario em arquivo publico"),
    "RISK_RASTER_DOWNLOAD_TOO_EARLY": (
        "ALTA", "baixar raster/crop antes de lineage forte do corpus",
        "marco metadata-only; downloads/crops/raster sempre 0; raster bloqueado"),
    "RISK_DAY10_FALSE_UNLOCK": (
        "ALTA", "tratar probe metadata como desbloqueio do Dia 10",
        "corpus_day10_unlocked=false; so lineage forte + raster validado desbloqueia"),
    "RISK_SANDBOX_FALSE_UNLOCK": (
        "ALTA", "abrir sandbox/treino com base no probe",
        "sandbox_unlocked=false; can_train=false"),
}


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i, rt in enumerate(RISK_TYPES, start=1):
        severity, evidence, mitigation = RISK_SPEC[rt]
        rows.append({
            "risk_id": f"MV2_DATA_04_RISK_{i:02d}",
            "target_id": "GLOBAL",
            "severity": severity,
            "risk_type": rt,
            "evidence": evidence,
            "mitigation": mitigation,
            "status": "MITIGADO_FAIL_CLOSED",
            "blocks_next_step": "true",
        })
    return rows


def main() -> None:
    ensure_public_dir()
    rows = build_rows()
    out = PUBLIC_DIR / "mv2_data_04_risk_matrix.csv"
    write_csv(out, COLUMNS, rows)
    present = {str(r["risk_type"]) for r in rows}
    missing = [rt for rt in RISK_TYPES if rt not in present]
    assert not missing, f"risk_types obrigatorios ausentes: {missing}"
    print(
        f"[mv2_data_04_risk] {len(rows)} riscos | tipos={len(present)}/{len(RISK_TYPES)} | "
        f"saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
