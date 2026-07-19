"""MV2-DATA-03 Tarefa 10 — Matriz de risco do live API/canary.

Registra os riscos de operar API real + canary privado, com mitigação fail-closed. Garante
todos os RISK_TYPES obrigatórios.

Saída: outputs_public/mv2_data_live_api_canary/mv2_data_03_live_api_canary_risk_matrix.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_data_03_common import (
    PROJECT_ROOT,
    PUBLIC_DIR,
    RISK_TYPES,
    effective_flags,
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
    "RISK_SECRET_LEAK": (
        "ALTA", "token/credencial gravado em arquivo ou log",
        "credencial so por env var; nenhum valor impresso/gravado; validador busca segredo"),
    "RISK_PRIVATE_PATH_LEAK": (
        "MEDIA", "caminho absoluto privado exposto em outputs_public",
        "private_output_ref sanitizado relativo; validador rejeita home de usuario"),
    "RISK_RASTER_PUBLIC_LEAK": (
        "ALTA", "raster gravado/copiado para outputs_public",
        "raster so em diretorio privado git-ignored; validador detecta vazamento"),
    "RISK_CANARY_AS_CORPUS_UNLOCK": (
        "ALTA", "tratar canary oficial como desbloqueio do corpus",
        "can_use_for_corpus_day10=false; matriz de impacto formaliza separacao"),
    "RISK_BULK_DOWNLOAD_ACCIDENT": (
        "ALTA", "download acidental dos 128 targets do corpus",
        "executor so aceita canary oficial; nunca itera o corpus; max 1 execucao"),
    "RISK_STAC_ASSET_DOWNLOAD_TOO_EARLY": (
        "MEDIA", "baixar assets STAC antes de lineage/consenso forte",
        "STAC probe e metadata-only; assets nunca baixados no probe"),
    "RISK_GEE_EXPORT_TOO_EARLY": (
        "MEDIA", "exportar imagem GEE prematuramente",
        "GEE probe metadata-only; nenhum Export.image"),
    "RISK_TILE_AUTO_JOIN": (
        "ALTA", "auto-vincular T23KPR a PET_* do corpus",
        "tile vem do scene_id oficial documentado; canary nao e patch do corpus"),
    "RISK_CREDENTIAL_LOGGING": (
        "ALTA", "imprimir/logar valores de env vars de credencial",
        "secret_value_logged=false sempre; so presenca booleana e registrada"),
    "RISK_COST_QUOTA_RATE_LIMIT": (
        "MEDIA", "custo/quota/rate-limit em chamadas de API",
        "probes limitados a <=2 candidates, timeout curto, opt-in"),
    "RISK_DAY10_FALSE_UNLOCK": (
        "ALTA", "desbloquear Dia 10 com base no canary",
        "corpus_day10_unlocked=false; so lineage+raster do corpus validado desbloqueia"),
    "RISK_SANDBOX_FALSE_UNLOCK": (
        "ALTA", "abrir sandbox/treino com base no canary",
        "sandbox_unlocked=false; can_train=false"),
}


def build_rows() -> list[dict[str, object]]:
    flags = effective_flags()
    network_on = bool(flags.get("allow_network"))
    rows: list[dict[str, object]] = []
    for i, rt in enumerate(RISK_TYPES, start=1):
        severity, evidence, mitigation = RISK_SPEC[rt]
        status = "ATIVO_REQUER_ATENCAO" if network_on and rt in {
            "RISK_COST_QUOTA_RATE_LIMIT", "RISK_STAC_ASSET_DOWNLOAD_TOO_EARLY",
            "RISK_GEE_EXPORT_TOO_EARLY",
        } else "MITIGADO_FAIL_CLOSED"
        rows.append(
            {
                "risk_id": f"MV2_DATA_03_RISK_{i:02d}",
                "target_id": "GLOBAL",
                "severity": severity,
                "risk_type": rt,
                "evidence": evidence,
                "mitigation": mitigation,
                "status": status,
                "blocks_next_step": "true",
            }
        )
    return rows


def main() -> None:
    ensure_public_dir()
    rows = build_rows()
    out = PUBLIC_DIR / "mv2_data_03_live_api_canary_risk_matrix.csv"
    write_csv(out, COLUMNS, rows)
    present = {str(r["risk_type"]) for r in rows}
    missing = [rt for rt in RISK_TYPES if rt not in present]
    assert not missing, f"risk_types obrigatorios ausentes: {missing}"
    print(
        f"[mv2_data_03_risk] {len(rows)} riscos | tipos={len(present)}/{len(RISK_TYPES)} | "
        f"saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
