"""MV2-DATA-02 Tarefa 9 — Matriz de risco do harness de API/raster.

Registra os riscos metodológicos e operacionais de mexer com API real e raster, com a
mitigação fail-closed correspondente. Garante que todos os RISK_TYPES obrigatórios estejam
presentes.

Saída: outputs_public/mv2_data_api_raster_harness/mv2_data_02_api_raster_risk_matrix.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_data_02_api_config import load_config
from mv2_data_02_common import (
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

# (severity, evidence, mitigation)
RISK_SPEC = {
    "RISK_API_WITHOUT_AUTH": (
        "ALTA",
        "chamar API autenticada sem credencial valida",
        "credencial so por env var; sem credencial -> NOT_EXECUTED_AUTH_MISSING",
    ),
    "RISK_NETWORK_CALL_UNCONTROLLED": (
        "ALTA",
        "chamada de rede disparada sem controle de config",
        "allow_network=false por padrao; toda chamada gated por config",
    ),
    "RISK_STAC_WITHOUT_LINEAGE": (
        "MEDIA",
        "consultar/promover STAC sem scene_id/datetime confiavel",
        "lineage forte exigido antes de promover; metadata-only por padrao",
    ),
    "RISK_ODATA_DOWNLOAD_WITHOUT_REVIEW": (
        "ALTA",
        "baixar produto OData sem revisao do binding",
        "cliente OData nunca baixa; download so no executor canary com gates",
    ),
    "RISK_RASTER_IN_OUTPUTS_PUBLIC": (
        "ALTA",
        "raster gravado em outputs_public",
        "raster so em diretorio privado git-ignored; validador detecta vazamento",
    ),
    "RISK_BULK_DOWNLOAD_TOO_EARLY": (
        "ALTA",
        "download em lote dos 128 targets antes de lineage forte",
        "can_promote_to_bulk_download=false sempre neste marco",
    ),
    "RISK_CANARY_WITHOUT_STRONG_LINEAGE": (
        "ALTA",
        "canary sobre target com lineage fraco/ausente",
        "canary_allowed_by_data=true so com scene_id+datetime+crs",
    ),
    "RISK_CROP_WITHOUT_CRS": (
        "ALTA",
        "crop sem CRS/geotransform confiavel",
        "crop_allowed_now=false por padrao; CRS validado no intake privado",
    ),
    "RISK_MISSING_SCL": (
        "MEDIA",
        "crop sem banda SCL (mascara de cena)",
        "SCL exigida no plano e validada no intake privado",
    ),
    "RISK_PNG_AS_RASTER": (
        "ALTA",
        "PNG/NPZ/render tratado como raster Sentinel nativo",
        "nenhum raster nativo criado neste marco; intake valida extensao/CRS",
    ),
    "RISK_DAY10_FALSE_UNLOCK": (
        "ALTA",
        "tratar o harness como se o Dia 10 estivesse desbloqueado",
        "can_unlock_day10_now=false; so raster privado validado desbloqueia",
    ),
    "RISK_SECRET_LEAK": (
        "ALTA",
        "token/segredo gravado em arquivo publico",
        "config publica so com flags+nomes de env var; chaves proibidas filtradas",
    ),
    "RISK_PRIVATE_PATH_LEAK": (
        "MEDIA",
        "caminho absoluto privado de usuario exposto em outputs_public",
        "validador rejeita prefixo de home de usuario em qualquer arquivo publico",
    ),
    "RISK_COST_QUOTA_RATE_LIMIT": (
        "MEDIA",
        "custo/quota/rate-limit ao consultar API em lote",
        "metadata-only opt-in; canary limitado a <=3 alvos e max_download_mb",
    ),
}


def build_rows() -> list[dict[str, object]]:
    cfg = load_config()
    network_on = bool(cfg.get("allow_network"))
    rows: list[dict[str, object]] = []
    for i, rt in enumerate(RISK_TYPES, start=1):
        severity, evidence, mitigation = RISK_SPEC[rt]
        # Em modo fail-closed (rede off) a maioria está MITIGADO_FAIL_CLOSED.
        status = "ATIVO_REQUER_ATENCAO" if network_on and rt in {
            "RISK_NETWORK_CALL_UNCONTROLLED", "RISK_COST_QUOTA_RATE_LIMIT"
        } else "MITIGADO_FAIL_CLOSED"
        rows.append(
            {
                "risk_id": f"MV2_DATA_02_RISK_{i:02d}",
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
    out = PUBLIC_DIR / "mv2_data_02_api_raster_risk_matrix.csv"
    write_csv(out, COLUMNS, rows)
    present = {str(r["risk_type"]) for r in rows}
    missing = [rt for rt in RISK_TYPES if rt not in present]
    assert not missing, f"risk_types obrigatorios ausentes: {missing}"
    print(
        f"[mv2_data_02_risk] {len(rows)} riscos | tipos={len(present)}/{len(RISK_TYPES)} | "
        f"saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
