"""MV2-12 — Tabela de prontidão de download (sem baixar nada).

Lê o backlog externo do MV2-10 (mv2_10_external_data_backlog.csv, presente localmente)
e o transforma numa tabela de readiness de download: acesso, licença, tamanho estimado,
se deve baixar agora, diretório-alvo local-only e se pode ir para outputs_public.

Guardrails:
- publish_to_outputs_public = false para TODO bruto pesado;
- download_target_dir aponta sempre para diretório local-only/quarentena (git-ignored);
- nada é efetivamente baixado por este script.

Saída: outputs_public/mv2_data_readiness/mv2_12_download_readiness.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_12_data_readiness_common import (
    EXTERNAL_LOCAL_DIR_REL,
    MV2_10_EXTERNAL_BACKLOG,
    OUTPUT_DIR,
    PROJECT_ROOT,
    QUARANTINE_DIR_REL,
    ensure_output_dir,
    read_csv_rows,
    write_csv,
)

COLUMNS = [
    "source_name",
    "source_type",
    "region",
    "url_or_portal",
    "expected_file",
    "expected_format",
    "access_status",
    "license_or_terms_status",
    "estimated_size_class",
    "should_download_now",
    "download_target_dir",
    "publish_to_outputs_public",
    "checksum_required",
    "reason",
    "risk",
]

# Override por id MV2-10 (campos que exigem julgamento de licença/tamanho/ação).
# size: GB (raster pesado) | MB (vetor/PDF/CSV) | KB (metadado).
OVERRIDES: dict[str, dict[str, str]] = {
    "EXT_001": {"format": "GeoTIFF/JP2 multibanda", "license": "ABERTO_COPERNICUS", "size": "GB", "now": "false", "risk": "pesado; exige scene_id por patch para download auditável"},
    "EXT_002": {"format": "log/CSV de export GEE", "license": "PRIVADO_USUARIA", "size": "KB", "now": "false", "risk": "recuperação manual do histórico GEE"},
    "EXT_003": {"format": "CSV cloud_cover por cena", "license": "ABERTO_COPERNICUS", "size": "KB", "now": "false", "risk": "depende de scene_id"},
    "EXT_004": {"format": "PNG raster (vetor indisponível)", "license": "EMS_TERMOS", "size": "MB", "now": "false", "risk": "vetor exige acesso premium; PNG já em quarentena"},
    "EXT_005": {"format": "CSV lat/lon ou SHP", "license": "LAI_OU_CONTATO", "size": "MB", "now": "false", "risk": "pontos existem mas conflitam; exige revisão humana"},
    "EXT_006": {"format": "CSV/PDF de alertas", "license": "LAI_REQUERIDA", "size": "MB", "now": "false", "risk": "solicitação formal LAI; prazo 30 dias"},
    "EXT_007": {"format": "CSV série por estação", "license": "CONTATO_DIRETO", "size": "MB", "now": "false", "risk": "série não pública; PDF já baixado"},
    "EXT_008": {"format": "CSV série diária/horária", "license": "ABERTO_ANA", "size": "MB", "now": "true", "risk": "download leve público; consulta anterior veio vazia"},
    "EXT_009": {"format": "SHP/GeoJSON cicatrizes", "license": "CONTATO_DRM", "size": "MB", "now": "false", "risk": "cohort mass_movement separado de flood"},
    "EXT_010": {"format": "GeoJSON/SHP recorte municipal", "license": "ABERTO_SGB", "size": "GB", "now": "false", "risk": "SIG completo 1.8GB; recorte ausente"},
    "EXT_011": {"format": "CSV pontos / GeoJSON", "license": "ABERTO_GEOCURITIBA", "size": "MB", "now": "false", "risk": "deep crawl 0 geometria; solicitação formal pronta"},
    "EXT_012": {"format": "GRD backscatter GeoTIFF", "license": "ABERTO_COPERNICUS", "size": "GB", "now": "false", "risk": "exige processamento de backscatter"},
    "EXT_013": {"format": "raster/CSV grade 1km", "license": "LICENCA_A_VERIFICAR", "size": "MB", "now": "false", "risk": "licença grade 2022 a reverificar antes de baixar"},
    "EXT_014": {"format": "SHP/GeoJSON landslide scars", "license": "ACESSO_PAGO", "size": "MB", "now": "false", "risk": "produto vetorial pago ou organização elegível"},
    "EXT_015": {"format": "raster cobertura 30m", "license": "ABERTO_MAPBIOMAS", "size": "GB", "now": "false", "risk": "raster por patch via GEE; XLSX já baixado"},
}


def _target_dir(now: str) -> str:
    # Bruto baixável agora vai para quarentena; o resto fica catalogado em external_data_local.
    if now == "true":
        return QUARANTINE_DIR_REL
    return EXTERNAL_LOCAL_DIR_REL


def build_rows() -> list[dict[str, str]]:
    ext_rows = read_csv_rows(MV2_10_EXTERNAL_BACKLOG)
    rows: list[dict[str, str]] = []
    for src in ext_rows:
        ext_id = src.get("id", "")
        ov = OVERRIDES.get(ext_id, {})
        size = ov.get("size", "MB")
        now = ov.get("now", "false")
        license_status = ov.get("license", "A_VERIFICAR")
        # Bruto pesado NUNCA é publicado em outputs_public.
        publish = "false"
        rows.append(
            {
                "source_name": f"{src.get('instituicao', '')} — {ext_id}",
                "source_type": src.get("fonte_tipo", ""),
                "region": src.get("regiao", ""),
                "url_or_portal": src.get("url_ou_portal", ""),
                "expected_file": src.get("nome_arquivo_necessario", "")[:160],
                "expected_format": ov.get("format", ""),
                "access_status": src.get("status_atual", "")[:120],
                "license_or_terms_status": license_status,
                "estimated_size_class": size,
                "should_download_now": now,
                "download_target_dir": _target_dir(now),
                "publish_to_outputs_public": publish,
                "checksum_required": "true",
                "reason": src.get("obs", "")[:160],
                "risk": ov.get("risk", ""),
            }
        )
    return rows


def main() -> None:
    ensure_output_dir()
    rows = build_rows()
    out = OUTPUT_DIR / "mv2_12_download_readiness.csv"
    write_csv(out, COLUMNS, rows)
    n_now = sum(1 for r in rows if r["should_download_now"] == "true")
    n_pub = sum(1 for r in rows if r["publish_to_outputs_public"] == "true")
    print(f"[mv2_12_download] fontes: {len(rows)} | baixar agora: {n_now} | publicar bruto: {n_pub}")
    print(f"[mv2_12_download] saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
