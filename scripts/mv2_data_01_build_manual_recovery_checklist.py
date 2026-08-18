"""MV2-DATA-01 Tarefa 5 — Checklist manual de recuperação por target.

Gera, para cada target, a sequência de passos manuais que um humano deve seguir no GEE
para recuperar o lineage, com a evidência esperada, o formato aceito, a regra de rejeição
e se o passo bloqueia o Dia 10. Nada é executado; é um roteiro de revisão humana.

Saída: outputs_public/mv2_data_gee_lineage_targets/mv2_data_01_manual_recovery_checklist.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_data_01_common import (
    OUTPUT_DIR,
    PROJECT_ROOT,
    build_targets,
    ensure_output_dir,
    write_csv,
)

COLUMNS = [
    "checklist_id",
    "target_id",
    "asset_id",
    "patch_id",
    "region",
    "step_order",
    "manual_step",
    "expected_evidence",
    "accepted_value_format",
    "rejection_rule",
    "blocks_day10",
    "notes",
]

# (manual_step, expected_evidence, accepted_value_format, rejection_rule, blocks_day10)
STEPS = [
    (
        "abrir o GEE Code Editor / Tasks autenticado",
        "sessao_GEE_ativa",
        "screenshot_ou_url_de_sessao",
        "sem_acesso_ao_GEE_nao_prossegue",
        "true",
    ),
    (
        "localizar o script/export task original deste patch",
        "nome_do_script_ou_export_task",
        "texto_export_task_name_ou_caminho_de_script",
        "nao_inferir_export_por_nome_de_arquivo_de_patch",
        "true",
    ),
    (
        "localizar a imagem Sentinel-2 efetivamente usada",
        "image_id_da_colecao_S2",
        "ee_image_id_ou_system_index",
        "nao_assumir_imagem_por_cidade_regiao_ou_tile",
        "true",
    ),
    (
        "copiar o PRODUCT_ID",
        "PRODUCT_ID_da_imagem",
        "S2[AB]_MSIL2A_AAAAMMDDTHHMMSS_..._TXXYYY_AAAAMMDDTHHMMSS",
        "nao_aceitar_PRODUCT_ID_sem_origem_GEE",
        "true",
    ),
    (
        "copiar o acquisition datetime",
        "system_time_start_convertido_para_ISO",
        "AAAA-MM-DDTHH:MM:SS",
        "rejeitar_datetime_futuro_ou_fora_de_2015_2024",
        "true",
    ),
    (
        "copiar o MGRS_TILE",
        "MGRS_TILE_da_imagem",
        "TXXLLL_ex_T25M...",
        "nao_usar_T23KPR_nem_tile_por_zona_como_vinculo_automatico",
        "true",
    ),
    (
        "copiar o CLOUDY_PIXEL_PERCENTAGE",
        "CLOUDY_PIXEL_PERCENTAGE",
        "float_0_a_100",
        "nao_inventar_cobertura_de_nuvem",
        "false",
    ),
    (
        "copiar o DATATAKE_IDENTIFIER",
        "DATATAKE_IDENTIFIER",
        "GS2[AB]_AAAAMMDDTHHMMSS_NNNNNN_NXXYY",
        "nao_inferir_datatake_de_outro_produto",
        "false",
    ),
    (
        "registrar a collection usada",
        "id_da_colecao_GEE",
        "COPERNICUS/S2_SR_HARMONIZED_ou_equivalente",
        "nao_assumir_colecao_sem_evidencia",
        "false",
    ),
    (
        "registrar a evidencia local (export/print/log)",
        "referencia_local_da_evidencia",
        "caminho_local_relativo_ou_id_de_evidencia",
        "nao_publicar_raster_bruto_como_evidencia",
        "false",
    ),
    (
        "marcar review_status (CONFIRMED/PARTIAL/REJECTED)",
        "review_status_preenchido",
        "CONFIRMED|PARTIAL|REJECTED",
        "sem_revisor_nao_marcar_CONFIRMED",
        "false",
    ),
    (
        "nao preencher nada se nao houver evidencia",
        "campos_vazios_quando_sem_evidencia",
        "vazio",
        "ausencia_de_evidencia_nunca_vira_valor_inventado_nem_negativo",
        "true",
    ),
]


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for t in build_targets():
        tid = str(t["target_id"])
        for order, step in enumerate(STEPS, start=1):
            manual_step, evidence, fmt, rejection, blocks = step
            rows.append(
                {
                    "checklist_id": f"{tid}_STEP_{order:02d}",
                    "target_id": tid,
                    "asset_id": t["asset_id"],
                    "patch_id": t["patch_id"],
                    "region": t["region"],
                    "step_order": order,
                    "manual_step": manual_step,
                    "expected_evidence": evidence,
                    "accepted_value_format": fmt,
                    "rejection_rule": rejection,
                    "blocks_day10": blocks,
                    "notes": "passo_de_revisao_humana;nenhuma_execucao_automatica",
                }
            )
    return rows


def main() -> None:
    ensure_output_dir()
    rows = build_rows()
    out = OUTPUT_DIR / "mv2_data_01_manual_recovery_checklist.csv"
    write_csv(out, COLUMNS, rows)
    print(
        f"[mv2_data_01_checklist] {len(rows)} passos ({len(STEPS)}/target) | "
        f"saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
