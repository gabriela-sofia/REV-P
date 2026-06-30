"""SUSC-17C10 pacote de solicitacao formal.

Gera rascunhos e manifests review-only para acao manual futura. Nao envia
mensagens, nao abre protocolo, nao baixa dados, nao executa pipelines pesados e
nao cria features reais.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

from susc_io import read_csv, read_json, rel, write_csv, write_json, write_markdown  # noqa: E402

DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

FEATURES = DAT / "susc_features_by_patch_v1.csv"
SCORE_V6 = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"

AUDIT = OUT / "SUSC_WORKTREE_AUDIT_BEFORE_17C10.md"

C9_INPUTS = [
    OUT / "susc_17c9_source_artifact_inventory.csv",
    OUT / "susc_17c9_pipeline_reuse_audit.csv",
    OUT / "susc_17c9_candidate_patch_input_requirements.csv",
    OUT / "susc_17c9_external_artifact_request_plan.csv",
    OUT / "susc_17c9_lightweight_export_plan.csv",
    OUT / "susc_17c9_embedding_tile_requirement_plan.csv",
    OUT / "susc_17c9_feature_extraction_unblock_matrix.csv",
    OUT / "susc_17c9_no_fake_feature_policy.json",
    OUT / "susc_17c9_readiness_summary.json",
    OUT / "susc_17c9_promotion_blockers.csv",
]
C8_INPUTS = [
    OUT / "susc_17c8_extraction_run_registry.csv",
    OUT / "susc_17c8_feature_quality_flags.csv",
    OUT / "susc_17c8_feature_provenance_manifest.csv",
    OUT / "susc_17c8_readiness_summary.json",
    OUT / "susc_17c8_promotion_blockers.csv",
]
C4_INPUTS = [
    OUT / "susc_17c4_artifact_inventory.csv",
    OUT / "susc_17c4_formal_request_queue.csv",
    OUT / "susc_17c4_extracted_reference_candidates.csv",
    OUT / "susc_17c4_summary.json",
]
C3_INPUTS = [
    OUT / "susc_17c3_official_source_acquisition_targets.csv",
    OUT / "susc_17c3_missing_official_artifacts_manifest.csv",
    OUT / "susc_17c3_next_action_decision_table.csv",
    OUT / "susc_17c3_summary.json",
]

TEMPLATE_REF = ROOT / "docs" / "templates" / "protocolo_c_solicitacao_fonte_observacional.md"

REPORT = OUT / "SUSC_17C10_PACOTE_SOLICITACAO_FORMAL_REPORT.md"
REGISTRY = OUT / "susc_17c10_formal_request_registry.csv"
PROVIDER_INDEX = OUT / "susc_17c10_provider_package_index.csv"
REQUIRED_FIELDS = OUT / "susc_17c10_required_fields_by_request.csv"
TRACEABILITY = OUT / "susc_17c10_request_to_blocker_traceability.csv"
IMPACT_MATRIX = OUT / "susc_17c10_request_impact_matrix.csv"
CHECKLIST = OUT / "susc_17c10_manual_submission_checklist.csv"
MESSAGE_TEMPLATES = OUT / "susc_17c10_request_message_templates.md"
ATTACHMENTS = OUT / "susc_17c10_attachment_manifest.csv"
RESPONSE_INTAKE = OUT / "susc_17c10_response_intake_schema_plan.csv"
SUMMARY = OUT / "susc_17c10_readiness_summary.json"
BLOCKERS = OUT / "susc_17c10_promotion_blockers.csv"

REQUEST_SCHEMA = SCHEMAS / "susc_17c10_formal_request_schema_v1.json"
IMPACT_SCHEMA = SCHEMAS / "susc_17c10_request_impact_schema_v1.json"

REQUIRED_INPUTS = [FEATURES, SCORE_V6, TEMPLATE_REF, *C9_INPUTS, *C8_INPUTS, *C4_INPUTS, *C3_INPUTS]
REQUIRED_OUTPUTS = [
    AUDIT,
    REPORT,
    REGISTRY,
    PROVIDER_INDEX,
    REQUIRED_FIELDS,
    TRACEABILITY,
    IMPACT_MATRIX,
    CHECKLIST,
    MESSAGE_TEMPLATES,
    ATTACHMENTS,
    RESPONSE_INTAKE,
    SUMMARY,
    BLOCKERS,
    REQUEST_SCHEMA,
    IMPACT_SCHEMA,
]

GOV = {"review_only": "true", "trainable": "false", "ground_truth": "false"}


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _run_git(args: list[str]) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def _require_inputs() -> None:
    missing = [p for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError("; ".join(rel(p) for p in missing))
    for path in C9_INPUTS + C8_INPUTS + C4_INPUTS + C3_INPUTS:
        if path.suffix == ".json":
            read_json(path)
        else:
            read_csv(path)


def request_specs() -> list[dict]:
    specs = [
        {
            "formal_request_id": "P0_DRM_RJ_PKG_FR_PET_001",
            "priority": "P0",
            "provider_name": "DRM-RJ/NADE",
            "provider_type": "servico_geologico",
            "city_or_region": "Petropolis",
            "related_event_id": "PET_2022_02_15",
            "related_package_id": "PKG_FR_PET_001",
            "request_title": "Solicitacao de geometria oficial do evento Petropolis 2022",
            "artifact_requested": "geometria oficial, ponto, poligono, endereco georreferenciado ou tabela de ocorrencias do pacote PKG_FR_PET_001",
            "artifact_type_requested": "official_observed_event_geometry",
            "minimum_required_fields": "data_do_evento;tipo_do_fenomeno;geometria_ou_coordenada;crs;incerteza_ou_escala;fonte_responsavel",
            "why_needed": "destravar evidencia observacional oficial para avaliacao patch-level review-only",
            "revp_use_case": "evidencia_observacional_petropolis_2022",
            "unblocks_feature_layer": "observational_evidence",
            "unblocks_observational_evidence": True,
            "unblocks_candidate_patch_features": False,
            "unblocks_embedding_input": False,
            "unblocks_qa": True,
            "unblocks_17b_future": True,
            "package_group": "DRM-RJ / Petropolis 2022",
        },
        {
            "formal_request_id": "P0_COMPDEC_RECIFE_PKG_FR_REC_002",
            "priority": "P0",
            "provider_name": "COMPDEC/Defesa Civil PE",
            "provider_type": "defesa_civil",
            "city_or_region": "Recife",
            "related_event_id": "REC_2022_05_24_30",
            "related_package_id": "PKG_FR_REC_002",
            "request_title": "Solicitacao de pontos, enderecos ou geometria oficial do evento Recife maio 2022",
            "artifact_requested": "pontos, enderecos georreferenciados, setores atingidos ou geometria oficial do pacote PKG_FR_REC_002",
            "artifact_type_requested": "official_observed_event_geometry_or_table",
            "minimum_required_fields": "data_do_evento;tipo_do_fenomeno;coordenada_ou_endereco;bairro;crs_se_houver;fonte_responsavel",
            "why_needed": "complementar o footprint coarse do International Charter com evidencia oficial local",
            "revp_use_case": "evidencia_observacional_recife_2022",
            "unblocks_feature_layer": "observational_evidence;qa",
            "unblocks_observational_evidence": True,
            "unblocks_candidate_patch_features": False,
            "unblocks_embedding_input": False,
            "unblocks_qa": True,
            "unblocks_17b_future": True,
            "package_group": "COMPDEC Recife / Recife maio 2022",
        },
        {
            "formal_request_id": "P1_SGB_CPRM_VECTOR_OR_COORDINATES",
            "priority": "P1",
            "provider_name": "SGB/CPRM",
            "provider_type": "servico_geologico",
            "city_or_region": "Petropolis",
            "related_event_id": "PET_2022_02_15",
            "related_package_id": "SGB_CPRM_PDF_ZIP_SEM_VETOR",
            "request_title": "Solicitacao de vetores ou coordenadas associados a PDF/ZIP SGB/CPRM",
            "artifact_requested": "vetor, tabela de coordenadas ou legenda georreferenciada dos anexos e relatorio tecnico",
            "artifact_type_requested": "vector_or_coordinates_from_pdf_zip",
            "minimum_required_fields": "coordenada;crs;data;fenomeno;escala;origem_do_anexo;metodo_de_digitalizacao_se_houver",
            "why_needed": "PDF/ZIP sem vetor nao fecha geometria forte nem link patch-level",
            "revp_use_case": "converter evidencia documental SGB em geometria auditavel",
            "unblocks_feature_layer": "observational_evidence",
            "unblocks_observational_evidence": True,
            "unblocks_candidate_patch_features": False,
            "unblocks_embedding_input": False,
            "unblocks_qa": True,
            "unblocks_17b_future": True,
            "package_group": "SGB/CPRM / vetores ou coordenadas",
        },
        {
            "formal_request_id": "P1_PET_2024_VALPARAISO_FLORESTA_GEOMETRY",
            "priority": "P1",
            "provider_name": "Defesa Civil Petropolis / DRM-RJ (a confirmar)",
            "provider_type": "defesa_civil",
            "city_or_region": "Petropolis",
            "related_event_id": "PET_2024_03_21_28",
            "related_package_id": "PET_2024_VALPARAISO_FLORESTA",
            "request_title": "Solicitacao de geometria ou ocorrencias Petropolis 2024 Valparaiso/Floresta",
            "artifact_requested": "geometria, pontos, enderecos ou tabela de ocorrencias para Valparaiso/Floresta",
            "artifact_type_requested": "official_observed_event_geometry_or_table",
            "minimum_required_fields": "data_ou_periodo;localidade;coordenada_ou_endereco;fenomeno;fonte;incerteza_ou_escala",
            "why_needed": "recuperar evidencia observacional oficial para evento PET 2024 citado no 17C3/17C4",
            "revp_use_case": "evidencia_observacional_petropolis_2024",
            "unblocks_feature_layer": "observational_evidence;qa",
            "unblocks_observational_evidence": True,
            "unblocks_candidate_patch_features": False,
            "unblocks_embedding_input": False,
            "unblocks_qa": True,
            "unblocks_17b_future": True,
            "package_group": "Petropolis 2024 Valparaiso/Floresta",
        },
        {
            "formal_request_id": "P1_DEM_HAND_DRAINAGE_COVERAGE_CANDIDATE_GRID",
            "priority": "P1",
            "provider_name": "GEE/DEM/HAND/drenagem - fonte a confirmar",
            "provider_type": "infraestrutura_dados",
            "city_or_region": "Recife",
            "related_event_id": "REC_2022_05_24_30",
            "related_package_id": "S17C6_CANARY_GRID_RECIFE_CHARTER758_V1",
            "request_title": "Solicitacao de DEM/HAND/drenagem cobrindo a grade candidata",
            "artifact_requested": "DEM/HAND, drenagem, distancia a agua, flow accumulation e TWI por bbox candidata",
            "artifact_type_requested": "candidate_specific_physical_source_artifacts",
            "minimum_required_fields": "bbox;crs;resolucao;versao;fonte;licenca;metodo_de_agregacao;campos_zonais",
            "why_needed": "destravar features fisicas sem copiar valores de patch oficial",
            "revp_use_case": "features_fisicas_candidatas",
            "unblocks_feature_layer": "physical_static",
            "unblocks_observational_evidence": False,
            "unblocks_candidate_patch_features": True,
            "unblocks_embedding_input": False,
            "unblocks_qa": False,
            "unblocks_17b_future": True,
            "package_group": "Insumos geoespaciais para grade candidata",
        },
        {
            "formal_request_id": "P1_CHIRPS_PRE_EVENT_WINDOW_REC_2022_05_24_30",
            "priority": "P1",
            "provider_name": "CHIRPS/GEE - fonte a confirmar",
            "provider_type": "agencia_hidrometeorologica",
            "city_or_region": "Recife",
            "related_event_id": "REC_2022_05_24_30",
            "related_package_id": "REC_2022_05_24_30",
            "request_title": "Solicitacao de janela CHIRPS pre-evento Recife maio 2022",
            "artifact_requested": "acumulados CHIRPS 3d, 7d, 30d e runoff_context por patch candidato",
            "artifact_type_requested": "candidate_specific_rainfall_zonal_stats",
            "minimum_required_fields": "janela_temporal;acumulado_3d;acumulado_7d;acumulado_30d;resolucao;metodo_de_agregacao;fonte",
            "why_needed": "destravar chuva/gatilho pre-evento sem usar dado pos-evento",
            "revp_use_case": "features_chuva_candidatas",
            "unblocks_feature_layer": "rainfall_trigger",
            "unblocks_observational_evidence": False,
            "unblocks_candidate_patch_features": True,
            "unblocks_embedding_input": False,
            "unblocks_qa": False,
            "unblocks_17b_future": True,
            "package_group": "Insumos geoespaciais para grade candidata",
        },
        {
            "formal_request_id": "P1_SENTINEL2_TILE_CANDIDATE_GRID",
            "priority": "P1",
            "provider_name": "STAC/CDSE/GEE - fonte a confirmar",
            "provider_type": "sensoriamento_remoto",
            "city_or_region": "Recife",
            "related_event_id": "REC_2022_05_24_30",
            "related_package_id": "S17C6_CANARY_GRID_RECIFE_CHARTER758_V1",
            "request_title": "Solicitacao de tile Sentinel-2 real para patches candidatos",
            "artifact_requested": "tile ou metadata Sentinel-2 pre-evento cobrindo bboxes candidatas",
            "artifact_type_requested": "candidate_specific_sentinel2_tile_or_metadata",
            "minimum_required_fields": "product_id;scene_id;data_de_aquisicao;bandas;resolucao;cloud_mask;bbox;formato",
            "why_needed": "destravar NDVI/NDWI/MNDWI/NDBI e tile base para embedding",
            "revp_use_case": "features_sentinel2_e_input_embedding",
            "unblocks_feature_layer": "sentinel2_spectral;embedding_representation",
            "unblocks_observational_evidence": False,
            "unblocks_candidate_patch_features": True,
            "unblocks_embedding_input": True,
            "unblocks_qa": False,
            "unblocks_17b_future": True,
            "package_group": "Sentinel-2/DINO",
        },
        {
            "formal_request_id": "P2_DINO_SATMAE_INPUT_TILE",
            "priority": "P2",
            "provider_name": "Pipeline interno DINO/SatMAE - sem execucao nesta sprint",
            "provider_type": "runtime_tecnico",
            "city_or_region": "Recife",
            "related_event_id": "REC_2022_05_24_30",
            "related_package_id": "S17C6_EMBEDDING_CONTRACT",
            "request_title": "Preparacao de tile real para DINO/SatMAE",
            "artifact_requested": "manifest de tile real com bandas, resolucao, politica de nuvem e pre-processamento",
            "artifact_type_requested": "embedding_input_manifest",
            "minimum_required_fields": "tile_path_hash;bandas;resolucao;data;cloud_policy;normalizacao;modelo_alvo",
            "why_needed": "destravar embedding real sem usar smoke test sintetico",
            "revp_use_case": "input_embedding_real_review_only",
            "unblocks_feature_layer": "embedding_representation",
            "unblocks_observational_evidence": False,
            "unblocks_candidate_patch_features": False,
            "unblocks_embedding_input": True,
            "unblocks_qa": False,
            "unblocks_17b_future": True,
            "package_group": "Sentinel-2/DINO",
        },
        {
            "formal_request_id": "P2_SAR_RUNTIME_OR_EXPORT_PATH",
            "priority": "P2",
            "provider_name": "Runtime tecnico SAR - fonte a confirmar",
            "provider_type": "runtime_tecnico",
            "city_or_region": "Recife",
            "related_event_id": "REC_2022_05_24_30",
            "related_package_id": "SUSC_17C2_SAR_PLAN",
            "request_title": "Solicitacao de runtime ou caminho de export SAR review-only",
            "artifact_requested": "runtime, credenciais ou caminho de export SAR com regra anti-leakage documentada",
            "artifact_type_requested": "sar_runtime_or_export_plan",
            "minimum_required_fields": "runtime;credencial_requerida;produto;janela_temporal;politica_pos_evento;formato_export",
            "why_needed": "registrar dependencia tecnica SAR sem executar SAR nesta sprint",
            "revp_use_case": "sar_observacional_futuro_review_only",
            "unblocks_feature_layer": "sar_observational",
            "unblocks_observational_evidence": True,
            "unblocks_candidate_patch_features": False,
            "unblocks_embedding_input": False,
            "unblocks_qa": True,
            "unblocks_17b_future": True,
            "package_group": "SAR runtime/export",
        },
    ]
    for spec in specs:
        spec["submission_channel_known"] = False
        spec["submission_channel_reference"] = "not_available"
        spec["request_status"] = "needs_contact_channel"
        spec["do_not_mark_as_received"] = True
    return specs


def build_formal_request_registry() -> list[dict]:
    rows = []
    for spec in sorted(request_specs(), key=lambda item: item["formal_request_id"]):
        row = {k: spec[k] for k in [
            "formal_request_id", "priority", "provider_name", "provider_type",
            "city_or_region", "related_event_id", "related_package_id",
            "request_title", "artifact_requested", "artifact_type_requested",
            "minimum_required_fields", "why_needed", "revp_use_case",
            "unblocks_feature_layer",
        ]}
        row.update({
            "unblocks_observational_evidence": _bool_text(spec["unblocks_observational_evidence"]),
            "unblocks_candidate_patch_features": _bool_text(spec["unblocks_candidate_patch_features"]),
            "unblocks_embedding_input": _bool_text(spec["unblocks_embedding_input"]),
            "unblocks_qa": _bool_text(spec["unblocks_qa"]),
            "unblocks_17b_future": _bool_text(spec["unblocks_17b_future"]),
            "submission_channel_known": _bool_text(spec["submission_channel_known"]),
            "submission_channel_reference": spec["submission_channel_reference"],
            "request_status": spec["request_status"],
            "do_not_mark_as_received": _bool_text(spec["do_not_mark_as_received"]),
            **GOV,
        })
        rows.append(row)
    return rows


def build_provider_package_index() -> list[dict]:
    packages = [
        ("S17C10_PROVIDER_PKG_001", "DRM-RJ / Petropolis 2022", "DRM-RJ/NADE", "servico_geologico", ["P0_DRM_RJ_PKG_FR_PET_001"], "Petropolis", "PET_2022_02_15", "obter geometria oficial P0"),
        ("S17C10_PROVIDER_PKG_002", "COMPDEC Recife / Recife maio 2022", "COMPDEC/Defesa Civil PE", "defesa_civil", ["P0_COMPDEC_RECIFE_PKG_FR_REC_002"], "Recife", "REC_2022_05_24_30", "obter pontos/enderecos oficiais P0"),
        ("S17C10_PROVIDER_PKG_003", "SGB/CPRM / vetores ou coordenadas", "SGB/CPRM", "servico_geologico", ["P1_SGB_CPRM_VECTOR_OR_COORDINATES"], "Petropolis", "PET_2022_02_15", "converter PDF/ZIP em vetor ou coordenadas auditaveis"),
        ("S17C10_PROVIDER_PKG_004", "Petropolis 2024 Valparaiso/Floresta", "Defesa Civil Petropolis / DRM-RJ (a confirmar)", "defesa_civil", ["P1_PET_2024_VALPARAISO_FLORESTA_GEOMETRY"], "Petropolis", "PET_2024_03_21_28", "obter geometria ou ocorrencias oficiais"),
        ("S17C10_PROVIDER_PKG_005", "Insumos geoespaciais para grade candidata", "fontes geoespaciais a confirmar", "infraestrutura_dados", ["P1_DEM_HAND_DRAINAGE_COVERAGE_CANDIDATE_GRID", "P1_CHIRPS_PRE_EVENT_WINDOW_REC_2022_05_24_30"], "Recife", "REC_2022_05_24_30", "destravar features fisicas e chuva"),
        ("S17C10_PROVIDER_PKG_006", "Sentinel-2/DINO", "STAC/CDSE/GEE e pipeline interno", "sensoriamento_remoto", ["P1_SENTINEL2_TILE_CANDIDATE_GRID", "P2_DINO_SATMAE_INPUT_TILE"], "Recife", "REC_2022_05_24_30", "destravar Sentinel-2 e tile real de embedding"),
        ("S17C10_PROVIDER_PKG_007", "SAR runtime/export", "runtime tecnico SAR a confirmar", "runtime_tecnico", ["P2_SAR_RUNTIME_OR_EXPORT_PATH"], "Recife", "REC_2022_05_24_30", "registrar caminho tecnico futuro para SAR review-only"),
    ]
    priority_rank = {"P0": 0, "P1": 1, "P2": 2}
    req = {r["formal_request_id"]: r for r in request_specs()}
    rows = []
    for pkg_id, package_name, provider, ptype, reqs, region, events, purpose in packages:
        priority_max = sorted([req[r]["priority"] for r in reqs], key=lambda p: priority_rank[p])[0]
        rows.append({
            "provider_package_id": pkg_id,
            "provider_name": provider,
            "provider_type": ptype,
            "requests_in_package": ";".join(reqs),
            "priority_max": priority_max,
            "city_or_region": region,
            "related_events": events,
            "package_purpose": purpose,
            "manual_submission_ready": "false",
            "missing_contact_or_channel": "true",
            "attachments_expected": "susc_17c10_attachment_manifest.csv",
            "review_only": "true",
        })
    return rows


FIELD_SETS = {
    "observed": [
        ("data_do_evento", "data ou periodo do evento", "alinhamento temporal", True, True, False, False, False, True),
        ("tipo_do_fenomeno", "inundacao, alagamento, deslizamento ou outro fenomeno", "evitar mistura de fenomenos", True, False, False, False, False, True),
        ("geometria_ou_coordenada", "poligono, ponto, endereco georreferenciado ou tabela", "ligacao patch-level", True, False, True, False, False, True),
        ("incerteza_ou_escala", "escala, precisao ou incerteza espacial", "avaliar qualidade da evidencia", True, False, True, False, False, True),
        ("fonte_responsavel", "orgao, setor ou responsavel institucional", "proveniencia e QA", False, False, False, False, False, True),
        ("crs_se_houver", "sistema de referencia de coordenadas", "reproducibilidade espacial", True, False, True, False, False, True),
    ],
    "physical": [
        ("DEM_HAND_cobrindo_bbox", "DEM/HAND cobrindo a bbox candidata", "feature fisica real", True, False, True, True, False, False),
        ("crs", "sistema de referencia", "zonal stats reproduzivel", True, False, True, True, False, False),
        ("resolucao", "resolucao espacial", "comparabilidade por patch", False, False, False, True, False, False),
        ("data_versao", "data ou versao do produto", "proveniencia", False, True, False, True, False, True),
        ("fonte_licenca_formato", "fonte, licenca e formato", "uso publico auditavel", False, False, False, True, False, True),
    ],
    "rain": [
        ("acumulado_pre_evento", "acumulado 3d, 7d e 30d", "gatilho de chuva", False, True, False, True, False, False),
        ("janela_temporal", "periodo relativo ao evento", "bloquear leakage pos-evento", False, True, False, True, False, True),
        ("fonte_resolucao", "fonte CHIRPS/GEE e resolucao", "proveniencia", False, False, False, True, False, False),
        ("metodo_agregacao", "media, soma ou zonal stats", "reprodutibilidade", False, False, False, True, False, False),
    ],
    "sentinel": [
        ("tile_real_cobrindo_bbox", "tile real cobrindo bbox candidata", "input espectral e embedding", True, True, True, True, True, False),
        ("data_aquisicao", "data de aquisicao da cena", "politica pre-evento", False, True, False, True, True, True),
        ("bandas_resolucao", "bandas e resolucao", "calculo de indices e tile", False, False, False, True, True, False),
        ("politica_nuvem", "cloud mask ou risco de nuvem", "qualidade do tile", False, False, False, True, True, True),
        ("formato_export", "formato de metadata, manifest ou tile local-only", "ingestao futura", False, False, False, True, True, False),
    ],
    "sar": [
        ("runtime_credencial", "runtime e credencial requeridos", "execucao futura controlada", False, False, False, False, False, True),
        ("produto_janela_temporal", "produto SAR e janela temporal", "bloquear leakage", False, True, False, False, False, True),
        ("formato_export", "formato leve ou local-only", "governanca de bruto pesado", False, False, False, False, False, True),
    ],
}


def _field_set_for_request(request_id: str) -> str:
    if "DEM_HAND" in request_id:
        return "physical"
    if "CHIRPS" in request_id:
        return "rain"
    if "SENTINEL2" in request_id or "DINO" in request_id:
        return "sentinel"
    if "SAR" in request_id:
        return "sar"
    return "observed"


def build_required_fields() -> list[dict]:
    rows = []
    for spec in request_specs():
        for field in FIELD_SETS[_field_set_for_request(spec["formal_request_id"])]:
            name, desc, why, patch, temporal, geometry, extraction, embedding, qa = field
            rows.append({
                "required_field_id": f"S17C10_FIELD_{len(rows) + 1:05d}",
                "formal_request_id": spec["formal_request_id"],
                "field_name": name,
                "field_description": desc,
                "why_required": why,
                "required_for_patch_level": _bool_text(patch),
                "required_for_temporal_alignment": _bool_text(temporal),
                "required_for_geometry": _bool_text(geometry),
                "required_for_feature_extraction": _bool_text(extraction),
                "required_for_embedding": _bool_text(embedding),
                "required_for_qa": _bool_text(qa),
                "minimum_acceptance_condition": "campo presente, origem documentada e sem substituicao por placeholder",
                "review_only": "true",
            })
    return rows


def build_traceability() -> list[dict]:
    blocker_map = {
        "P0_DRM_RJ_PKG_FR_PET_001": ("S17C3_REQ_PET_2022_02_15_drm_rj_nade_pkg_fr_pet_001", "SUSC-17C3", "official_data_not_public_requires_formal_request"),
        "P0_COMPDEC_RECIFE_PKG_FR_REC_002": ("S17C4_REQ_REC_2022_05_24_30_compdec_defesa_civil_pe_pkg_fr_rec_002", "SUSC-17C4", "PKG_FR_REC_002 pending_formal_request"),
        "P1_SGB_CPRM_VECTOR_OR_COORDINATES": ("S17C3_REQ_SGB_CPRM_PDF_ZIP", "SUSC-17C3", "pdf_or_zip_without_vector_needs_coordinate_or_vector_extraction"),
        "P1_PET_2024_VALPARAISO_FLORESTA_GEOMETRY": ("S17C3_ST_PET_2024_03_21_28_valparaiso_floresta", "SUSC-17C3", "temporal_windows_ready_but_acquisition_status_not_acquired"),
        "P1_DEM_HAND_DRAINAGE_COVERAGE_CANDIDATE_GRID": ("S17C9_BLOCKER_001", "SUSC-17C9", "candidate_specific_artifacts_missing"),
        "P1_CHIRPS_PRE_EVENT_WINDOW_REC_2022_05_24_30": ("S17C9_REQ_0004", "SUSC-17C9", "missing_candidate_specific_chirps_pre_event_zonal_stats"),
        "P1_SENTINEL2_TILE_CANDIDATE_GRID": ("S17C9_BLOCKER_005", "SUSC-17C9", "sentinel2_tile_missing"),
        "P2_DINO_SATMAE_INPUT_TILE": ("S17C9_BLOCKER_006", "SUSC-17C9", "embedding_tile_missing"),
        "P2_SAR_RUNTIME_OR_EXPORT_PATH": ("S17C9_BLOCKER_007", "SUSC-17C9", "sar_runtime_missing"),
    }
    rows = []
    for spec in request_specs():
        bid, milestone, desc = blocker_map[spec["formal_request_id"]]
        rows.append({
            "traceability_id": f"S17C10_TRACE_{len(rows) + 1:05d}",
            "formal_request_id": spec["formal_request_id"],
            "source_blocker_id": bid,
            "blocker_source_milestone": milestone,
            "blocker_description": desc,
            "required_resolution": "resposta oficial ou artefato candidato-especifico com metadados minimos e hash em intake futuro",
            "resolved_if_received": "false",
            "still_requires_human_qa": "true",
            "still_requires_patch_policy": "true",
            "still_requires_feature_extraction": "true",
            "still_blocks_17b": "true",
            "review_only": "true",
        })
    return rows


def build_impact_matrix() -> list[dict]:
    rows = []
    for spec in request_specs():
        layer = spec["unblocks_feature_layer"]
        rows.append({
            "impact_id": f"S17C10_IMPACT_{len(rows) + 1:05d}",
            "formal_request_id": spec["formal_request_id"],
            "impacts_observational_evidence": _bool_text(spec["unblocks_observational_evidence"]),
            "impacts_physical_static": _bool_text("physical_static" in layer),
            "impacts_urban_territorial": _bool_text("urban_territorial" in layer),
            "impacts_sentinel2_spectral": _bool_text("sentinel2_spectral" in layer),
            "impacts_rainfall_trigger": _bool_text("rainfall_trigger" in layer),
            "impacts_sar_observational": _bool_text("sar_observational" in layer),
            "impacts_embedding_representation": _bool_text("embedding_representation" in layer),
            "impact_on_candidate_patches": "prepara desbloqueio futuro, sem criar patch oficial",
            "impact_on_official_patch_policy": "nao altera politica; ainda requer decisao humana futura",
            "impact_on_17b_future": "pode reduzir bloqueio futuro, mas nao libera 17B nesta sprint",
            "impact_level": "critical" if spec["priority"] == "P0" else ("high" if spec["priority"] == "P1" else "medium"),
            "why": spec["why_needed"],
            "review_only": "true",
        })
    return rows


def build_checklist() -> list[dict]:
    steps = [
        ("revisar_template", "Revisar modelo de mensagem e substituir placeholders", "pesquisadora"),
        ("confirmar_canal", "Descobrir canal oficial de submissao sem inventar contato", "pesquisadora"),
        ("selecionar_anexos", "Selecionar anexos seguros e revisar sensibilidade", "pesquisadora"),
        ("registrar_envio_externo", "Registrar manualmente somente apos envio real fora do 17C10", "pesquisadora"),
    ]
    rows = []
    for spec in request_specs():
        for order, (code, desc, actor) in enumerate(steps, start=1):
            rows.append({
                "checklist_id": f"S17C10_CHECK_{len(rows) + 1:05d}",
                "formal_request_id": spec["formal_request_id"],
                "step_order": str(order),
                "step_description": f"{code}: {desc}",
                "responsible_actor": actor,
                "requires_human_action": "true",
                "requires_attachment": _bool_text(code == "selecionar_anexos"),
                "requires_contact_discovery": _bool_text(code == "confirmar_canal"),
                "done": "false",
                "notes": "nao executar automaticamente; manter rastro manual",
                "review_only": "true",
            })
    return rows


TEMPLATE_GROUPS = [
    ("DRM-RJ / Petropolis 2022", "P0_DRM_RJ_PKG_FR_PET_001", "Solicitacao de geometria oficial - PKG_FR_PET_001"),
    ("COMPDEC Recife / Recife maio 2022", "P0_COMPDEC_RECIFE_PKG_FR_REC_002", "Solicitacao de pontos, enderecos ou geometria - PKG_FR_REC_002"),
    ("SGB/CPRM", "P1_SGB_CPRM_VECTOR_OR_COORDINATES", "Solicitacao de vetores ou coordenadas associados a PDF/ZIP"),
    ("Petropolis 2024 Valparaiso/Floresta", "P1_PET_2024_VALPARAISO_FLORESTA_GEOMETRY", "Solicitacao de geometria ou ocorrencias oficiais"),
    ("Insumos geoespaciais para grade candidata", "P1_DEM_HAND_DRAINAGE_COVERAGE_CANDIDATE_GRID", "Solicitacao de DEM/HAND/drenagem"),
    ("Sentinel-2/tile real para patches candidatos", "P1_SENTINEL2_TILE_CANDIDATE_GRID", "Solicitacao de tile Sentinel-2 e metadados"),
    ("SAR runtime/export tecnico", "P2_SAR_RUNTIME_OR_EXPORT_PATH", "Solicitacao de caminho tecnico SAR review-only"),
]


def build_message_templates() -> str:
    req = {r["formal_request_id"]: r for r in request_specs()}
    parts = [
        "# SUSC-17C10 - Modelos de mensagem formal",
        "",
        "Uso: modelos para adaptacao manual. nao enviar sem revisar placeholders, anexos, canal oficial e dados institucionais reais.",
        "",
    ]
    for title, request_id, subject in TEMPLATE_GROUPS:
        spec = req[request_id]
        parts.extend([
            f"## {title}",
            "",
            f"**Assunto sugerido:** {subject}",
            "",
            "Prezadas(os),",
            "",
            "Meu nome e [NOME_DA_SOLICITANTE], vinculada(o) a [INSTITUICAO/ORIENTACAO_A_CONFIRMAR]. Escrevo no contexto de pesquisa academica e metodologica do projeto REV-P, em carater review-only, sem uso comercial e sem objetivo de criar ground truth operacional.",
            "",
            f"Solicito, se disponivel, o seguinte artefato: {spec['artifact_requested']}.",
            "",
            f"Campos minimos solicitados: {spec['minimum_required_fields']}.",
            "",
            f"Justificativa tecnica: {spec['why_needed']}. O material sera usado apenas para auditoria metodologica, reproducibilidade e avaliacao humana futura; qualquer resposta sera validada antes de uso cientifico.",
            "",
            "Formato preferencial: arquivo vetorial, tabela CSV/GeoJSON, metadata auditavel ou manifest leve, conforme aplicavel. Dados brutos pesados podem ser mantidos em canal externo ou local controlado, sem publicacao direta.",
            "",
            "Agradeco a atencao e fico a disposicao para ajustar o escopo do pedido aos procedimentos do orgao.",
            "",
            "[NOME]",
            "[INSTITUICAO]",
            "[CONTATO]",
            "[ANEXOS_A_REVISAR]",
            "",
        ])
    return "\n".join(parts)


def build_attachment_manifest() -> list[dict]:
    base_attachments = [
        ("grade_candidata_17c6", "geometria candidata review-only", OUT / "susc_17c6_candidate_patch_grid.geojson"),
        ("plano_17c9", "plano de artefatos fonte e bloqueios", OUT / "susc_17c9_external_artifact_request_plan.csv"),
        ("relatorio_17c10", "contexto do pacote formal", REPORT),
    ]
    rows = []
    for spec in request_specs():
        for name, purpose, path in base_attachments:
            rows.append({
                "attachment_id": f"S17C10_ATTACH_{len(rows) + 1:05d}",
                "formal_request_id": spec["formal_request_id"],
                "attachment_name": name,
                "attachment_purpose": purpose,
                "attachment_path": rel(path),
                "attachment_exists": _bool_text(path.exists()),
                "safe_to_send": "false",
                "contains_sensitive_data": "false",
                "contains_raw_heavy_data": "false",
                "needs_manual_review_before_sending": "true",
                "review_only": "true",
            })
    return rows


def build_response_intake_plan() -> list[dict]:
    rows = []
    for spec in request_specs():
        rows.append({
            "intake_plan_id": f"S17C10_INTAKE_{len(rows) + 1:05d}",
            "formal_request_id": spec["formal_request_id"],
            "expected_response_type": spec["artifact_type_requested"],
            "expected_format": "csv;geojson;shp_zip;pdf;metadata_manifest;email_text",
            "minimum_metadata_required": spec["minimum_required_fields"],
            "validation_needed": "hash;crs;data;fonte;licenca;no_leakage;qa_humana",
            "where_to_store_raw_response": "local_only/suscetibilidade/formal_responses/",
            "where_to_store_public_manifest": "outputs_public/suscetibilidade/future_response_manifest.csv",
            "hash_required": "true",
            "manual_qa_required": "true",
            "expected_next_milestone": "SUSC-17C11 Intake de Respostas Formais",
            "review_only": "true",
        })
    return rows


def build_blockers() -> list[dict]:
    codes = [
        "formal_requests_not_submitted",
        "responses_not_received",
        "candidate_specific_artifacts_missing",
        "candidate_patch_policy_missing",
        "pipeline_adapters_missing",
        "official_patch_id_dependency",
        "sentinel2_tile_missing",
        "embedding_tile_missing",
        "sar_runtime_missing",
        "qa_not_accepted",
        "17b_blocked_until_real_features_and_policy",
    ]
    return [{
        "blocker_id": f"S17C10_BLOCKER_{i:03d}",
        "blocker_code": code,
        "description": code.replace("_", " "),
        "blocks_17b": "true",
        "blocks_score_v7": "true",
        "blocks_training": "true",
        **GOV,
    } for i, code in enumerate(codes, start=1)]


def build_summary() -> dict:
    registry = build_formal_request_registry()
    packages = build_provider_package_index()
    attachments = build_attachment_manifest()
    templates_count = len(TEMPLATE_GROUPS)
    return {
        "milestone": "SUSC-17C10",
        "formal_requests_count": len(registry),
        "p0_requests_count": sum(1 for r in registry if r["priority"] == "P0"),
        "p1_requests_count": sum(1 for r in registry if r["priority"] == "P1"),
        "p2_requests_count": sum(1 for r in registry if r["priority"] == "P2"),
        "provider_packages_count": len(packages),
        "manual_submission_ready_count": sum(1 for p in packages if p["manual_submission_ready"] == "true"),
        "needs_contact_channel_count": sum(1 for r in registry if r["request_status"] == "needs_contact_channel"),
        "attachment_manifest_count": len(attachments),
        "message_templates_count": templates_count,
        "requests_marked_sent_count": 0,
        "responses_received_count": 0,
        "features_extracted_this_sprint": 0,
        "score_v6_changed": False,
        "score_v7_created": SCORE_V7.exists(),
        "official_patch_created": False,
        "official_patch_link_created": False,
        "raw_raster_committed": False,
        "eligible_for_17b_now": False,
        "eligible_for_score_v7": False,
        "review_only": True,
        "trainable": False,
        "ground_truth": False,
        "recommended_next_milestone": "SUSC-17C11 Descoberta de Canais Oficiais",
    }


def build_request_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SUSC-17C10 formal request schema v1",
        "type": "object",
        "required": list(build_formal_request_registry()[0].keys()),
        "properties": {
            "priority": {"enum": ["P0", "P1", "P2"]},
            "provider_type": {"enum": ["defesa_civil", "servico_geologico", "prefeitura", "agencia_hidrometeorologica", "portal_dados", "sensoriamento_remoto", "infraestrutura_dados", "runtime_tecnico", "outro"]},
            "request_status": {"enum": ["draft_ready_for_manual_submission", "needs_contact_channel", "needs_provider_confirmation", "context_only_do_not_request_now", "blocked_insufficient_provider_info"]},
            "do_not_mark_as_received": {"const": "true"},
            "review_only": {"const": "true"},
            "trainable": {"const": "false"},
            "ground_truth": {"const": "false"},
        },
    }


def build_impact_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SUSC-17C10 request impact schema v1",
        "type": "object",
        "required": list(build_impact_matrix()[0].keys()),
        "properties": {
            "impact_level": {"enum": ["critical", "high", "medium", "low", "context_only"]},
            "review_only": {"const": "true"},
        },
    }


def build_report() -> str:
    summary = build_summary()
    return "\n".join([
        "# SUSC-17C10 - Pacote de Solicitacao Formal",
        "",
        "O 17C9 revelou que a extracao local nao e o proximo gargalo principal: faltam artefatos oficiais, artefatos candidato-especificos e canais formais para recuperar geometria, DEM/HAND, drenagem, CHIRPS, Sentinel-2, tile de embedding e dependencias SAR.",
        "",
        "Por isso, o 17C10 prepara solicitacoes formais manuais. Nenhum e-mail foi enviado, nenhum protocolo foi aberto e nenhuma resposta foi recebida.",
        "",
        "## Pedidos P0",
        "",
        "- DRM-RJ/NADE `PKG_FR_PET_001`.",
        "- COMPDEC/Defesa Civil PE `PKG_FR_REC_002`.",
        "",
        "## Pedidos P1",
        "",
        "- SGB/CPRM vetores ou coordenadas associados a PDF/ZIP.",
        "- Petropolis 2024 Valparaiso/Floresta.",
        "- DEM/HAND/drenagem para grade candidata.",
        "- CHIRPS pre-evento Recife maio 2022.",
        "- Sentinel-2 real para grade candidata.",
        "",
        "## Pedidos P2",
        "",
        "- Tile real DINO/SatMAE.",
        "- Runtime ou caminho de export SAR.",
        "",
        "Cada pedido tem campos minimos, rastreabilidade para bloqueadores e plano de ingestao futura. Mesmo se uma resposta chegar, ainda serao necessarios hash, manifest, QA humana, politica de patch candidato e extracao real futura.",
        "",
        f"Solicitacoes formais: {summary['formal_requests_count']}. Pacotes por fornecedor: {summary['provider_packages_count']}. Modelos de mensagem: {summary['message_templates_count']}.",
        "",
        "Score v6 nao foi alterado. Score v7 e 17B continuam bloqueados porque nao ha feature real, resposta oficial validada, QA ou politica de promocao.",
        "",
        f"Proximo marco recomendado: `{summary['recommended_next_milestone']}`.",
    ])


def run_all() -> None:
    _require_inputs()
    write_json(REQUEST_SCHEMA, build_request_schema())
    write_json(IMPACT_SCHEMA, build_impact_schema())
    write_csv(REGISTRY, build_formal_request_registry())
    write_csv(PROVIDER_INDEX, build_provider_package_index())
    write_csv(REQUIRED_FIELDS, build_required_fields())
    write_csv(TRACEABILITY, build_traceability())
    write_csv(IMPACT_MATRIX, build_impact_matrix())
    write_csv(CHECKLIST, build_checklist())
    write_markdown(MESSAGE_TEMPLATES, build_message_templates())
    write_csv(ATTACHMENTS, build_attachment_manifest())
    write_csv(RESPONSE_INTAKE, build_response_intake_plan())
    write_json(SUMMARY, build_summary())
    write_csv(BLOCKERS, build_blockers())
    write_markdown(REPORT, build_report())


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _schema_violations(row: dict, schema: dict) -> list[str]:
    errors = []
    for key in schema.get("required", []):
        if key not in row:
            errors.append(f"missing:{key}")
        elif row[key] == "":
            errors.append(f"empty:{key}")
    for key, rule in schema.get("properties", {}).items():
        if key not in row:
            continue
        if "const" in rule and row[key] != rule["const"]:
            errors.append(f"const:{key}")
        if "enum" in rule and row[key] not in rule["enum"]:
            errors.append(f"enum:{key}")
    return errors


def _assert(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate() -> int:
    errors: list[str] = []
    try:
        _require_inputs()
        run_all()
        first = {p: _sha(p) for p in REQUIRED_OUTPUTS}
        run_all()
        second = {p: _sha(p) for p in REQUIRED_OUTPUTS}
        _assert(first == second, "build nao e byte-identico", errors)
        for path in REQUIRED_OUTPUTS:
            _assert(path.exists(), f"artefato ausente {rel(path)}", errors)

        request_schema = read_json(REQUEST_SCHEMA)
        impact_schema = read_json(IMPACT_SCHEMA)
        registry = read_csv(REGISTRY)
        impacts = read_csv(IMPACT_MATRIX)
        summary = read_json(SUMMARY)
        blockers = read_csv(BLOCKERS)

        for row in registry:
            errors.extend(f"{row.get('formal_request_id')}:{err}" for err in _schema_violations(row, request_schema))
            _assert(row["request_status"] != "sent", "solicitacao marcada como enviada", errors)
            _assert(row["do_not_mark_as_received"] == "true", "solicitacao pode ser marcada como recebida", errors)
            _assert(row["submission_channel_reference"] == "not_available", "canal/contato inventado", errors)
            _assert("@" not in row["submission_channel_reference"], "email inventado", errors)
            _assert(row["review_only"] == "true", "request nao review-only", errors)
            _assert(row["trainable"] == "false", "request trainable", errors)
            _assert(row["ground_truth"] == "false", "request ground_truth", errors)

        for row in impacts:
            errors.extend(f"{row.get('impact_id')}:{err}" for err in _schema_violations(row, impact_schema))
            _assert(row["review_only"] == "true", "impact nao review-only", errors)

        for path in [PROVIDER_INDEX, REQUIRED_FIELDS, TRACEABILITY, CHECKLIST, ATTACHMENTS, RESPONSE_INTAKE]:
            for row in read_csv(path):
                _assert(row["review_only"] == "true", f"{rel(path)} nao review-only", errors)
                if "done" in row:
                    _assert(row["done"] == "false", "checklist concluido indevidamente", errors)
                if "attachment_exists" in row:
                    p = ROOT / row["attachment_path"]
                    _assert(row["attachment_exists"] == _bool_text(p.exists()), "anexo ausente marcado como existente", errors)
                    _assert(row["contains_raw_heavy_data"] == "false", "anexo bruto pesado", errors)

        ids = [r["formal_request_id"] for r in registry]
        _assert(ids == sorted(ids), "formal_request_id nao deterministico", errors)
        _assert(len(ids) == len(set(ids)), "formal_request_id duplicado", errors)
        _assert("P0_DRM_RJ_PKG_FR_PET_001" in ids, "P0 DRM-RJ ausente", errors)
        _assert("P0_COMPDEC_RECIFE_PKG_FR_REC_002" in ids, "P0 COMPDEC ausente", errors)
        _assert("P1_SGB_CPRM_VECTOR_OR_COORDINATES" in ids, "P1 SGB/CPRM ausente", errors)

        text = MESSAGE_TEMPLATES.read_text(encoding="utf-8")
        for phrase in ["Prezadas(os)", "pesquisa academica", "review-only", "nao enviar"]:
            _assert(phrase in text, f"modelo sem frase obrigatoria: {phrase}", errors)
        _assert("@" not in text, "modelo contem email possivelmente inventado", errors)

        _assert(summary["formal_requests_count"] == len(registry), "summary requests count incorreto", errors)
        _assert(summary["p0_requests_count"] == 2, "summary P0 incorreto", errors)
        _assert(summary["p1_requests_count"] == 5, "summary P1 incorreto", errors)
        _assert(summary["p2_requests_count"] == 2, "summary P2 incorreto", errors)
        _assert(summary["requests_marked_sent_count"] == 0, "summary indica envio", errors)
        _assert(summary["responses_received_count"] == 0, "summary indica resposta recebida", errors)
        _assert(summary["features_extracted_this_sprint"] == 0, "feature real criada", errors)
        _assert(summary["score_v6_changed"] is False, "score v6 alterado no summary", errors)
        _assert(summary["score_v7_created"] is False, "score v7 criado no summary", errors)
        _assert(summary["official_patch_created"] is False, "patch oficial criado no summary", errors)
        _assert(summary["official_patch_link_created"] is False, "patch-link oficial criado no summary", errors)
        _assert(summary["raw_raster_committed"] is False, "raster bruto commitado no summary", errors)
        _assert(summary["eligible_for_17b_now"] is False, "17B elegivel indevidamente", errors)
        _assert(summary["eligible_for_score_v7"] is False, "score v7 elegivel indevidamente", errors)
        _assert(summary["review_only"] is True, "summary nao review-only", errors)
        _assert(summary["trainable"] is False, "summary trainable", errors)
        _assert(summary["ground_truth"] is False, "summary ground_truth", errors)
        _assert(len(blockers) > 0, "bloqueadores vazios", errors)

        _assert(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)]) == "", "score v6 tem diff", errors)
        _assert(_run_git(["diff", "--name-only", "--", rel(FEATURES)]) == "", "dataset oficial de features tem diff", errors)
        _assert(not SCORE_V7.exists(), "score v7 existe", errors)

        if errors:
            for err in errors:
                print(f"17C10 validation error: {err}", file=sys.stderr)
            return 1
        print(
            "17C10 -> "
            f"formal_requests={summary['formal_requests_count']} "
            f"P0={summary['p0_requests_count']} P1={summary['p1_requests_count']} P2={summary['p2_requests_count']} "
            "sent=0 responses=0 score_v7_created=False"
        )
        return 0
    except Exception as exc:
        print(f"17C10 validation exception: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    run_all()
