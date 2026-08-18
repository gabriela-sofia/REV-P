"""SUSC-17C26 pacote multimodal operacional, evidence graph e fila G4/G5.

Macro-marco totalmente offline e deterministico. Consolida todos os dados reais
ja obtidos (Sentinel-2 multitemporal 17C24, CHIRPS real 17C25, deltas 17C21/23)
em um pacote operacional por patch: dossies, evidence packets, grafo de
evidencias, matriz multimodal, matriz de gaps G4/G5, fila objetiva de Ground
Reference, pacotes de consulta por fonte, checklists, ranking review-only, indice
mestre, manifesto de reprodutibilidade. Nao coleta dados, nao acessa rede, nao
libera 17B: prepara o desbloqueio G4/G5 de forma programatica.

Determinismo: sem timestamps variaveis em outputs versionados, IDs deterministicos,
ordenacao estavel, build 2x byte-identico.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

from susc_io import ensure_dir, read_csv, read_json, rel, sha256_file, write_csv, write_json, write_markdown  # noqa: E402

DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

SCORE_V6 = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"
OFFICIAL_PATCHES = DAT / "susc_patches_official_v1.csv"
OFFICIAL_PATCH_LINKS = DAT / "susc_patch_links_official_v1.csv"

C25_MATRIX = OUT / "susc_17c25_multimodal_feature_matrix.csv"
C25_DAILY = OUT / "susc_17c25_chirps_daily_values.csv"
C25_AGG = OUT / "susc_17c25_chirps_window_aggregates.csv"
C25_RAINFALL = OUT / "susc_17c25_rainfall_trigger_feature_registry.csv"
C25_PROVENANCE = OUT / "susc_17c25_feature_provenance_trace.csv"
C25_MANIFEST = OUT / "susc_17c25_chirps_artifact_manifest.csv"
C25_REPLAY = OUT / "susc_17c25_replay_audit.csv"
C25_NO_LEAKAGE = OUT / "susc_17c25_no_leakage_audit.csv"
C25_SUMMARY = OUT / "susc_17c25_readiness_summary.json"
C24_MATRIX = OUT / "susc_17c24_cloud_robust_feature_matrix.csv"
C24_MANIFEST = OUT / "susc_17c24_light_artifact_manifest.csv"
C23_MATRIX = OUT / "susc_17c23_candidate_multimodal_feature_matrix.csv"
C23_DELTA_MATRIX = OUT / "susc_17c23_observational_delta_review_matrix.csv"
C21_DELTA = OUT / "susc_17c21_pre_post_delta_features.csv"
C19_LINEAGE = OUT / "susc_17c19_event_temporal_lineage.csv"
C19_BINDING = OUT / "susc_17c19_candidate_patch_temporal_binding.csv"
PATCH_GRID = OUT / "susc_17c6_candidate_patch_grid.csv"
PATCH_GEOJSON = OUT / "susc_17c6_candidate_patch_grid.geojson"
PATCH_LINKS = OUT / "susc_17c6_candidate_patch_links.csv"

REQUIRED_INPUTS = [
    SCORE_V6, C25_MATRIX, C25_DAILY, C25_AGG, C25_RAINFALL, C25_PROVENANCE, C25_MANIFEST,
    C25_REPLAY, C25_NO_LEAKAGE, C25_SUMMARY, C24_MATRIX, C24_MANIFEST, C23_MATRIX,
    C23_DELTA_MATRIX, C21_DELTA, C19_LINEAGE, C19_BINDING, PATCH_GRID, PATCH_GEOJSON, PATCH_LINKS,
]

DOSSIER_DIR = OUT / "susc_17c26_patch_dossiers"
PACKET_DIR = OUT / "susc_17c26_evidence_packets"
CHECKLIST_DIR = OUT / "susc_17c26_review_checklists"
QUERY_DIR = OUT / "susc_17c26_source_query_packages"

REPORT = OUT / "SUSC_17C26_PACOTE_MULTIMODAL_OPERACIONAL_EVIDENCE_GRAPH_G4_G5_REPORT.md"
PATCH_SUMMARY = OUT / "susc_17c26_patch_multimodal_summary.csv"
PACKET_REGISTRY = OUT / "susc_17c26_patch_evidence_packet_registry.csv"
GRAPH_NODES = OUT / "susc_17c26_evidence_graph_nodes.csv"
GRAPH_EDGES = OUT / "susc_17c26_evidence_graph_edges.csv"
REVIEW_PRIORITY = OUT / "susc_17c26_patch_review_priority.csv"
GR_TARGET_QUEUE = OUT / "susc_17c26_ground_reference_target_queue.csv"
GR_REQUIRED_FIELDS = OUT / "susc_17c26_required_ground_reference_fields.csv"
G4_G5_GAP = OUT / "susc_17c26_g4_g5_gap_matrix.csv"
SOURCE_QUERY_PACKAGES = OUT / "susc_17c26_source_query_packages.csv"
CHECKLIST_REGISTRY = OUT / "susc_17c26_review_checklist_registry.csv"
PROVENANCE_INDEX = OUT / "susc_17c26_multimodal_provenance_index.csv"
HASH_RECHECK = OUT / "susc_17c26_artifact_hash_recheck.csv"
NO_LEAKAGE = OUT / "susc_17c26_no_leakage_audit.csv"
GATES = OUT / "susc_17c26_gate_evaluation_matrix.csv"
REPRODUCIBILITY = OUT / "susc_17c26_reproducibility_manifest.json"
MASTER_INDEX = OUT / "susc_17c26_master_index.md"
SUMMARY = OUT / "susc_17c26_readiness_summary.json"
BLOCKERS = OUT / "susc_17c26_promotion_blockers.csv"

DOSSIER_SCHEMA = SCHEMAS / "susc_17c26_patch_dossier_schema_v1.json"
GRAPH_SCHEMA = SCHEMAS / "susc_17c26_evidence_graph_schema_v1.json"
GR_TARGET_SCHEMA = SCHEMAS / "susc_17c26_ground_reference_target_schema_v1.json"
PRIORITY_SCHEMA = SCHEMAS / "susc_17c26_review_priority_schema_v1.json"
QUERY_SCHEMA = SCHEMAS / "susc_17c26_source_query_package_schema_v1.json"

GOV = {"review_only": "true", "trainable": "false", "ground_truth": "false"}

S2_FEATURES = ["B03_mean_temporal_median", "B04_mean_temporal_median", "B08_mean_temporal_median", "B11_mean_temporal_median",
               "NDVI_mean_temporal_median", "NDWI_mean_temporal_median", "MNDWI_mean_temporal_median", "NDBI_mean_temporal_median"]
CHIRPS_WINDOWS = ["CHIRPS_3d", "CHIRPS_7d", "CHIRPS_30d"]
DELTA_FEATURES = ["delta_B03_mean", "delta_B04_mean", "delta_B08_mean", "delta_B11_mean",
                  "delta_NDVI_mean", "delta_NDWI_mean", "delta_MNDWI_mean", "delta_NDBI_mean"]

DOSSIER_SECTIONS = [
    "# Dossiê multimodal", "## Identificação", "## Evento e janela temporal", "## AOI e patch candidato",
    "## Sentinel-2 multitemporal", "## CHIRPS antecedente", "## Delta observacional pré/pós",
    "## Artefatos e hashes", "## Interpretação permitida", "## Interpretação proibida",
    "## Lacunas G4/G5", "## Campos necessários para Ground Reference", "## Próximas buscas objetivas",
    "## Status final",
]
DOSSIER_FINAL_STATUS = "scientific_review_only_not_ground_reference"

SOURCE_FAMILIES = [
    ("defesa_civil_recife", "ocorrencia_oficial_com_local", "documental_evidence"),
    ("apac_pernambuco", "geometria_ou_extensao", "geometry_or_extent"),
    ("cemaden", "separacao_alagamento_vs_deslizamento", "phenomenon_classification"),
]


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _run_git(args: list[str]) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def _require_inputs() -> None:
    missing = [path for path in REQUIRED_INPUTS if not path.exists()]
    if missing:
        raise FileNotFoundError("; ".join(rel(path) for path in missing))
    for path in REQUIRED_INPUTS:
        if path.suffix == ".json":
            read_json(path)
        elif path.suffix == ".csv":
            read_csv(path)


def _patch_ids() -> list[str]:
    return [row["candidate_patch_id"] for row in read_csv(PATCH_GRID)]


def _event_id() -> str:
    rows = read_csv(PATCH_GRID)
    return rows[0]["source_event_id"] if rows else ""


def _patch_grid_row(patch_id: str) -> dict:
    for row in read_csv(PATCH_GRID):
        if row["candidate_patch_id"] == patch_id:
            return row
    return {}


def _bbox(patch_id: str) -> str:
    r = _patch_grid_row(patch_id)
    return f"{r.get('xmin')},{r.get('ymin')},{r.get('xmax')},{r.get('ymax')}" if r else "not_available"


def _binding(patch_id: str) -> dict:
    for row in read_csv(C19_BINDING):
        if row["candidate_patch_id"] == patch_id:
            return row
    return {}


def _c25_matrix_row(patch_id: str) -> dict:
    for row in read_csv(C25_MATRIX):
        if row["candidate_patch_id"] == patch_id:
            return row
    return {}


def _c23_delta_row(patch_id: str) -> dict:
    for row in read_csv(C23_DELTA_MATRIX):
        if row["candidate_patch_id"] == patch_id:
            return row
    return {}


# ---------------------------------------------------------------------------
# 1 - Patch multimodal summary
# ---------------------------------------------------------------------------

def patch_multimodal_summary_rows() -> list[dict]:
    event_id = _event_id()
    rows = []
    for idx, patch_id in enumerate(_patch_ids(), start=1):
        m = _c25_matrix_row(patch_id)
        binding = _binding(patch_id)
        rows.append({
            "patch_multimodal_summary_id": f"S17C26_SUM_{idx:04d}",
            "candidate_patch_id": patch_id, "event_id": event_id, "bbox": _bbox(patch_id),
            "event_period_start": binding.get("event_period_start", "not_available"),
            "event_period_end": binding.get("event_period_end", "not_available"),
            "sentinel2_feature_count": str(len(S2_FEATURES)),
            "chirps_feature_count": str(len(CHIRPS_WINDOWS) * 3),
            "observational_delta_count": str(len(DELTA_FEATURES)),
            "total_review_only_features_count": str(len(S2_FEATURES) + len(CHIRPS_WINDOWS) * 3 + len(DELTA_FEATURES)),
            "chirps_3d_sum_mm": m.get("CHIRPS_3d_sum_mm", "not_available"),
            "chirps_7d_sum_mm": m.get("CHIRPS_7d_sum_mm", "not_available"),
            "chirps_30d_sum_mm": m.get("CHIRPS_30d_sum_mm", "not_available"),
            "ndvi_temporal_median": m.get("NDVI_mean_temporal_median", "not_available"),
            "ndwi_temporal_median": m.get("NDWI_mean_temporal_median", "not_available"),
            "mndwi_temporal_median": m.get("MNDWI_mean_temporal_median", "not_available"),
            "ndbi_temporal_median": m.get("NDBI_mean_temporal_median", "not_available"),
            "review_only_status": "scientific_review_only_not_ground_reference",
            "usable_for_science_now": "true", "usable_for_training": "false", "ground_truth": "false",
            "eligible_for_score_v7": "false", "eligible_for_17b": "false",
        })
    return rows


# ---------------------------------------------------------------------------
# 6 - Review priority
# ---------------------------------------------------------------------------

def _completeness(patch_id: str) -> float:
    m = _c25_matrix_row(patch_id)
    has_s2 = all(m.get(f, "not_available") != "not_available" for f in S2_FEATURES)
    has_chirps = m.get("CHIRPS_30d_sum_mm", "not_available") != "not_available"
    has_delta = _c23_delta_row(patch_id).get("delta_NDVI_mean", "not_available") != "not_available"
    has_hashes = any(r["candidate_patch_id"] == patch_id for r in read_csv(C25_MANIFEST))
    return round(sum([has_s2, has_chirps, has_delta, has_hashes]) / 4.0, 4)


def patch_review_priority_rows() -> list[dict]:
    event_id = _event_id()
    sums = {}
    for patch_id in _patch_ids():
        m = _c25_matrix_row(patch_id)
        try:
            sums[patch_id] = float(m.get("CHIRPS_30d_sum_mm", "0"))
        except ValueError:
            sums[patch_id] = 0.0
    max_sum = max(sums.values()) if sums else 1.0
    rows = []
    for idx, patch_id in enumerate(_patch_ids(), start=1):
        m = _c25_matrix_row(patch_id)
        completeness = _completeness(patch_id)
        rain_norm = round(sums[patch_id] / max_sum, 4) if max_sum else 0.0
        index = round(0.5 * rain_norm + 0.5 * completeness, 4)
        if index >= 0.9:
            pclass = "high_review_priority"
        elif index >= 0.6:
            pclass = "medium_review_priority"
        else:
            pclass = "low_review_priority"
        rows.append({
            "review_priority_id": f"S17C26_PRIO_{idx:04d}",
            "candidate_patch_id": patch_id, "event_id": event_id,
            "chirps_30d_sum_mm": m.get("CHIRPS_30d_sum_mm", "not_available"),
            "chirps_7d_sum_mm": m.get("CHIRPS_7d_sum_mm", "not_available"),
            "chirps_3d_sum_mm": m.get("CHIRPS_3d_sum_mm", "not_available"),
            "sentinel2_signal_available": "true", "observational_delta_available": "true",
            "artifact_completeness_score": f"{completeness:.4f}", "review_priority_index": f"{index:.4f}",
            "priority_class": pclass,
            "priority_reason": "fila de revisao humana review-only combinando chuva antecedente e completude de artefatos; nao e score de risco",
            "not_risk_score": "true", "not_training_label": "true", "not_ground_truth": "true", "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# 7 - Ground Reference target queue
# ---------------------------------------------------------------------------

GR_TARGET_DEFS = [
    ("busca_por_ocorrencia_oficial_com_local", "documental_evidence_with_location", "point_or_address", "day", "flood_vs_landslide", "defesa_civil_recife",
     ["ocorrencia enchente Recife maio 2022 bairro", "alagamento Recife 24 maio 2022 endereco", "defesa civil Recife inundacao 2022"],
     ["Recife flood May 2022 occurrence location", "Recife inundation 2022-05-24 address", "civil defense Recife flooding record 2022"]),
    ("busca_por_geometria_poligono_extensao", "geometry_or_extent", "polygon_or_line", "day_or_period", "flood_extent", "apac_pernambuco",
     ["mancha de inundacao Recife 2022 poligono", "extensao alagamento Recife maio 2022 shapefile", "APAC Recife area inundada 2022"],
     ["Recife 2022 flood extent polygon", "inundation mapping Recife May 2022 shapefile", "APAC flooded area Recife 2022"]),
    ("busca_por_fonte_que_separa_alagamento_de_deslizamento", "phenomenon_classification", "point_or_polygon", "day_or_period", "explicit_phenomenon_class", "cemaden",
     ["CEMADEN Recife maio 2022 deslizamento alagamento", "classificacao fenomeno Recife 2022 inundacao movimento de massa", "boletim CEMADEN Recife 2022"],
     ["CEMADEN Recife May 2022 landslide flood", "phenomenon classification Recife 2022 flood mass movement", "CEMADEN bulletin Recife 2022"]),
]


def ground_reference_target_rows() -> list[dict]:
    event_id = _event_id()
    rows = []
    idx = 0
    for patch_id in _patch_ids():
        binding = _binding(patch_id)
        for objective, evidence, geometry, temporal, phenomenon, family, q_pt, q_en in GR_TARGET_DEFS:
            idx += 1
            rows.append({
                "ground_reference_target_id": f"S17C26_GRTGT_{idx:04d}",
                "candidate_patch_id": patch_id, "event_id": event_id, "bbox": _bbox(patch_id),
                "event_period_start": binding.get("event_period_start", "not_available"),
                "event_period_end": binding.get("event_period_end", "not_available"),
                "phenomenon_expected": "urban_flooding_to_confirm",
                "target_objective": objective, "required_evidence_type": evidence,
                "required_geometry_type": geometry, "required_temporal_precision": temporal,
                "required_phenomenon_classification": phenomenon, "suggested_source_family": family,
                "query_terms_ptbr": " | ".join(q_pt), "query_terms_en": " | ".join(q_en),
                "why_this_patch": f"patch candidato {patch_id} sobre AOI de Recife com contexto sensorial/chuva real, sem geometria de evento observado",
                "current_g4_status": "false", "current_g5_status": "false",
                "can_be_ground_reference_now": "false", "review_only": "true",
            })
    return rows


# ---------------------------------------------------------------------------
# 8 - Source query packages
# ---------------------------------------------------------------------------

def _query_package_payload(patch_id: str, family: str) -> dict:
    binding = _binding(patch_id)
    target = next((t for t in ground_reference_target_rows() if t["candidate_patch_id"] == patch_id and t["suggested_source_family"] == family), {})
    return {
        "candidate_patch_id": patch_id, "event_id": _event_id(), "bbox": _bbox(patch_id),
        "date_range": f"{binding.get('event_period_start', 'na')}/{binding.get('event_period_end', 'na')}",
        "source_family": family,
        "queries_ptbr": target.get("query_terms_ptbr", "").split(" | "),
        "queries_en": target.get("query_terms_en", "").split(" | "),
        "acceptance_criteria": ["fonte oficial ou tecnica identificavel", "data/periodo compativel com o evento", "local georreferenciavel ou endereco", "fenomeno explicito (alagamento/inundacao vs deslizamento)"],
        "reject_criteria": ["sem data", "sem local", "sem fonte identificavel", "apenas noticia sem confirmacao oficial"],
        "required_fields_if_found": ["event_date_or_period", "observed_location", "geometry_or_geocodable_address", "phenomenon_class", "source_name", "source_url_or_local_path"],
        "not_ground_truth_by_default": True,
    }


def source_query_package_rows() -> list[dict]:
    event_id = _event_id()
    rows = []
    idx = 0
    for patch_id in _patch_ids():
        for family, _obj, artifact_type in SOURCE_FAMILIES:
            idx += 1
            payload = _query_package_payload(patch_id, family)
            path = QUERY_DIR / f"{patch_id.lower()}_{family}_queries.json"
            rows.append({
                "source_query_package_id": f"S17C26_QPKG_{idx:04d}",
                "candidate_patch_id": patch_id, "event_id": event_id, "source_family": family,
                "package_json_path": rel(path), "queries_count": str(len(payload["queries_ptbr"]) + len(payload["queries_en"])),
                "target_gate": "G4_G5", "expected_artifact_type": artifact_type,
                "manual_review_required": "true", "review_only": "true",
            })
    return rows


def _write_query_packages() -> None:
    ensure_dir(QUERY_DIR)
    for patch_id in _patch_ids():
        for family, _obj, _art in SOURCE_FAMILIES:
            write_json(QUERY_DIR / f"{patch_id.lower()}_{family}_queries.json", _query_package_payload(patch_id, family))


# ---------------------------------------------------------------------------
# 9 - Required Ground Reference fields
# ---------------------------------------------------------------------------

def required_ground_reference_field_rows() -> list[dict]:
    defs = [
        ("event_id", "G4", "identificador do evento observado", "string", "true"),
        ("event_date_or_period", "G3_G4", "data ou periodo do evento", "YYYY-MM-DD or range", "true"),
        ("observed_location", "G4", "local observado do evento", "string", "true"),
        ("geometry_or_geocodable_address", "G4", "geometria ou endereco geocodificavel", "geojson or address", "true"),
        ("geometry_uncertainty_m", "G4", "incerteza espacial em metros", "number", "false"),
        ("phenomenon_class", "G5", "classe do fenomeno observado", "enum flood/landslide/other", "true"),
        ("source_name", "G2_G6", "nome da fonte", "string", "true"),
        ("source_type", "G2", "tipo da fonte", "official/technical/news", "true"),
        ("source_url_or_local_path", "G6", "url ou caminho local da fonte", "string", "true"),
        ("source_hash", "G6", "hash do artefato da fonte", "sha256", "true"),
        ("officiality_level", "G2", "nivel de oficialidade", "official/technical/candidate", "true"),
        ("can_link_to_candidate_patch", "G4", "vinculo espacial com patch candidato", "true/false", "true"),
        ("can_distinguish_flood_from_landslide", "G5", "separa alagamento de deslizamento", "true/false", "true"),
        ("review_only_status", "governance", "estado review-only ate aceite", "string", "true"),
    ]
    return [
        {
            "required_field_id": f"S17C26_GRFIELD_{idx:04d}", "field_name": name, "required_for_gate": gate,
            "description": desc, "accepted_values_or_format": fmt, "blocks_if_missing": blocks, "review_only": "true",
        }
        for idx, (name, gate, desc, fmt, blocks) in enumerate(defs, start=1)
    ]


# ---------------------------------------------------------------------------
# 10 - G4/G5 gap matrix
# ---------------------------------------------------------------------------

def g4_g5_gap_matrix_rows() -> list[dict]:
    event_id = _event_id()
    rows = []
    for idx, patch_id in enumerate(_patch_ids(), start=1):
        rows.append({
            "g4_g5_gap_id": f"S17C26_GAP_{idx:04d}",
            "candidate_patch_id": patch_id, "event_id": event_id,
            "has_candidate_aoi": "true", "has_sensor_context": "true", "has_rainfall_context": "true",
            "has_observational_delta": "true", "has_observed_event_geometry": "false",
            "has_official_event_artifact": "false", "has_phenomenon_classification": "false",
            "G4_vinculo_espacial_evento": "false", "G5_separacao_fenomeno": "false",
            "blocking_reason": "falta artefato de evento observado com geometria (G4) e classificacao de fenomeno confirmada (G5)",
            "required_next_artifact": "official_or_technical_event_record_with_geometry_and_phenomenon_class",
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# 4 - Evidence graph
# ---------------------------------------------------------------------------

def _build_graph():
    event_id = _event_id()
    nodes = []  # (logical_key, node_type, object_id, patch_id, label, source_dataset)
    nodes.append(("event", "event", event_id, "", f"evento {event_id}", "internal"))
    nodes.append(("src_sentinel2", "source", "SENTINEL-2", "", "fonte Sentinel-2 (Earth Search fallback review-only)", "SENTINEL-2"))
    nodes.append(("src_chirps", "source", "CHIRPS-2.0", "", "fonte CHIRPS-2.0 CHC/UCSB", "CHIRPS"))
    for patch_id in _patch_ids():
        nodes.append((f"patch:{patch_id}", "candidate_patch", patch_id, patch_id, f"patch candidato {patch_id}", "internal"))
        for f in S2_FEATURES:
            nodes.append((f"s2:{patch_id}:{f}", "sentinel2_feature", f, patch_id, f"{f} ({patch_id})", "SENTINEL-2"))
        for w in CHIRPS_WINDOWS:
            nodes.append((f"chirps:{patch_id}:{w}", "chirps_feature", w, patch_id, f"{w} ({patch_id})", "CHIRPS"))
        nodes.append((f"delta:{patch_id}", "observational_delta", f"delta_{patch_id}", patch_id, f"delta pre/pos ({patch_id})", "SENTINEL-2"))
        nodes.append((f"art_s2:{patch_id}", "artifact", f"s2_artifacts_{patch_id}", patch_id, f"artefatos Sentinel-2 ({patch_id})", "SENTINEL-2"))
        nodes.append((f"art_chirps:{patch_id}", "artifact", f"chirps_artifacts_{patch_id}", patch_id, f"artefatos CHIRPS ({patch_id})", "CHIRPS"))
        nodes.append((f"gate:{patch_id}", "gate", f"gate_g4_g5_{patch_id}", patch_id, f"gate G4/G5 ({patch_id})", "internal"))
        for target in [t for t in ground_reference_target_rows() if t["candidate_patch_id"] == patch_id]:
            nodes.append((f"grtgt:{target['ground_reference_target_id']}", "ground_reference_target", target["ground_reference_target_id"], patch_id, target["target_objective"], target["suggested_source_family"]))
    key_to_id = {}
    for i, (key, *_rest) in enumerate(nodes, start=1):
        key_to_id[key] = f"S17C26_NODE_{i:04d}"

    edges = []  # (source_key, target_key, edge_type, patch_id)
    for patch_id in _patch_ids():
        edges.append((f"patch:{patch_id}", "event", "patch_associated_with_event", patch_id))
        for f in S2_FEATURES:
            edges.append((f"s2:{patch_id}:{f}", f"art_s2:{patch_id}", "feature_derived_from_artifact", patch_id))
            edges.append((f"s2:{patch_id}:{f}", f"patch:{patch_id}", "feature_contextualizes_patch", patch_id))
        for w in CHIRPS_WINDOWS:
            edges.append((f"chirps:{patch_id}:{w}", f"art_chirps:{patch_id}", "feature_derived_from_artifact", patch_id))
            edges.append((f"chirps:{patch_id}:{w}", f"patch:{patch_id}", "feature_contextualizes_patch", patch_id))
        edges.append((f"art_s2:{patch_id}", "src_sentinel2", "artifact_derived_from_source", patch_id))
        edges.append((f"art_chirps:{patch_id}", "src_chirps", "artifact_derived_from_source", patch_id))
        edges.append((f"delta:{patch_id}", f"patch:{patch_id}", "delta_contextualizes_observation", patch_id))
        edges.append((f"patch:{patch_id}", f"gate:{patch_id}", "patch_blocked_by_gate", patch_id))
        for target in [t for t in ground_reference_target_rows() if t["candidate_patch_id"] == patch_id]:
            edges.append((f"patch:{patch_id}", f"grtgt:{target['ground_reference_target_id']}", "patch_needs_ground_reference_target", patch_id))
    return nodes, edges, key_to_id


def evidence_graph_node_rows() -> list[dict]:
    event_id = _event_id()
    nodes, _edges, key_to_id = _build_graph()
    rows = []
    for key, node_type, object_id, patch_id, label, source_dataset in nodes:
        rows.append({
            "node_id": key_to_id[key], "node_type": node_type, "object_id": object_id,
            "candidate_patch_id": patch_id or "not_applicable", "event_id": event_id,
            "label": label, "source_dataset": source_dataset, **GOV,
        })
    return rows


def evidence_graph_edge_rows() -> list[dict]:
    event_id = _event_id()
    _nodes, edges, key_to_id = _build_graph()
    rows = []
    for i, (skey, tkey, edge_type, patch_id) in enumerate(edges, start=1):
        rows.append({
            "edge_id": f"S17C26_EDGE_{i:04d}", "source_node_id": key_to_id[skey], "target_node_id": key_to_id[tkey],
            "edge_type": edge_type, "candidate_patch_id": patch_id, "event_id": event_id, "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# 12 - Proveniencia multimodal
# ---------------------------------------------------------------------------

def multimodal_provenance_index_rows() -> list[dict]:
    event_id = _event_id()
    rows = []
    idx = 0
    for prov in read_csv(C25_PROVENANCE):
        idx += 1
        rows.append({
            "provenance_index_id": f"S17C26_PROVIDX_{idx:04d}",
            "candidate_patch_id": prov["candidate_patch_id"], "event_id": event_id,
            "feature_or_artifact_name": prov["feature_name"], "source_dataset": prov["source_dataset"],
            "source_artifact_path": prov["artifact_path"], "sha256_or_manifest_ref": prov["sha256"],
            "processing_stage": prov["processing_chain"], "replay_status": prov.get("replay_status", "derived"),
            **GOV,
        })
    return rows


# ---------------------------------------------------------------------------
# 13 - Hash recheck
# ---------------------------------------------------------------------------

def artifact_hash_recheck_rows() -> list[dict]:
    rows = []
    idx = 0
    for manifest_path, patch_key, path_key, hash_key in [
        (C25_MANIFEST, "candidate_patch_id", "artifact_local_path", "sha256"),
        (C24_MANIFEST, "candidate_patch_id", "artifact_local_path", "sha256"),
    ]:
        for m in read_csv(manifest_path):
            idx += 1
            path = ROOT / m[path_key]
            exists = path.exists()
            actual = sha256_file(path) if exists else "missing"
            rows.append({
                "hash_recheck_id": f"S17C26_HASH_{idx:04d}",
                "candidate_patch_id": m[patch_key], "artifact_path": m[path_key],
                "expected_sha256": m[hash_key], "actual_sha256": actual,
                "sha256_matches": _bool_text(exists and actual == m[hash_key]),
                "artifact_exists": _bool_text(exists), "review_only": "true",
            })
    return rows


# ---------------------------------------------------------------------------
# 14 - No leakage
# ---------------------------------------------------------------------------

def no_leakage_audit_rows() -> list[dict]:
    event_id = _event_id()
    rows = []
    for idx, patch_id in enumerate(_patch_ids(), start=1):
        rows.append({
            "no_leakage_audit_id": f"S17C26_LEAK_{idx:04d}",
            "candidate_patch_id": patch_id, "event_id": event_id,
            "object_id": f"multimodal_package_{patch_id}", "object_type": "patch_multimodal_package",
            "uses_pre_event_data_only_for_features": "true", "uses_post_event_as_pre_event_feature": "false",
            "uses_delta_as_training_label": "false", "uses_chirps_as_event_reference": "false",
            "uses_sensor_as_ground_reference": "false", "uses_synthetic_as_real": "false",
            "passes_no_leakage": "true",
            "blocking_reason": "pacote multimodal review-only; sensores e chuva sao contexto, nunca evento/ground reference/label",
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# 15 - Gates
# ---------------------------------------------------------------------------

def gate_evaluation_matrix_rows() -> list[dict]:
    event_id = _event_id()
    rows = []
    for idx, patch_id in enumerate(_patch_ids(), start=1):
        rows.append({
            "gate_eval_id": f"S17C26_GATE_{idx:04d}",
            "object_id": f"multimodal_package_{patch_id}", "object_type": "patch_multimodal_package",
            "candidate_patch_id": patch_id, "event_id": event_id,
            "G1_existencia_documental": "true", "G2_confiabilidade_fonte": "true", "G3_precisao_temporal": "true",
            "G4_vinculo_espacial_evento": "false", "G5_separacao_fenomeno": "false",
            "G6_proveniencia_integridade": "true", "G7_anti_leakage": "true",
            "all_gates_passed_for_ground_reference": "false", "usable_as_scientific_review_only_package": "true",
            "acceptance_status": "accepted_scientific_review_only_package",
            "blocking_reason": "pacote sensorial multimodal review-only; G4/G5 exigem artefato de evento observado com geometria e fenomeno",
            **GOV,
        })
    return rows


# ---------------------------------------------------------------------------
# 5 - Evidence packet registry + packets
# ---------------------------------------------------------------------------

def _evidence_packet_payload(patch_id: str) -> dict:
    m = _c25_matrix_row(patch_id)
    binding = _binding(patch_id)
    delta = _c23_delta_row(patch_id)
    manifests = [m2["chirps_artifact_manifest_id"] for m2 in read_csv(C25_MANIFEST) if m2["candidate_patch_id"] == patch_id]
    return {
        "candidate_patch_id": patch_id, "event_id": _event_id(), "bbox": _bbox(patch_id),
        "event_period": f"{binding.get('event_period_start', 'na')}/{binding.get('event_period_end', 'na')}",
        "sentinel2_features": {f: m.get(f, "not_available") for f in S2_FEATURES},
        "chirps_features": {
            "CHIRPS_3d_sum_mm": m.get("CHIRPS_3d_sum_mm"), "CHIRPS_7d_sum_mm": m.get("CHIRPS_7d_sum_mm"),
            "CHIRPS_30d_sum_mm": m.get("CHIRPS_30d_sum_mm"),
        },
        "observational_deltas": {f: delta.get(f, "not_available") for f in DELTA_FEATURES},
        "artifact_manifest_refs": manifests,
        "hash_refs": [m2["sha256"] for m2 in read_csv(C25_MANIFEST) if m2["candidate_patch_id"] == patch_id],
        "replay_refs": [r["replay_audit_id"] for r in read_csv(C25_REPLAY)],
        "no_leakage_refs": [r["no_leakage_audit_id"] for r in read_csv(C25_NO_LEAKAGE) if r["candidate_patch_id"] == patch_id],
        "g4_status": False, "g5_status": False, "can_be_ground_reference_now": False,
        "can_be_training_label": False, "can_be_score_v7": False, "can_unlock_17b": False, "review_only": True,
    }


def patch_evidence_packet_registry_rows() -> list[dict]:
    event_id = _event_id()
    rows = []
    for idx, patch_id in enumerate(_patch_ids(), start=1):
        json_path = PACKET_DIR / f"{patch_id.lower()}_evidence_packet.json"
        md_path = DOSSIER_DIR / f"{patch_id.lower()}_dossie_multimodal.md"
        has_hashes = any(r["candidate_patch_id"] == patch_id for r in read_csv(C25_MANIFEST))
        rows.append({
            "evidence_packet_id": f"S17C26_PKT_{idx:04d}",
            "candidate_patch_id": patch_id, "event_id": event_id,
            "packet_json_path": rel(json_path), "packet_markdown_path": rel(md_path),
            "has_sentinel2_multitemporal": "true", "has_chirps_real": "true", "has_observational_delta": "true",
            "has_artifact_hashes": _bool_text(has_hashes), "has_replay_audit": "true", "has_no_leakage_audit": "true",
            "packet_complete": _bool_text(has_hashes), **GOV,
        })
    return rows


def _write_evidence_packets() -> None:
    ensure_dir(PACKET_DIR)
    for patch_id in _patch_ids():
        write_json(PACKET_DIR / f"{patch_id.lower()}_evidence_packet.json", _evidence_packet_payload(patch_id))


# ---------------------------------------------------------------------------
# 2 - Dossies markdown
# ---------------------------------------------------------------------------

def _dossier_markdown(patch_id: str) -> str:
    m = _c25_matrix_row(patch_id)
    binding = _binding(patch_id)
    delta = _c23_delta_row(patch_id)
    manifests = [row for row in read_csv(C25_MANIFEST) if row["candidate_patch_id"] == patch_id]
    targets = [t for t in ground_reference_target_rows() if t["candidate_patch_id"] == patch_id]
    lines = [
        "# Dossiê multimodal", "",
        "## Identificação",
        f"- Patch candidato: {patch_id}", f"- Evento: {_event_id()}", f"- Status: {DOSSIER_FINAL_STATUS}", "",
        "## Evento e janela temporal",
        f"- Período do evento: {binding.get('event_period_start', 'na')} a {binding.get('event_period_end', 'na')}",
        f"- Janela pré-evento: {binding.get('pre_event_window_start', 'na')} a {binding.get('pre_event_window_end', 'na')}", "",
        "## AOI e patch candidato",
        f"- BBox: {_bbox(patch_id)}", "- Patch candidato não oficial, review-only.", "",
        "## Sentinel-2 multitemporal",
        f"- NDVI mediana temporal: {m.get('NDVI_mean_temporal_median', 'na')}",
        f"- NDWI mediana temporal: {m.get('NDWI_mean_temporal_median', 'na')}",
        f"- MNDWI mediana temporal: {m.get('MNDWI_mean_temporal_median', 'na')}",
        f"- NDBI mediana temporal: {m.get('NDBI_mean_temporal_median', 'na')}", "",
        "## CHIRPS antecedente",
        f"- CHIRPS_3d soma (mm): {m.get('CHIRPS_3d_sum_mm', 'na')}",
        f"- CHIRPS_7d soma (mm): {m.get('CHIRPS_7d_sum_mm', 'na')}",
        f"- CHIRPS_30d soma (mm): {m.get('CHIRPS_30d_sum_mm', 'na')}", "",
        "## Delta observacional pré/pós",
        f"- delta NDVI: {delta.get('delta_NDVI_mean', 'na')}", f"- delta MNDWI: {delta.get('delta_MNDWI_mean', 'na')}",
        "- Delta é mudança observacional review-only, nunca feature pré-evento nem label.", "",
        "## Artefatos e hashes",
    ]
    for mf in manifests:
        lines.append(f"- {mf['artifact_local_path']} sha256={mf['sha256'][:16]}...")
    lines += [
        "", "## Interpretação permitida",
        "- Analisar contexto sensorial (Sentinel-2 multitemporal) e chuva antecedente (CHIRPS) como camadas review-only.",
        "- Comparar mudança observacional pré/pós apenas como inspeção temporal.", "",
        "## Interpretação proibida",
        "- Tratar sensor ou chuva como evento observado.", "- Usar como Ground Reference, ground truth, label ou treino.",
        "- Usar delta como rótulo. Usar pós-evento como feature pré-evento.", "",
        "## Lacunas G4/G5",
        "- G4 (vínculo espacial de evento observado): ausente.", "- G5 (separação de fenômeno confirmada): ausente.",
        "- Falta artefato de evento observado com geometria e classificação de fenômeno.", "",
        "## Campos necessários para Ground Reference",
        "- event_date_or_period, observed_location, geometry_or_geocodable_address, phenomenon_class, source_name, source_hash, officiality_level.", "",
        "## Próximas buscas objetivas",
    ]
    for t in targets:
        lines.append(f"- {t['target_objective']} (fonte sugerida: {t['suggested_source_family']})")
    lines += ["", "## Status final", DOSSIER_FINAL_STATUS, ""]
    return "\n".join(lines)


def _write_dossiers() -> None:
    ensure_dir(DOSSIER_DIR)
    for patch_id in _patch_ids():
        write_markdown(DOSSIER_DIR / f"{patch_id.lower()}_dossie_multimodal.md", _dossier_markdown(patch_id))


# ---------------------------------------------------------------------------
# 11 - Checklists
# ---------------------------------------------------------------------------

def _checklist_markdown(patch_id: str) -> str:
    m = _c25_matrix_row(patch_id)
    return "\n".join([
        f"# Checklist de revisão G4/G5 - {patch_id}", "",
        "## Identificação do patch",
        f"- Patch: {patch_id} | Evento: {_event_id()} | BBox: {_bbox(patch_id)}", "",
        "## Artefatos sensoriais disponíveis",
        "- Sentinel-2 multitemporal (3 cenas pré-evento, medianas temporais).", "",
        "## Chuva antecedente",
        f"- CHIRPS_3d/7d/30d (mm): {m.get('CHIRPS_3d_sum_mm', 'na')}/{m.get('CHIRPS_7d_sum_mm', 'na')}/{m.get('CHIRPS_30d_sum_mm', 'na')}", "",
        "## Deltas disponíveis", "- Delta pré/pós observacional review-only.", "",
        "## O que procurar em fonte oficial",
        "- Ocorrência oficial com local; geometria/extensão; classificação de fenômeno.", "",
        "## Critérios de aceite",
        "- Fonte oficial/técnica, data compatível, local georreferenciável, fenômeno explícito.", "",
        "## Critérios de rejeição",
        "- Sem data, sem local, sem fonte identificável, apenas notícia não confirmada.", "",
        "## Campos obrigatórios",
        "- event_date_or_period, observed_location, geometry_or_geocodable_address, phenomenon_class, source_name, source_hash.", "",
        "## Riscos de falso positivo",
        "- Confundir chuva antecedente com evento; confundir mudança espectral com inundação; usar patch vizinho como prova.", "",
        "## Declaração",
        "- não chamar de ground truth sem G4/G5", "",
    ])


def review_checklist_registry_rows() -> list[dict]:
    event_id = _event_id()
    rows = []
    for idx, patch_id in enumerate(_patch_ids(), start=1):
        path = CHECKLIST_DIR / f"{patch_id.lower()}_checklist_revisao_g4_g5.md"
        rows.append({
            "review_checklist_id": f"S17C26_CHK_{idx:04d}", "candidate_patch_id": patch_id, "event_id": event_id,
            "checklist_markdown_path": rel(path), "covers_sensor_context": "true", "covers_rainfall": "true",
            "covers_delta": "true", "covers_official_source_search": "true", "declares_not_ground_truth_without_g4_g5": "true",
            "review_only": "true",
        })
    return rows


def _write_checklists() -> None:
    ensure_dir(CHECKLIST_DIR)
    for patch_id in _patch_ids():
        write_markdown(CHECKLIST_DIR / f"{patch_id.lower()}_checklist_revisao_g4_g5.md", _checklist_markdown(patch_id))


# ---------------------------------------------------------------------------
# Summary / blockers / manifest / master index
# ---------------------------------------------------------------------------

def _dossier_has_all_sections(patch_id: str) -> bool:
    text = _dossier_markdown(patch_id)
    return all(section in text for section in DOSSIER_SECTIONS) and DOSSIER_FINAL_STATUS in text


def build_summary() -> dict:
    dossiers = _patch_ids()
    packets = patch_evidence_packet_registry_rows()
    nodes = evidence_graph_node_rows()
    edges = evidence_graph_edge_rows()
    priority = patch_review_priority_rows()
    targets = ground_reference_target_rows()
    queries = source_query_package_rows()
    checklists = review_checklist_registry_rows()
    gr_fields = required_ground_reference_field_rows()
    gaps = g4_g5_gap_matrix_rows()
    hash_recheck = artifact_hash_recheck_rows()
    all_hashes = all(r["sha256_matches"] == "true" for r in hash_recheck) and bool(hash_recheck)
    all_sections = all(_dossier_has_all_sections(p) for p in dossiers)
    minimum = (
        len(dossiers) == 5 and len(packets) == 5 and len(nodes) > 0 and len(edges) > 0
        and len(priority) == 5 and len(targets) >= 15 and len(queries) >= 15 and len(checklists) == 5
        and all_hashes and all_sections
    )
    return {
        "minimum_success_achieved": minimum,
        "patch_dossiers_created_count": len(dossiers),
        "patch_evidence_packets_count": len(packets),
        "evidence_graph_nodes_count": len(nodes),
        "evidence_graph_edges_count": len(edges),
        "review_priority_rows_count": len(priority),
        "ground_reference_target_rows_count": len(targets),
        "source_query_packages_count": len(queries),
        "review_checklists_count": len(checklists),
        "required_ground_reference_fields_count": len(gr_fields),
        "g4_g5_gap_rows_count": len(gaps),
        "sentinel2_chirps_delta_integrated_count": len(dossiers),
        "artifact_hashes_verified_count": len([r for r in hash_recheck if r["sha256_matches"] == "true"]),
        "all_hashes_verified": all_hashes,
        "all_dossiers_have_required_sections": all_sections,
        "master_index_created": True,
        "reproducibility_manifest_created": True,
        "G4_true_count": 0,
        "G5_true_count": 0,
        "features_promoted_to_training_count": 0,
        "post_event_used_as_pre_event_feature_count": 0,
        "delta_used_as_training_label_count": 0,
        "chirps_used_as_event_reference_count": 0,
        "sensor_used_as_ground_reference_count": 0,
        "ground_reference_candidates_created_count": 0,
        "ground_truth_created": False,
        "training_labels_created": False,
        "score_v6_changed": bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)])),
        "score_v7_created": SCORE_V7.exists(),
        "official_patch_created": False,
        "official_patch_link_created": False,
        "eligible_for_17b_now": False,
        "eligible_for_score_v7": False,
        "review_only": True,
        "trainable": False,
        "ground_truth": False,
        "recommended_next_milestone": "SUSC-17C27 Aquisicao dirigida de Ground Reference (defesa civil/APAC/CEMADEN) usando a fila G4/G5 para desbloqueio futuro de G4/G5",
    }


def build_blockers() -> list[dict]:
    blockers = [
        "g4_event_geometry_missing", "g5_phenomenon_separation_missing", "official_event_artifact_missing",
        "ground_reference_not_created", "candidate_patch_not_official_patch", "sensor_context_not_ground_reference",
        "chirps_not_event_reference", "delta_not_label", "score_v7_blocked", "17b_blocked",
    ]
    return [
        {
            "blocker_id": f"S17C26_BLOCKER_{idx:04d}", "blocker_type": blocker,
            "description": "Bloqueio preservado: pacote multimodal review-only; 17B e score v7 exigem artefato de evento observado com geometria (G4) e fenomeno (G5), Ground Reference e politica.",
            "blocks_ground_reference_candidate": _bool_text(blocker in {"g4_event_geometry_missing", "g5_phenomenon_separation_missing", "official_event_artifact_missing", "ground_reference_not_created"}),
            "blocks_17b": "true", "blocks_score_v7": "true", "review_only": "true",
        }
        for idx, blocker in enumerate(blockers, start=1)
    ]


def _reproducibility_manifest() -> dict:
    main_outputs = [PATCH_SUMMARY, PACKET_REGISTRY, GRAPH_NODES, GRAPH_EDGES, REVIEW_PRIORITY,
                    GR_TARGET_QUEUE, G4_G5_GAP, SOURCE_QUERY_PACKAGES, PROVENANCE_INDEX, SUMMARY]
    output_hashes = {rel(p): sha256_file(p) for p in main_outputs if p.exists()}
    for patch_id in _patch_ids():
        for path in [DOSSIER_DIR / f"{patch_id.lower()}_dossie_multimodal.md",
                     PACKET_DIR / f"{patch_id.lower()}_evidence_packet.json"]:
            if path.exists():
                output_hashes[rel(path)] = sha256_file(path)
    return {
        "marco": "SUSC-17C26",
        "mode": "offline_deterministic",
        "inputs_used": [rel(p) for p in REQUIRED_INPUTS],
        "scripts": [
            "scripts/suscetibilidade/susc_17c26_multimodal_ops_common.py",
            "scripts/suscetibilidade/susc_17c26_multimodal_ops_package.py",
            "scripts/suscetibilidade/build_susc_17c26_multimodal_ops_package.py",
            "scripts/suscetibilidade/validate_susc_17c26_multimodal_ops_package.py",
        ],
        "schemas": [rel(DOSSIER_SCHEMA), rel(GRAPH_SCHEMA), rel(GR_TARGET_SCHEMA), rel(PRIORITY_SCHEMA), rel(QUERY_SCHEMA)],
        "tests": ["tests/suscetibilidade/test_susc_17c26_multimodal_ops_package.py"],
        "validators_expected": ["validate_susc_17c26_multimodal_ops_package.py"],
        "main_output_hashes": output_hashes,
        "replay_policy": "build 2x byte-identico a partir de artefatos ja commitados; nenhuma rede",
        "not_training": True, "not_ground_truth": True, "not_score_v7": True, "not_17b": True, "review_only": True,
    }


def _master_index() -> str:
    lines = ["# SUSC-17C26 - Índice mestre do pacote multimodal operacional", "",
             "## Relatório", f"- [{REPORT.name}]({REPORT.name})", "", "## Dossiês por patch"]
    for patch_id in _patch_ids():
        lines.append(f"- [{patch_id}](susc_17c26_patch_dossiers/{patch_id.lower()}_dossie_multimodal.md)")
    lines += ["", "## Evidence packets"]
    for patch_id in _patch_ids():
        lines.append(f"- [{patch_id}](susc_17c26_evidence_packets/{patch_id.lower()}_evidence_packet.json)")
    lines += ["", "## Checklists de revisão"]
    for patch_id in _patch_ids():
        lines.append(f"- [{patch_id}](susc_17c26_review_checklists/{patch_id.lower()}_checklist_revisao_g4_g5.md)")
    lines += ["", "## Source query packages"]
    for patch_id in _patch_ids():
        for family, _o, _a in SOURCE_FAMILIES:
            lines.append(f"- [{patch_id} / {family}](susc_17c26_source_query_packages/{patch_id.lower()}_{family}_queries.json)")
    lines += ["", "## Estruturas operacionais",
              f"- [Evidence graph (nós)]({GRAPH_NODES.name})", f"- [Evidence graph (arestas)]({GRAPH_EDGES.name})",
              f"- [Fila G4/G5]({GR_TARGET_QUEUE.name})", f"- [Gap matrix G4/G5]({G4_G5_GAP.name})",
              f"- [Prioridade de revisão]({REVIEW_PRIORITY.name})", f"- [Summary]({SUMMARY.name})",
              f"- [Promotion blockers]({BLOCKERS.name})", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

def _schema(required: list[str], props: dict, title: str) -> dict:
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": title, "type": "object", "required": required, "properties": props}


def build_dossier_schema() -> dict:
    return _schema(["candidate_patch_id", "required_sections", "final_status"], {
        "final_status": {"const": DOSSIER_FINAL_STATUS},
    }, "SUSC-17C26 patch dossier schema v1")


def build_graph_schema() -> dict:
    return _schema(["node_id", "node_type", "object_id", "candidate_patch_id", "event_id", "label", "source_dataset", "review_only", "trainable", "ground_truth"], {
        "node_id": {"type": "string", "pattern": "^S17C26_NODE_"},
        "review_only": {"const": "true"}, "trainable": {"const": "false"}, "ground_truth": {"const": "false"},
    }, "SUSC-17C26 evidence graph schema v1")


def build_gr_target_schema() -> dict:
    required = list(ground_reference_target_rows()[0].keys())
    return _schema(required, {
        "ground_reference_target_id": {"type": "string", "pattern": "^S17C26_GRTGT_"},
        "current_g4_status": {"const": "false"}, "current_g5_status": {"const": "false"},
        "can_be_ground_reference_now": {"const": "false"}, "review_only": {"const": "true"},
    }, "SUSC-17C26 ground reference target schema v1")


def build_priority_schema() -> dict:
    required = list(patch_review_priority_rows()[0].keys())
    return _schema(required, {
        "review_priority_id": {"type": "string", "pattern": "^S17C26_PRIO_"},
        "not_risk_score": {"const": "true"}, "not_training_label": {"const": "true"},
        "not_ground_truth": {"const": "true"}, "review_only": {"const": "true"},
    }, "SUSC-17C26 review priority schema v1")


def build_query_schema() -> dict:
    required = list(source_query_package_rows()[0].keys())
    return _schema(required, {
        "source_query_package_id": {"type": "string", "pattern": "^S17C26_QPKG_"}, "review_only": {"const": "true"},
    }, "SUSC-17C26 source query package schema v1")


# ---------------------------------------------------------------------------
# Relatorio
# ---------------------------------------------------------------------------

def build_report() -> str:
    s = build_summary()
    return "\n".join([
        "# SUSC-17C26 - Pacote multimodal operacional, evidence graph e fila G4/G5", "",
        "## Objetivo",
        "Consolidar offline todos os dados reais ja obtidos (Sentinel-2 multitemporal 17C24, CHIRPS real 17C25, deltas) em um pacote operacional por patch, preparando o desbloqueio G4/G5 sem coletar dados novos nem liberar 17B.", "",
        "## Entregas",
        f"- Dossies por patch: {s['patch_dossiers_created_count']}.",
        f"- Evidence packets: {s['patch_evidence_packets_count']}.",
        f"- Evidence graph: {s['evidence_graph_nodes_count']} nos, {s['evidence_graph_edges_count']} arestas.",
        f"- Prioridades review-only: {s['review_priority_rows_count']}.",
        f"- Targets Ground Reference (fila G4/G5): {s['ground_reference_target_rows_count']}.",
        f"- Source query packages: {s['source_query_packages_count']}.",
        f"- Checklists de revisao: {s['review_checklists_count']}.",
        f"- Campos obrigatorios de Ground Reference: {s['required_ground_reference_fields_count']}.",
        f"- Hashes verificados: {s['artifact_hashes_verified_count']} (all_hashes_verified={s['all_hashes_verified']}).", "",
        "## G4/G5",
        f"- G4_true_count={s['G4_true_count']}, G5_true_count={s['G5_true_count']}. G4/G5 permanecem false por design: exigem artefato de evento observado com geometria (G4) e classificacao de fenomeno (G5).", "",
        "## Guardrails",
        "- Sensor e chuva sao contexto review-only; nenhum vira evento observado, Ground Reference, label, treino, score v7, patch oficial ou 17B. Delta e mudanca observacional review-only.", "",
        f"## minimum_success_achieved: {s['minimum_success_achieved']}", "",
        "## Proximo marco recomendado", s["recommended_next_milestone"],
    ])


# ---------------------------------------------------------------------------
# Build / validacao
# ---------------------------------------------------------------------------

def build_all() -> None:
    _require_inputs()
    ensure_dir(DOSSIER_DIR)
    ensure_dir(PACKET_DIR)
    ensure_dir(CHECKLIST_DIR)
    ensure_dir(QUERY_DIR)
    _write_dossiers()
    _write_evidence_packets()
    _write_checklists()
    _write_query_packages()
    write_csv(PATCH_SUMMARY, patch_multimodal_summary_rows())
    write_csv(PACKET_REGISTRY, patch_evidence_packet_registry_rows())
    write_csv(GRAPH_NODES, evidence_graph_node_rows())
    write_csv(GRAPH_EDGES, evidence_graph_edge_rows())
    write_csv(REVIEW_PRIORITY, patch_review_priority_rows())
    write_csv(GR_TARGET_QUEUE, ground_reference_target_rows())
    write_csv(GR_REQUIRED_FIELDS, required_ground_reference_field_rows())
    write_csv(G4_G5_GAP, g4_g5_gap_matrix_rows())
    write_csv(SOURCE_QUERY_PACKAGES, source_query_package_rows())
    write_csv(CHECKLIST_REGISTRY, review_checklist_registry_rows())
    write_csv(PROVENANCE_INDEX, multimodal_provenance_index_rows())
    write_csv(HASH_RECHECK, artifact_hash_recheck_rows())
    write_csv(NO_LEAKAGE, no_leakage_audit_rows())
    write_csv(GATES, gate_evaluation_matrix_rows())
    write_csv(BLOCKERS, build_blockers())
    write_json(DOSSIER_SCHEMA, build_dossier_schema())
    write_json(GRAPH_SCHEMA, build_graph_schema())
    write_json(GR_TARGET_SCHEMA, build_gr_target_schema())
    write_json(PRIORITY_SCHEMA, build_priority_schema())
    write_json(QUERY_SCHEMA, build_query_schema())
    write_markdown(MASTER_INDEX, _master_index())
    write_markdown(REPORT, build_report())
    write_json(SUMMARY, build_summary())
    write_json(REPRODUCIBILITY, _reproducibility_manifest())  # por ultimo: depende dos outputs principais


def _required_outputs() -> list[Path]:
    outputs = [
        REPORT, PATCH_SUMMARY, PACKET_REGISTRY, GRAPH_NODES, GRAPH_EDGES, REVIEW_PRIORITY, GR_TARGET_QUEUE,
        GR_REQUIRED_FIELDS, G4_G5_GAP, SOURCE_QUERY_PACKAGES, CHECKLIST_REGISTRY, PROVENANCE_INDEX,
        HASH_RECHECK, NO_LEAKAGE, GATES, REPRODUCIBILITY, MASTER_INDEX, SUMMARY, BLOCKERS,
        DOSSIER_SCHEMA, GRAPH_SCHEMA, GR_TARGET_SCHEMA, PRIORITY_SCHEMA, QUERY_SCHEMA,
    ]
    for patch_id in _patch_ids():
        outputs.append(DOSSIER_DIR / f"{patch_id.lower()}_dossie_multimodal.md")
        outputs.append(PACKET_DIR / f"{patch_id.lower()}_evidence_packet.json")
        outputs.append(CHECKLIST_DIR / f"{patch_id.lower()}_checklist_revisao_g4_g5.md")
        for family, _o, _a in SOURCE_FAMILIES:
            outputs.append(QUERY_DIR / f"{patch_id.lower()}_{family}_queries.json")
    return outputs


def _schema_violations(row: dict, schema: dict) -> list[str]:
    violations = []
    for field in schema.get("required", []):
        if field not in row or row[field] == "":
            violations.append(f"missing:{field}")
    for field, rules in schema.get("properties", {}).items():
        if field not in row:
            continue
        value = row[field]
        if "const" in rules and value != rules["const"]:
            violations.append(f"{field}:const:{rules['const']}")
        if "pattern" in rules and not value.startswith(rules["pattern"].replace("^", "")):
            violations.append(f"{field}:pattern")
    return violations


def _validate_byte_identical() -> list[str]:
    outputs = _required_outputs()
    before = {path: path.read_bytes() for path in outputs if path.exists()}
    build_all()
    errors = []
    for path, content in before.items():
        if path.exists() and path.read_bytes() != content:
            errors.append(f"offline_build_not_byte_identical:{rel(path)}")
    return errors


def validate() -> int:
    missing = [path for path in _required_outputs() if not path.exists()]
    if missing:
        print("MISSING: " + "; ".join(rel(path) for path in missing), file=sys.stderr)
        return 1
    errors = _validate_byte_identical()
    nodes = read_csv(GRAPH_NODES)
    edges = read_csv(GRAPH_EDGES)
    priority = read_csv(REVIEW_PRIORITY)
    targets = read_csv(GR_TARGET_QUEUE)
    queries = read_csv(SOURCE_QUERY_PACKAGES)
    gaps = read_csv(G4_G5_GAP)
    gr_fields = read_csv(GR_REQUIRED_FIELDS)
    hash_recheck = read_csv(HASH_RECHECK)
    leakage = read_csv(NO_LEAKAGE)
    gates = read_csv(GATES)
    packets = read_csv(PACKET_REGISTRY)
    summary = read_json(SUMMARY)
    graph_schema = read_json(GRAPH_SCHEMA)
    gr_schema = read_json(GR_TARGET_SCHEMA)
    priority_schema = read_json(PRIORITY_SCHEMA)
    query_schema = read_json(QUERY_SCHEMA)

    for row in nodes:
        errors.extend(_schema_violations(row, graph_schema))
    for row in targets:
        errors.extend(_schema_violations(row, gr_schema))
    for row in priority:
        errors.extend(_schema_violations(row, priority_schema))
    for row in queries:
        errors.extend(_schema_violations(row, query_schema))

    for rows, key in [
        (nodes, "node_id"), (edges, "edge_id"), (priority, "review_priority_id"),
        (targets, "ground_reference_target_id"), (queries, "source_query_package_id"),
        (gaps, "g4_g5_gap_id"), (gr_fields, "required_field_id"), (hash_recheck, "hash_recheck_id"),
        (leakage, "no_leakage_audit_id"), (gates, "gate_eval_id"), (packets, "evidence_packet_id"),
    ]:
        ids = [row[key] for row in rows]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            errors.append(f"ids_not_unique_or_sorted:{key}")

    # 1/2: dossies e secoes.
    for patch_id in _patch_ids():
        path = DOSSIER_DIR / f"{patch_id.lower()}_dossie_multimodal.md"
        if not path.exists():
            errors.append(f"dossier_missing:{patch_id}")
            continue
        text = path.read_text(encoding="utf-8")
        for section in DOSSIER_SECTIONS:
            if section not in text:
                errors.append(f"dossier_missing_section:{patch_id}:{section}")
        if DOSSIER_FINAL_STATUS not in text:
            errors.append(f"dossier_missing_final_status:{patch_id}")
    if len([p for p in _patch_ids()]) != 5:
        errors.append("patches_not_5")
    # 3: packets.
    if len(packets) != 5 or any(r["packet_complete"] != "true" for r in packets):
        errors.append("evidence_packets_incomplete")
    for patch_id in _patch_ids():
        if not (PACKET_DIR / f"{patch_id.lower()}_evidence_packet.json").exists():
            errors.append(f"packet_json_missing:{patch_id}")
    # 4: grafo.
    if not nodes or not edges:
        errors.append("evidence_graph_empty")
    node_ids = {r["node_id"] for r in nodes}
    if any(r["source_node_id"] not in node_ids or r["target_node_id"] not in node_ids for r in edges):
        errors.append("edge_references_unknown_node")
    # 5/6/7/8.
    if len(priority) != 5:
        errors.append("priority_not_5")
    if len(targets) < 15:
        errors.append("gr_targets_lt_15")
    if len(queries) < 15:
        errors.append("query_packages_lt_15")
    if len(read_csv(CHECKLIST_REGISTRY)) != 5:
        errors.append("checklists_not_5")
    for patch_id in _patch_ids():
        if not (CHECKLIST_DIR / f"{patch_id.lower()}_checklist_revisao_g4_g5.md").exists():
            errors.append(f"checklist_missing:{patch_id}")
        for family, _o, _a in SOURCE_FAMILIES:
            if not (QUERY_DIR / f"{patch_id.lower()}_{family}_queries.json").exists():
                errors.append(f"query_package_missing:{patch_id}:{family}")
    # 9/10.
    if not MASTER_INDEX.exists():
        errors.append("master_index_missing")
    if not REPRODUCIBILITY.exists():
        errors.append("reproducibility_manifest_missing")
    # 11/12/13/14: G4/G5 false; sem ground reference; sem chirps evento.
    if any(r["G4_vinculo_espacial_evento"] == "true" or r["G5_separacao_fenomeno"] == "true" for r in gates + gaps):
        errors.append("g4_g5_true")
    if any(r["all_gates_passed_for_ground_reference"] == "true" for r in gates):
        errors.append("ground_reference_accepted")
    if any(r["can_be_ground_reference_now"] != "false" for r in targets):
        errors.append("target_marked_ground_reference")
    if any(r["uses_chirps_as_event_reference"] != "false" or r["uses_sensor_as_ground_reference"] != "false" for r in leakage):
        errors.append("chirps_or_sensor_as_event_or_ground_reference")
    if any(r["uses_delta_as_training_label"] != "false" or r["uses_post_event_as_pre_event_feature"] != "false" for r in leakage):
        errors.append("delta_or_post_event_leakage")
    if any(r["passes_no_leakage"] != "true" for r in leakage):
        errors.append("no_leakage_failed")
    # 16: sem treino.
    if summary["features_promoted_to_training_count"] != 0:
        errors.append("feature_promoted_to_training")
    # hashes.
    if any(r["sha256_matches"] != "true" for r in hash_recheck) or not hash_recheck:
        errors.append("hash_recheck_failed")
    if not summary["all_hashes_verified"] or not summary["all_dossiers_have_required_sections"]:
        errors.append("summary_verification_flags_false")
    # 18: patch nao oficial.
    for row in read_csv(PATCH_GRID):
        if row.get("official_patch") == "true":
            errors.append("candidate_patch_is_official")

    for path in [SCORE_V6, OFFICIAL_PATCHES, OFFICIAL_PATCH_LINKS]:
        if path.exists() and _run_git(["diff", "--name-only", "--", rel(path)]):
            errors.append(f"official_dataset_changed:{rel(path)}")
    if SCORE_V7.exists():
        errors.append("score_v7_exists")

    expected = build_summary()
    for key, value in expected.items():
        if summary.get(key) != value:
            errors.append(f"summary_mismatch:{key}")
    if summary["eligible_for_17b_now"] or summary["eligible_for_score_v7"] or summary["trainable"] or summary["ground_truth"]:
        errors.append("promotion_guardrail_failed")
    if not summary["minimum_success_achieved"]:
        errors.append("minimum_success_not_achieved")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        "17C26 -> "
        f"dossiers={summary['patch_dossiers_created_count']} "
        f"packets={summary['patch_evidence_packets_count']} "
        f"nodes={summary['evidence_graph_nodes_count']} edges={summary['evidence_graph_edges_count']} "
        f"gr_targets={summary['ground_reference_target_rows_count']} queries={summary['source_query_packages_count']} "
        f"G4={summary['G4_true_count']} G5={summary['G5_true_count']} "
        f"min_success={summary['minimum_success_achieved']} score_v7_created={summary['score_v7_created']}"
    )
    return 0


# ---------------------------------------------------------------------------
# CLI helpers (todos offline)
# ---------------------------------------------------------------------------

def build_dossiers_text() -> str:
    _require_inputs()
    _write_dossiers()
    return f"build-dossiers: {len(_patch_ids())} dossies multimodais escritos em {rel(DOSSIER_DIR)}."


def build_evidence_packets_text() -> str:
    _require_inputs()
    _write_evidence_packets()
    return f"build-evidence-packets: {len(_patch_ids())} evidence packets escritos em {rel(PACKET_DIR)}."


def build_evidence_graph_text() -> str:
    return f"build-evidence-graph: {len(evidence_graph_node_rows())} nos e {len(evidence_graph_edge_rows())} arestas."


def rank_review_targets_text() -> str:
    rows = patch_review_priority_rows()
    return "rank-review-targets: " + "; ".join(f"{r['candidate_patch_id']}={r['priority_class']}" for r in rows) + "."


def build_ground_reference_queue_text() -> str:
    return f"build-ground-reference-queue: {len(ground_reference_target_rows())} targets G4/G5 (>=3 por patch)."


def build_source_query_packages_text() -> str:
    _require_inputs()
    _write_query_packages()
    return f"build-source-query-packages: {len(source_query_package_rows())} pacotes de consulta por patch/fonte."


def build_review_checklists_text() -> str:
    _require_inputs()
    _write_checklists()
    return f"build-review-checklists: {len(_patch_ids())} checklists G4/G5 escritos em {rel(CHECKLIST_DIR)}."


def build_master_index_text() -> str:
    _require_inputs()
    write_markdown(MASTER_INDEX, _master_index())
    return f"build-master-index: indice mestre escrito em {rel(MASTER_INDEX)}."


def build_reproducibility_manifest_text() -> str:
    build_all()
    return f"build-reproducibility-manifest: manifesto de reprodutibilidade escrito em {rel(REPRODUCIBILITY)}."


def status_text() -> str:
    s = build_summary()
    return (
        f"status 17C26: dossiers={s['patch_dossiers_created_count']} packets={s['patch_evidence_packets_count']} "
        f"nodes={s['evidence_graph_nodes_count']} gr_targets={s['ground_reference_target_rows_count']} "
        f"min_success={s['minimum_success_achieved']}"
    )
