#!/usr/bin/env python3
"""v2bk Recife candidate reference human-review dossier and data-request pack, fail-closed.

Strictly additive. Consumes the real v2bi/v2bj intake outputs and produces a human-review
dossier for the Recife candidate reference, formal data-request templates for the missing
Charter vector/CRS (C4) and the missing Cemaden/APAC local rainfall series (C2), a C5/C6
adjudication checklist and a decision matrix. It creates no ground truth, label, negative,
training or new geometry. A request pack is not evidence; a dossier is not ground truth; a
Charter raster is not a vector; ANA river stage is not precipitation; an APAC monthly PDF is
not a station series; an INMET regional proxy is not the local Recife station; a candidate
reference is not final truth. C7 stays BLOCKED.
"""

import argparse
import csv
import hashlib
import os

DATASET_DIR = os.environ.get("DATASET_DIR", "datasets/protocolo_c")
DOCS_DIR = os.environ.get("V2BK_DOCS_DIR", "docs/protocolo_c/v2bk_recife_human_review_dossier")
REFRESH = os.environ.get("V2BK_REFRESH", "1") == "1"

CANDIDATE_ID = "REC_2022_05_24_30"
PACKAGE_ID = "ARP_v2az_0005"
EVENT_PATCH_ID = "FACT_v2at_0005"
PRODUCT_ID = "CH758_RECIFE_20220602_001"
PRODUCT_TITLE = "Landslides after effects in Recife/PE - Brazil"
ACTIVATION_ID = "758"
REQUEST_START = "2022-05-24"
REQUEST_END = "2022-06-02"

INVARIANTS = {
    "can_create_ground_truth": "false", "can_create_patch_truth": "false", "can_create_label": "false",
    "can_create_negative": "false", "can_train_model": "false",
    "request_pack_is_not_evidence": "true", "human_review_dossier_is_not_ground_truth": "true",
    "charter_raster_is_not_vector_geometry": "true", "ana_stage_is_not_precipitation": "true",
    "apac_pdf_is_not_station_series": "true", "inmet_proxy_is_not_local_station": "true",
    "candidate_reference_is_not_final_truth": "true", "raw_data_versioned": "false",
}

GATES = ("C0_PROVENANCE", "C1_TEMPORALITY", "C2_VALID_SERIES_OR_STATION", "C3_SPATIAL_ANCHOR",
         "C4_CANDIDATE_GEOMETRY", "C5_HUMAN_REVIEW", "C6_CANDIDATE_REFERENCE", "C7_FINAL_GROUND_TRUTH")

OUTPUTS = {
    "dossier_index": "v2bk_recife_human_review_dossier_index.csv",
    "charter_request": "v2bk_charter_vector_crs_request_pack.csv",
    "temporal_request": "v2bk_cemaden_apac_temporal_request_pack.csv",
    "checklist": "v2bk_c5_c6_adjudication_checklist.csv",
    "decision": "v2bk_recife_candidate_decision_matrix.csv",
    "guardrail": "v2bk_guardrail_regression.csv",
    "manifest": "v2bk_orchestrator_manifest.csv",
}

INPUTS = {
    "intake": "v2bj_recife_intake_result_summary.csv", "reconcile": "v2bj_recife_candidate_gate_reconciliation.csv",
    "queue": "v2bj_recife_candidate_reference_queue.csv", "inmet": "v2bj_inmet_proxy_availability_audit.csv",
    "charter_audit": "v2bi_charter_file_audit.csv", "charter_vector": "v2bi_charter_vector_metadata.csv",
    "charter_crs": "v2bi_charter_crs_geometry_validation.csv", "charter_readiness": "v2bi_charter_candidate_geometry_readiness.csv",
    "temporal_metrics": "v2bi_recife_temporal_metrics.csv", "registry": "v2bh_candidate_geometry_source_registry.csv",
    "v2bg_gates": "v2bg_recife_protocol_gate_status.csv",
}


def parse_args(argv=None):
    return argparse.ArgumentParser(description="v2bk orchestrator").parse_args(argv)


def dataset_path(name):
    return os.path.join(DATASET_DIR, name)


def doc_path(*parts):
    return os.path.join(DOCS_DIR, *parts)


def with_invariants(row):
    return {**row, **INVARIANTS}


def clean(value):
    return str(value or "").strip()


def is_true(value):
    return clean(value).lower() == "true"


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing empty output: {path}")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def refresh_inputs():
    """Regenerate the upstream v2bj (and v2bi) intake outputs from the live cache so this
    dossier reflects the real evidence. Disabled in tests via V2BK_REFRESH=0."""
    if not REFRESH:
        return "SKIPPED"
    try:
        import revp_v2bj_common as v2bj
    except ImportError:
        import scripts.protocolo_c.revp_v2bj_common as v2bj
    v2bj.run_orchestrator()
    return "REFRESHED"


def charter_facts():
    """Charter feature/license/date taken from the cached activation page metadata only
    (reference, not raw copy). Fail-closed to UNKNOWN when the cache is absent."""
    try:
        import revp_v2bj_common as v2bj
    except ImportError:
        import scripts.protocolo_c.revp_v2bj_common as v2bj
    try:
        return v2bj.extract_charter_facts()
    except Exception:
        return {"feature_type_candidate": "UNKNOWN", "license_terms": "UNKNOWN",
                "product_date": "UNKNOWN", "source_html_present": "false"}


def load_state():
    reconcile = {r.get("gate_id"): r for r in load_csv(dataset_path(INPUTS["reconcile"]))
                 if r.get("candidate_id") == CANDIDATE_ID}
    gates = {g: clean(reconcile.get(g, {}).get("reconciled_status")) or "UNKNOWN" for g in GATES}
    actions = {g: clean(reconcile.get(g, {}).get("human_action_required")) for g in GATES}
    queue = next((r for r in load_csv(dataset_path(INPUTS["queue"]))
                  if r.get("candidate_id") == CANDIDATE_ID), {})
    inmet = load_csv(dataset_path(INPUTS["inmet"]))
    a301 = next((r for r in inmet if r.get("station_code") == "A301"), {})
    readiness_rows = load_csv(dataset_path(INPUTS["charter_readiness"]))
    readiness = next((r for r in readiness_rows if r.get("product_id") == PRODUCT_ID),
                     readiness_rows[0] if readiness_rows else {})
    crs = next((r for r in load_csv(dataset_path(INPUTS["charter_crs"]))
                if r.get("product_id") == PRODUCT_ID), {})
    metrics = next((r for r in load_csv(dataset_path(INPUTS["temporal_metrics"]))
                    if r.get("event_patch_package_id") == EVENT_PATCH_ID), {})
    intake = load_csv(dataset_path(INPUTS["intake"]))

    def present(prefix):
        return any(is_true(r.get("file_present")) and clean(r.get("source")).startswith(prefix) for r in intake)

    facts = charter_facts()
    return {
        "gates": gates, "actions": actions,
        "reference_status": clean(queue.get("reference_status")) or "CANDIDATE_REFERENCE_PENDING_HUMAN_REVIEW",
        "charter_feature": facts["feature_type_candidate"], "charter_license": facts["license_terms"],
        "charter_product_date": facts["product_date"] if facts["product_date"] != "UNKNOWN" else "2022-06-02",
        "charter_readiness": clean(readiness.get("updated_candidate_status")) or "PENDING_VECTOR_CRS",
        "charter_crs_status": "PRESENT" if is_true(crs.get("crs_present")) else "ABSENT_OR_UNKNOWN",
        "charter_geometry_validity": clean(crs.get("geometry_validity_status")) or "NOT_AVAILABLE",
        "charter_map_present": present("International Charter"),
        "apac_present": present("APAC"), "ana_present": present("ANA HidroWeb"),
        "cemaden_present": present("Cemaden"),
        "temporal_status": clean(metrics.get("temporal_status")) or "NO_SERIES_AVAILABLE",
        "a301_coverage": clean(a301.get("coverage_status")) or "UNKNOWN",
        "event_window": f"{REQUEST_START} a {REQUEST_END}",
    }


# --------------------------------------------------------------------------- #
# Task 1 - human review dossier index.
# --------------------------------------------------------------------------- #

def run_build_recife_human_review_dossier(args=None):
    s = load_state()
    g = s["gates"]
    precip_local = "EMPTY_NOT_USABLE" if s["a301_coverage"] == "PRECIP_FULL_GAP" else s["a301_coverage"]
    row = with_invariants({
        "dossier_id": "DOSSIER_v2bk_001", "recife_package_id": PACKAGE_ID, "event_patch_package_id": EVENT_PATCH_ID,
        "event_date_or_window": s["event_window"], "candidate_status": s["reference_status"],
        "charter_product_status": "MAP_RASTER_PRESENT" if s["charter_map_present"] else "NO_FILE",
        "charter_feature_type": s["charter_feature"], "charter_crs_status": s["charter_crs_status"],
        "temporal_status": s["temporal_status"],
        "hydrological_context_status": "ANA_CAPIBARIBE_STAGE_PRESENT" if s["ana_present"] else "ABSENT",
        "precipitation_local_status": precip_local,
        "current_gate_c0": g["C0_PROVENANCE"], "current_gate_c1": g["C1_TEMPORALITY"],
        "current_gate_c2": g["C2_VALID_SERIES_OR_STATION"], "current_gate_c3": g["C3_SPATIAL_ANCHOR"],
        "current_gate_c4": g["C4_CANDIDATE_GEOMETRY"], "current_gate_c5": g["C5_HUMAN_REVIEW"],
        "current_gate_c6": g["C6_CANDIDATE_REFERENCE"], "current_gate_c7": g["C7_FINAL_GROUND_TRUTH"],
        "dossier_markdown_path": doc_path("dossier", f"{CANDIDATE_ID}_review_dossier.md").replace("\\", "/"),
        "ready_for_human_review": "true",
    })
    write_csv(dataset_path(OUTPUTS["dossier_index"]), [row])
    return [row]


# --------------------------------------------------------------------------- #
# Task 2 - Charter vector/CRS request pack.
# --------------------------------------------------------------------------- #

CHARTER_ARTIFACTS = [
    ("VECTOR_FILE", "Vector delineation (SHP/GeoJSON/GPKG/KML) of the Recife product",
     "SHP|GEOJSON|GPKG|KML", "C4_CANDIDATE_GEOMETRY", "P0"),
    ("CRS_METADATA", "Coordinate reference system (EPSG) of the delivered geometry",
     "TEXT|PRJ", "C4_CANDIDATE_GEOMETRY", "P0"),
    ("FEATURE_DEFINITION", "Legend/classes and feature semantics of the mapped polygons",
     "PDF|TEXT", "C4_CANDIDATE_GEOMETRY|C5_HUMAN_REVIEW", "P0"),
    ("LICENSE_TERMS", "Use and citation terms of the product",
     "TEXT", "C4_CANDIDATE_GEOMETRY", "P1"),
    ("REDISTRIBUTION_TERMS", "Redistribution permissions for academic use",
     "TEXT", "C4_CANDIDATE_GEOMETRY", "P1"),
    ("METHOD_DESCRIPTION", "Delineation method and source image acquisition date",
     "PDF|TEXT", "C4_CANDIDATE_GEOMETRY|C5_HUMAN_REVIEW", "P1"),
]


def run_build_charter_vector_crs_request_pack(args=None):
    s = load_state()
    cenad_tpl = doc_path("request_templates", "request_charter_758_vector_crs_cenad.md")
    charter_tpl = doc_path("request_templates", "request_charter_758_vector_crs_charter.md")
    rows = []
    for i, (artifact, reason, filetype, gate, priority) in enumerate(CHARTER_ARTIFACTS, 1):
        rows.append(with_invariants({
            "request_id": f"REQ_CHARTER_v2bk_{i:03d}", "target_institution": "CENAD (Charter requestor) / International Charter",
            "target_contact_or_channel": "CENAD project management; International Charter via authorized user / value-adder",
            "activation_id": ACTIVATION_ID, "product_title": PRODUCT_TITLE, "product_date": s["charter_product_date"],
            "requested_artifact": artifact, "reason_for_request": reason, "expected_file_type": filetype,
            "priority": priority, "blocks_gate": gate, "request_template_path": cenad_tpl.replace("\\", "/"),
        }))
    write_csv(dataset_path(OUTPUTS["charter_request"]), rows)
    _write_charter_templates(s, cenad_tpl, charter_tpl)
    return rows


def _charter_request_body(channel, s):
    return f"""# Solicitacao de produto vetorial e metadados - Charter {ACTIVATION_ID} (Recife)

Destinatario: {channel}

Eu/equipe do projeto REV-P (analise estrutural observacional, sem fins operacionais)
solicito acesso ao produto cartografico abaixo e seus metadados para revisao humana e
citacao academica. Esta solicitacao nao cria ground truth, rotulo nem geometria nova.

- Ativacao: International Charter Space and Major Disasters, Activation {ACTIVATION_ID}
- Produto: "{PRODUCT_TITLE}"
- Data do produto: {s['charter_product_date']}
- Municipio: Recife/PE (o produto de Olinda deve permanecer separado, nao se aplica a Recife)

Solicito formalmente:

1. Arquivo vetorial do produto (SHP, GeoJSON, GPKG ou KML), se existir.
2. Sistema de referencia de coordenadas (CRS/EPSG) do produto.
3. Legenda e classes das feicoes mapeadas.
4. Confirmacao explicita do tipo de feicao: landslide scars, affected areas, flood extent
   ou outro. Atualmente identifico o produto como landslide scars; preciso de confirmacao.
5. Termos de uso e de redistribuicao para uso academico.
6. Descricao do metodo de delineamento.
7. Confirmacao da data da imagem-fonte e do produto.
8. Referencia correta para citacao academica (incluindo {s['charter_license']}).

Observacoes:
- Recebi publicamente apenas um mapa raster em resolucao plena (PNG), sem CRS legivel por
  maquina e sem vetor; por isso solicito o vetor e o CRS oficiais.
- Nao convertere o raster em vetor nem inferirei geometria; aguardo o produto oficial.

Atenciosamente,
Equipe REV-P
"""


def _write_charter_templates(s, cenad_tpl, charter_tpl):
    write_text(cenad_tpl, _charter_request_body("CENAD (orgao solicitante da ativacao {})".format(ACTIVATION_ID), s))
    write_text(charter_tpl, _charter_request_body("International Charter (usuario autorizado / value-adder)", s))


# --------------------------------------------------------------------------- #
# Task 3 - Cemaden/APAC temporal request pack.
# --------------------------------------------------------------------------- #

def run_build_cemaden_apac_temporal_request_pack(args=None):
    cemaden_tpl = doc_path("request_templates", "request_cemaden_recife_rmr_precip_20220524_20220602.md")
    apac_tpl = doc_path("request_templates", "request_apac_recife_rmr_precip_20220524_20220602.md")
    definitions = [
        ("REQ_CEMADEN_v2bk_001", "Cemaden/MCTI", "Mapa Interativo (Download de Dados, link por e-mail) / contato-cemaden",
         "automatic pluviometers (rain gauges)", cemaden_tpl, "P0"),
        ("REQ_APAC_v2bk_001", "APAC-PE", "gmmc@apac.pe.gov.br / portal de monitoramento RMR",
         "rain gauge / station network", apac_tpl, "P1"),
    ]
    rows = []
    for request_id, institution, channel, station_type, template, priority in definitions:
        rows.append(with_invariants({
            "request_id": request_id, "target_institution": institution, "target_contact_or_channel": channel,
            "requested_period_start": REQUEST_START, "requested_period_end": REQUEST_END,
            "requested_area": "Recife / Regiao Metropolitana do Recife (RMR)",
            "requested_variable": "precipitation (hourly or daily)", "requested_station_type": station_type,
            "reason_for_request": "Local rainfall series for the event window to advance C2 (valid local series/station).",
            "expected_file_type": "CSV|TXT|XLSX", "priority": priority,
            "blocks_gate": "C1_TEMPORALITY|C2_VALID_SERIES_OR_STATION",
            "request_template_path": template.replace("\\", "/"),
        }))
    write_csv(dataset_path(OUTPUTS["temporal_request"]), rows)
    _write_temporal_templates(cemaden_tpl, apac_tpl)
    return rows


def _temporal_request_body(institution, channel):
    return f"""# Solicitacao de serie de precipitacao local - Recife/RMR (maio/2022)

Destinatario: {institution} ({channel})

Eu/equipe do projeto REV-P solicito a serie instrumental de precipitacao local para o evento
extremo de Recife/RMR. Esta solicitacao nao cria ground truth, rotulo nem treino; o dado sera
usado como evidencia temporal para revisao humana.

Solicito formalmente:

1. Precipitacao horaria ou diaria.
2. Estacoes/pluviometros em Recife e na Regiao Metropolitana do Recife (RMR).
3. Coordenadas (lat/lon) ou codigo de cada estacao.
4. Timezone de referencia (UTC ou local).
5. Unidade de medida (mm).
6. Flags de qualidade/consistencia.
7. Periodo: {REQUEST_START} a {REQUEST_END}.
8. Confirmacao de cobertura local efetiva no periodo.
9. Instrucao de citacao/fonte do dado.

Observacao: a estacao automatica INMET A301 (Recife) esta com precipitacao vazia no periodo
e no ano; por isso solicito a serie local de {institution}. Nao usarei proxy regional como
substituto da estacao local.

Atenciosamente,
Equipe REV-P
"""


def _write_temporal_templates(cemaden_tpl, apac_tpl):
    write_text(cemaden_tpl, _temporal_request_body("Cemaden", "Mapa Interativo / contato institucional"))
    write_text(apac_tpl, _temporal_request_body("APAC", "gmmc@apac.pe.gov.br"))


# --------------------------------------------------------------------------- #
# Task 4 - C5/C6 adjudication checklist.
# --------------------------------------------------------------------------- #

ACCEPTABLE = "ACCEPT_FOR_CANDIDATE_REFERENCE|KEEP_PENDING|REJECT_FOR_NOW|REQUEST_MORE_EVIDENCE|MARK_HAZARD_AMBIGUOUS"


def run_build_c5_c6_adjudication_checklist(args=None):
    s = load_state()
    questions = [
        ("C5_HUMAN_REVIEW", "O mapa Charter realmente cobre Recife?",
         "Produto Charter 758 com extensao espacial em Recife",
         "Mapa raster presente; produto 'Recife/PE'" if s["charter_map_present"] else "Sem arquivo",
         "KEEP_PENDING", "Confirmar visualmente; nao inferir cobertura."),
        ("C5_HUMAN_REVIEW", "A feicao e deslizamento, inundacao, dano ou multihazard?",
         "Legenda/classes do produto",
         f"Identificado como {s['charter_feature']} (landslide scars); falta confirmacao oficial",
         "MARK_HAZARD_AMBIGUOUS", "Landslide scar nao e flood extent; aguardar confirmacao CENAD."),
        ("C5_HUMAN_REVIEW", "O produto e raster ou vetor?",
         "Tipo de arquivo do produto", "Raster (PNG) presente; vetor nao confirmado",
         "REQUEST_MORE_EVIDENCE", "Raster nao e geometria vetorial."),
        ("C5_HUMAN_REVIEW", "Existe CRS?",
         "Metadado de CRS/EPSG", f"CRS {s['charter_crs_status']}",
         "REQUEST_MORE_EVIDENCE", "Sem CRS legivel, geometria nao promove."),
        ("C5_HUMAN_REVIEW", "A geometria pode ser revisada manualmente?",
         "Vetor + CRS revisaveis", f"Validade geometrica: {s['charter_geometry_validity']}",
         "KEEP_PENDING", "Apenas mapa revisavel; geometria depende de vetor/CRS."),
        ("C5_HUMAN_REVIEW", "A evidencia temporal local existe?",
         "Serie de chuva local Recife/RMR na janela",
         f"Status temporal: {s['temporal_status']}; A301 {s['a301_coverage']}",
         "REQUEST_MORE_EVIDENCE", "Cemaden/APAC local ainda necessarios."),
        ("C5_HUMAN_REVIEW", "ANA cota e apenas contexto hidrologico?",
         "Papel da serie ANA",
         "ANA Capibaribe (Sao Lourenco da Mata/RMR) presente" if s["ana_present"] else "ANA ausente",
         "KEEP_PENDING", "Cota nao e precipitacao nem flood extent."),
        ("C5_HUMAN_REVIEW", "APAC PDF mensal e suficiente para C1, mas nao C2 completo?",
         "Natureza do PDF APAC",
         "APAC mensal presente" if s["apac_present"] else "APAC ausente",
         "KEEP_PENDING", "Agregado mensal: contexto de C1, nao serie de estacao para C2."),
        ("C6_CANDIDATE_REFERENCE", "Cemaden/APAC local ainda e necessario?",
         "Serie local de chuva", "Sim; pendente de download manual/solicitacao",
         "REQUEST_MORE_EVIDENCE", "C2 so completa com serie local."),
        ("C6_CANDIDATE_REFERENCE", "Ha base para candidate reference?",
         "C1 suportado + C3 pass + C4 revisavel",
         f"C1={s['gates']['C1_TEMPORALITY']}; C3={s['gates']['C3_SPATIAL_ANCHOR']}; C4={s['gates']['C4_CANDIDATE_GEOMETRY']}",
         "KEEP_PENDING", "Manter CANDIDATE_REFERENCE_PENDING_HUMAN_REVIEW."),
        ("C6_CANDIDATE_REFERENCE", "Ha base para ground truth final?",
         "Todos os gates resolvidos + revisao humana + vetor/CRS + serie local",
         "Nao: C2 parcial, C4 sem vetor/CRS, C7 bloqueado",
         "KEEP_PENDING", "Nao. Ground truth final permanece proibido (C7 BLOCKED)."),
    ]
    rows = []
    for i, (stage, q, required, current, recommend, cannot) in enumerate(questions, 1):
        rows.append(with_invariants({
            "checklist_id": f"CHK_v2bk_{i:03d}", "recife_package_id": PACKAGE_ID, "decision_stage": stage,
            "review_question": q, "required_evidence": required, "current_evidence": current,
            "acceptable_decision": ACCEPTABLE, "current_recommendation": recommend, "cannot_infer": cannot,
        }))
    write_csv(dataset_path(OUTPUTS["checklist"]), rows)
    lines = [f"# C5/C6 adjudication checklist - {CANDIDATE_ID}", "",
             "| stage | question | current evidence | recommendation | cannot infer |",
             "| --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['decision_stage']} | {r['review_question']} | {r['current_evidence']} | "
                     f"{r['current_recommendation']} | {r['cannot_infer']} |")
    write_text(doc_path("adjudication_checklists", f"{CANDIDATE_ID}_c5_c6_checklist.md"), "\n".join(lines) + "\n")
    return rows


# --------------------------------------------------------------------------- #
# Task 5 - decision matrix.
# --------------------------------------------------------------------------- #

def run_build_recife_decision_matrix(args=None):
    s = load_state()
    g = s["gates"]
    axes = [
        ("TEMPORALITY", g["C1_TEMPORALITY"], "ANA Capibaribe stage (dated) + APAC May accumulation",
         "Parsed local rainfall series absent", "HUMAN_REVIEW_TEMPORALITY"),
        ("LOCAL_PRECIPITATION", "PENDING_LOCAL_SERIES",
         "None usable (A301 precip empty; proxies regional)", "Cemaden/APAC local series missing",
         "REQUEST_CEMADEN_APAC_LOCAL_SERIES"),
        ("HYDROLOGICAL_CONTEXT", "PRESENT_CONTEXT_ONLY" if s["ana_present"] else "ABSENT",
         "ANA river stage at Sao Lourenco da Mata (39187800, RMR)",
         "River stage is not precipitation and not flood extent", "KEEP_AS_CONTEXT_ONLY"),
        ("CHARTER_SPATIAL_PRODUCT", g["C3_SPATIAL_ANCHOR"], "Charter 758 Recife landslide-scars map",
         "None for spatial anchor", "REVIEW_PRODUCT"),
        ("GEOMETRY_ACCESS", g["C4_CANDIDATE_GEOMETRY"], "Full-resolution raster map present",
         "No machine-readable CRS, no vector", "REQUEST_CHARTER_VECTOR_CRS_FROM_CENAD"),
        ("HAZARD_TYPING", "LANDSLIDE_SCARS_PENDING_CONFIRMATION",
         f"Product feature {s['charter_feature']}", "Official legend/class confirmation pending",
         "CONFIRM_FEATURE_NOT_FLOOD_EXTENT"),
        ("HUMAN_REVIEW", g["C5_HUMAN_REVIEW"], "Dossier assembled", "Awaiting human decision",
         "EXECUTE_HUMAN_REVIEW"),
        ("FINAL_TRUTH", g["C7_FINAL_GROUND_TRUTH"], "None", "Final ground truth prohibited",
         "NONE_FINAL_GROUND_TRUTH_PROHIBITED"),
    ]
    rows = []
    for axis, status, evidence, blocker, action in axes:
        rows.append(with_invariants({
            "recife_package_id": PACKAGE_ID, "decision_axis": axis, "current_status": status,
            "evidence_supporting": evidence, "blocker": blocker, "next_action": action,
            "promotion_allowed": "false",
        }))
    write_csv(dataset_path(OUTPUTS["decision"]), rows)
    lines = [f"# Recife candidate decision matrix - {CANDIDATE_ID}", "",
             "| axis | status | evidence | blocker | next action | promotion |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['decision_axis']} | {r['current_status']} | {r['evidence_supporting']} | "
                     f"{r['blocker']} | {r['next_action']} | {r['promotion_allowed']} |")
    write_text(doc_path("decision_matrix", f"{CANDIDATE_ID}_decision_matrix.md"), "\n".join(lines) + "\n")
    return rows


# --------------------------------------------------------------------------- #
# Task 6 - review-ready markdown + README.
# --------------------------------------------------------------------------- #

def run_generate_review_ready_markdown(args=None):
    s = load_state()
    g = s["gates"]
    checklist = load_csv(dataset_path(OUTPUTS["checklist"]))
    charter_rows = load_csv(dataset_path(OUTPUTS["charter_request"]))
    temporal_rows = load_csv(dataset_path(OUTPUTS["temporal_request"]))
    lines = [
        f"# Recife candidate reference - human review dossier ({CANDIDATE_ID})", "",
        "## 1. Identificacao do pacote",
        f"- Candidate: `{CANDIDATE_ID}` | Package: `{PACKAGE_ID}` | Event-patch: `{EVENT_PATCH_ID}`",
        f"- Produto Charter: `{PRODUCT_ID}` - \"{PRODUCT_TITLE}\" ({s['charter_product_date']})",
        f"- Status de referencia: `{s['reference_status']}`", "",
        "## 2. Resumo do evento Recife maio/2022",
        f"- Janela do evento: {s['event_window']} (chuvas extremas, deslizamentos e inundacoes na RMR).", "",
        "## 3. Evidencia Charter 758",
        f"- Mapa raster em resolucao plena presente: `{s['charter_map_present']}`.",
        f"- Feicao identificada: {s['charter_feature']} (landslide scars).",
        f"- CRS: {s['charter_crs_status']} | Validade geometrica: {s['charter_geometry_validity']}.",
        f"- Licenca/fonte: {s['charter_license']}.", "",
        "## 4. Evidencia APAC/ANA/INMET auditada",
        f"- APAC acumulado mensal maio/2022 presente: `{s['apac_present']}` (contexto).",
        f"- ANA HidroWeb cota Capibaribe (Sao Lourenco da Mata/RMR) presente: `{s['ana_present']}` (contexto hidrologico).",
        f"- INMET A301 Recife (chuva local): {s['a301_coverage']} - vazia, nao utilizavel.",
        f"- Cemaden local presente: `{s['cemaden_present']}` (pendente).", "",
        "## 5. O que cada fonte prova",
        "- Charter 758: existencia de produto cartografico oficial de deslizamento em Recife (ancora espacial).",
        "- ANA cota: que houve resposta hidrologica datada na bacia do Capibaribe na janela.",
        "- APAC mensal: magnitude do mes do evento.", "",
        "## 6. O que cada fonte NAO prova",
        "- Charter raster NAO prova geometria vetorial nem CRS; NAO e flood extent.",
        "- ANA cota NAO prova precipitacao local nem mancha de inundacao.",
        "- APAC mensal NAO e serie horaria/de estacao (nao fecha C2).",
        "- INMET A301 vazia NAO e substituida por proxy regional (A320 Joao Pessoa e outra cidade/estado).", "",
        "## 7. Gates C0-C7",
    ]
    for gate in GATES:
        lines.append(f"- {gate}: `{g[gate]}`")
    lines += ["", "## 8. C5 checklist"]
    for r in checklist:
        if r["decision_stage"] == "C5_HUMAN_REVIEW":
            lines.append(f"- {r['review_question']} -> recomendacao: `{r['current_recommendation']}` ({r['cannot_infer']})")
    lines += ["", "## 9. C6 candidate reference pending",
              f"- `{g['C6_CANDIDATE_REFERENCE']}` - decisao de referencia deferida a revisao humana, sem promocao.", "",
              "## 10. Requests pendentes",
              f"- Charter vetor/CRS ({len(charter_rows)} itens) -> templates em request_templates/.",
              f"- Cemaden/APAC chuva local ({len(temporal_rows)} itens) -> templates em request_templates/.", "",
              "## 11. Decisao recomendada",
              "- Manter como `CANDIDATE_REFERENCE_PENDING_HUMAN_REVIEW`.",
              "- Solicitar vetor/CRS do Charter (CENAD/Charter).",
              "- Solicitar serie local Cemaden/APAC (Recife/RMR).",
              "- NAO promover a ground truth final.", "",
              "## 12. Guardrails",
              "- can_create_ground_truth=false; can_create_label=false; can_create_negative=false; can_train_model=false.",
              "- request_pack_is_not_evidence=true; human_review_dossier_is_not_ground_truth=true.",
              "- charter_raster_is_not_vector_geometry=true; ana_stage_is_not_precipitation=true.",
              "- apac_pdf_is_not_station_series=true; inmet_proxy_is_not_local_station=true.",
              "- candidate_reference_is_not_final_truth=true; C7 BLOCKED.", ""]
    write_text(doc_path("dossier", f"{CANDIDATE_ID}_review_dossier.md"), "\n".join(lines))
    return [{"dossier": f"{CANDIDATE_ID}_review_dossier.md"}]


def run_generate_readme(args=None):
    s = load_state()
    write_text(doc_path("README.md"), f"""# v2bk Recife Human Review Dossier and Data Request Pack

Esta etapa existe porque Recife (`{CANDIDATE_ID}`) passou a ter um candidate reference
pendente de revisao humana apos o intake real da v2bi e a reconciliacao da v2bj. O gargalo
deixou de ser pesquisa generica: agora e (1) revisao humana e (2) pedido formal dos dados
que faltam.

A v2bk consome os resultados reais da v2bi/v2bj (mapa Charter 758, APAC mensal, ANA cota
Capibaribe, auditoria INMET) e gera, de forma estritamente aditiva: um dossie de revisao
humana, pacotes de solicitacao de vetor/CRS ao CENAD/Charter, pacotes de solicitacao de
chuva local ao Cemaden/APAC, um checklist C5/C6 e uma matriz de decisao.

Status de referencia: `{s['reference_status']}`. C7 continua BLOCKED porque o ground truth
final exige todos os gates resolvidos, revisao humana, vetor/CRS oficial e serie local de
chuva - nada disso esta completo. Um pacote de solicitacao nao e evidencia; um dossie nao e
ground truth; um raster Charter nao e vetor; cota ANA nao e precipitacao; PDF mensal APAC nao
e serie de estacao; proxy INMET nao e a estacao local de Recife.

Ground truth final, labels, negativos e treino = 0.
""")
    return [{"readme": "README.md"}]


# --------------------------------------------------------------------------- #
# Guardrail regression.
# --------------------------------------------------------------------------- #

def run_guardrail_regression(args=None):
    forbidden = {"can_create_ground_truth", "can_create_patch_truth", "can_create_label",
                 "can_create_negative", "can_train_model", "raw_data_versioned"}
    rows = []
    datasets = (OUTPUTS["dossier_index"], OUTPUTS["charter_request"], OUTPUTS["temporal_request"],
                OUTPUTS["checklist"], OUTPUTS["decision"])
    for number, name in enumerate(datasets, 1):
        data = load_csv(dataset_path(name))
        violations = sum(clean(r.get(field)).lower() == "true" for r in data for field in forbidden)
        rows.append({"regression_id": f"GR_v2bk_{number:03d}", "check": f"forbidden_flags::{name}",
                     "detail": "no forbidden invariant is true", "violation_count": str(violations),
                     "status": "PASS" if not violations else "FAIL"})
    decision = load_csv(dataset_path(OUTPUTS["decision"]))
    final = [r for r in decision if r["decision_axis"] == "FINAL_TRUTH"]
    c7_ok = bool(final) and all(clean(r["current_status"]).startswith("BLOCKED") or r["current_status"] == "BLOCKED"
                                for r in final)
    rows.append({"regression_id": "GR_v2bk_006", "check": "final_truth_blocked",
                 "detail": "decision FINAL_TRUTH axis stays blocked",
                 "violation_count": "0" if c7_ok else "1", "status": "PASS" if c7_ok else "FAIL"})
    promo = sum(clean(r.get("promotion_allowed")).lower() == "true" for r in decision)
    rows.append({"regression_id": "GR_v2bk_007", "check": "no_promotion", "detail": "no decision axis promotes",
                 "violation_count": str(promo), "status": "PASS" if not promo else "FAIL"})
    index = load_csv(dataset_path(OUTPUTS["dossier_index"]))
    ref_ok = bool(index) and all("FINAL" not in clean(r.get("candidate_status")).upper() and
                                 "GROUND_TRUTH" not in clean(r.get("candidate_status")).upper() for r in index)
    rows.append({"regression_id": "GR_v2bk_008", "check": "reference_not_final",
                 "detail": "candidate reference is not final truth",
                 "violation_count": "0" if ref_ok else "1", "status": "PASS" if ref_ok else "FAIL"})
    if any(r["status"] != "PASS" for r in rows):
        raise ValueError("v2bk guardrail regression failed")
    write_csv(dataset_path(OUTPUTS["guardrail"]), rows)
    return rows


def _steps():
    return [
        ("build_recife_human_review_dossier", run_build_recife_human_review_dossier, dataset_path(OUTPUTS["dossier_index"])),
        ("build_charter_vector_crs_request_pack", run_build_charter_vector_crs_request_pack, dataset_path(OUTPUTS["charter_request"])),
        ("build_cemaden_apac_temporal_request_pack", run_build_cemaden_apac_temporal_request_pack, dataset_path(OUTPUTS["temporal_request"])),
        ("build_c5_c6_adjudication_checklist", run_build_c5_c6_adjudication_checklist, dataset_path(OUTPUTS["checklist"])),
        ("build_recife_decision_matrix", run_build_recife_decision_matrix, dataset_path(OUTPUTS["decision"])),
        ("generate_review_ready_markdown", run_generate_review_ready_markdown,
         doc_path("dossier", f"{CANDIDATE_ID}_review_dossier.md")),
        ("generate_readme", run_generate_readme, doc_path("README.md")),
        ("run_guardrail_regression", run_guardrail_regression, dataset_path(OUTPUTS["guardrail"])),
    ]


def ensure_structure():
    for folder in (DOCS_DIR, doc_path("dossier"), doc_path("request_templates"),
                   doc_path("adjudication_checklists"), doc_path("decision_matrix"), doc_path("evidence_cache")):
        os.makedirs(folder, exist_ok=True)
    write_text(doc_path("evidence_cache", ".gitignore"), "*\n!.gitignore\n")


def run_orchestrator(args=None):
    ensure_structure()
    refresh_status = refresh_inputs()
    reconcile_path = dataset_path(INPUTS["reconcile"])
    manifest = [{"step_order": "0", "step_name": "refresh_v2bj_v2bi_inputs", "status": refresh_status,
                 "output": reconcile_path.replace("\\", "/"),
                 "output_hash": sha256(reconcile_path)[:16] if os.path.exists(reconcile_path) else "",
                 "notes": "Regenerates upstream intake from the live cache; no promotion."}]
    for number, (name, function, path) in enumerate(_steps(), 1):
        function(args)
        manifest.append({"step_order": str(number), "step_name": name, "status": "OK",
                         "output": path.replace("\\", "/"), "output_hash": sha256(path)[:16],
                         "notes": "Strictly additive; request pack is not evidence; no ground truth."})
    write_csv(dataset_path(OUTPUTS["manifest"]), manifest)
    return manifest


if __name__ == "__main__":
    run_orchestrator(parse_args())
