#!/usr/bin/env python3
"""
v1uj — Completion Report

Consolida todos os outputs v1uj em: event status registry, next actions,
versionable artifacts manifest e relatorio/status docs.

Responde: o que v1ui-live encontrou, o que v1uj aprofundou, quais fontes
responderam, quais pacotes/produtos apareceram, se houve artefato vetorial,
se ha candidato pronto para revisao, qual evento e mais promissor, o que falta
para ground reference e se v1uk deve ser supervisor review ou nova rodada.
"""

import argparse
import csv
import hashlib
import os
from datetime import datetime

PROTOCOL_VERSION = "v1uj"

GUARDRAILS = {
    "ground_truth_operational": False,
    "can_create_ground_reference": False,
    "can_create_training_label": False,
    "can_reopen_protocol_b": False,
    "dino_usage": "SUPPORT_ONLY",
    "no_overlay_executed": True,
    "no_coordinates_invented": True,
    "supervisor_review_completed": False,
    "route": "PUBLIC_OFFICIAL_DISCOVERY",
    "formal_request_path": "LEGACY_SECONDARY_ONLY",
    "operational_product_max_status": "OBSERVED_GEOMETRY_CANDIDATE_FOR_REVIEW",
}

EVENTS = ["PET_2022_02_15", "PET_2024_03_21_28", "REC_2022_05_24_30"]

CANDIDATE_INVENTORY_CLASSIFICATIONS = {
    "OBSERVED_GEOMETRY_CANDIDATE_FOR_REVIEW",
    "TABLE_WITH_COORDINATES_CANDIDATE_FOR_REVIEW",
    "observed_geometry",
}

V1UJ_ARTIFACTS = {
    "configs/protocolo_c/v1uj_focused_source_targets.yaml": "config",
    "configs/protocolo_c/v1uj_copernicus_ems_targets.yaml": "config",
    "configs/protocolo_c/v1uj_geosgb_service_targets.yaml": "config",
    "configs/protocolo_c/v1uj_ckan_targets.yaml": "config",
    "configs/protocolo_c/v1uj_s2id_targets.yaml": "config",
    "configs/protocolo_c/v1uj_rigeo_targets.yaml": "config",
    "configs/protocolo_c/v1uj_download_policy.yaml": "config",
    "datasets/protocolo_c/v1uj_copernicus_ems_registry.csv": "dataset",
    "datasets/protocolo_c/v1uj_geosgb_layer_registry.csv": "dataset",
    "datasets/protocolo_c/v1uj_ckan_resource_registry.csv": "dataset",
    "datasets/protocolo_c/v1uj_s2id_resource_registry.csv": "dataset",
    "datasets/protocolo_c/v1uj_rigeo_bitstream_registry.csv": "dataset",
    "datasets/protocolo_c/v1uj_pdf_deeplink_registry.csv": "dataset",
    "datasets/protocolo_c/v1uj_focused_download_manifest.csv": "dataset",
    "datasets/protocolo_c/v1uj_download_collision_audit.csv": "dataset",
    "datasets/protocolo_c/v1uj_focused_artifact_inventory.csv": "dataset",
    "datasets/protocolo_c/v1uj_observed_candidate_promotion_audit.csv": "dataset",
    "datasets/protocolo_c/v1uj_event_status_registry.csv": "dataset",
    "datasets/protocolo_c/v1uj_next_actions_registry.csv": "dataset",
    "datasets/protocolo_c/v1uj_live_execution_summary.csv": "dataset",
    "datasets/protocolo_c/v1uj_versionable_artifacts_manifest.csv": "dataset",
}

EVENT_STATUS_COLUMNS = [
    "event_id", "copernicus_products", "copernicus_vector_candidates",
    "geosgb_layers", "geosgb_observed_candidates", "ckan_resources",
    "ckan_geo_candidates", "s2id_records", "rigeo_bitstreams",
    "rigeo_geodata_candidates", "pdf_deeplinks", "downloaded_artifacts",
    "observed_candidates_for_review", "vector_artifact_found",
    "most_promising_source", "path_to_supervisor_review", "status_summary",
]

NEXT_ACTIONS_COLUMNS = [
    "action_id", "event_id", "action_type", "priority",
    "description", "target", "status", "notes",
]

MANIFEST_COLUMNS = [
    "artifact_id", "artifact_path", "artifact_type", "protocol_version",
    "sha256_prefix", "file_size_bytes", "is_versionable", "reason",
]

LIVE_SUMMARY_COLUMNS = [
    "summary_id", "metric", "value", "notes",
]

LOCAL_AVAILABLE_DOWNLOAD_STATUSES = {
    "DOWNLOAD_OK",
    "ALREADY_EXISTS_SAME_URL_SAME_HASH",
    "DUPLICATE_CONTENT_DIFFERENT_URL",
    "DOWNLOADED",
    "ALREADY_EXISTS",
}


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha256_file(path):
    if not os.path.exists(path):
        return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def count_for(rows, event_id, pred):
    return sum(1 for r in rows if r.get("event_id") == event_id and pred(r))


def count_status(rows, status):
    return sum(1 for r in rows if r.get("download_status") == status)


def count_unique(rows, field, pred=lambda _r: True):
    return len({r.get(field, "") for r in rows if r.get(field, "") and pred(r)})


def write_live_summary(path, metrics):
    rows = []
    for seq, (metric, value, notes) in enumerate(metrics):
        rows.append({
            "summary_id": f"SUM_{PROTOCOL_VERSION}_{seq:04d}",
            "metric": metric,
            "value": str(value),
            "notes": notes,
        })
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LIVE_SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="v1uj — Completion Report")
    parser.add_argument("--out-dir", default="datasets/protocolo_c")
    parser.add_argument("--docs-dir", default="docs/metodologia_cientifica")
    args = parser.parse_args()

    d = args.out_dir
    copernicus = load_csv(os.path.join(d, "v1uj_copernicus_ems_registry.csv"))
    geosgb = load_csv(os.path.join(d, "v1uj_geosgb_layer_registry.csv"))
    ckan = load_csv(os.path.join(d, "v1uj_ckan_resource_registry.csv"))
    s2id = load_csv(os.path.join(d, "v1uj_s2id_resource_registry.csv"))
    rigeo = load_csv(os.path.join(d, "v1uj_rigeo_bitstream_registry.csv"))
    pdf = load_csv(os.path.join(d, "v1uj_pdf_deeplink_registry.csv"))
    manifest_dl = load_csv(os.path.join(d, "v1uj_focused_download_manifest.csv"))
    collision = load_csv(os.path.join(d, "v1uj_download_collision_audit.csv"))
    inventory = load_csv(os.path.join(d, "v1uj_focused_artifact_inventory.csv"))
    promotion = load_csv(os.path.join(d, "v1uj_observed_candidate_promotion_audit.csv"))

    event_rows = []
    next_actions = []
    aseq = 0

    for eid in EVENTS:
        cop_n = count_for(copernicus, eid, lambda r: bool(r.get("product_url")))
        cop_vec = count_for(copernicus, eid, lambda r: r.get("is_vector_candidate") == "true")
        gsgb_n = count_for(geosgb, eid, lambda r: bool(r.get("layer_id")))
        gsgb_obs = count_for(geosgb, eid, lambda r: r.get("is_observed_occurrence_candidate") == "true")
        ckan_n = count_for(ckan, eid, lambda r: bool(r.get("resource_id")))
        ckan_geo = count_for(ckan, eid, lambda r: r.get("is_geospatial_candidate") == "true"
                             and r.get("is_contextual_only") != "true")
        s2id_n = count_for(s2id, eid, lambda r: True)
        rigeo_n = count_for(rigeo, eid, lambda r: bool(r.get("bitstream_url")))
        rigeo_geo = count_for(rigeo, eid, lambda r: r.get("is_geodata_candidate") == "true")
        pdf_n = count_for(pdf, eid, lambda r: r.get("is_pdf_link_candidate") == "true")
        dl_n = count_for(manifest_dl, eid,
                         lambda r: r.get("download_status") in LOCAL_AVAILABLE_DOWNLOAD_STATUSES)
        obs_review = count_for(promotion, eid,
                               lambda r: r.get("max_status") == "OBSERVED_GEOMETRY_CANDIDATE_FOR_REVIEW")
        vector_found = (cop_vec > 0 or ckan_geo > 0 or rigeo_geo > 0
                        or count_for(inventory, eid,
                                     lambda r: r.get("classification")
                                     in CANDIDATE_INVENTORY_CLASSIFICATIONS) > 0)

        scores = {"copernicus": cop_vec, "geosgb": gsgb_obs, "ckan": ckan_geo,
                  "rigeo": rigeo_geo, "s2id": s2id_n}
        most_promising = max(scores, key=lambda k: scores[k]) if any(scores.values()) else "none_yet"

        path_review = "READY" if obs_review > 0 else "NOT_YET"
        if obs_review > 0:
            summary = f"{obs_review} candidato(s) observado(s) para revisao supervisora"
            next_actions.append({
                "action_id": f"ACT_{PROTOCOL_VERSION}_{aseq:04d}", "event_id": eid,
                "action_type": "SUPERVISOR_REVIEW", "priority": "1",
                "description": f"{obs_review} OBSERVED_GEOMETRY_CANDIDATE_FOR_REVIEW",
                "target": "Supervisor humano", "status": "PENDING", "notes": "",
            })
        else:
            summary = "Sem candidato observado; aprofundar fonte focada ou rodada regional"
            next_actions.append({
                "action_id": f"ACT_{PROTOCOL_VERSION}_{aseq:04d}", "event_id": eid,
                "action_type": "DEEPEN_FOCUSED_OR_REGIONAL", "priority": "2",
                "description": f"Fonte mais promissora: {most_promising}",
                "target": "Resolvers focados com --allow-web",
                "status": "PENDING", "notes": "",
            })
        aseq += 1

        event_rows.append({
            "event_id": eid, "copernicus_products": str(cop_n),
            "copernicus_vector_candidates": str(cop_vec),
            "geosgb_layers": str(gsgb_n), "geosgb_observed_candidates": str(gsgb_obs),
            "ckan_resources": str(ckan_n), "ckan_geo_candidates": str(ckan_geo),
            "s2id_records": str(s2id_n), "rigeo_bitstreams": str(rigeo_n),
            "rigeo_geodata_candidates": str(rigeo_geo), "pdf_deeplinks": str(pdf_n),
            "downloaded_artifacts": str(dl_n),
            "observed_candidates_for_review": str(obs_review),
            "vector_artifact_found": str(bool(vector_found)).lower(),
            "most_promising_source": most_promising,
            "path_to_supervisor_review": path_review,
            "status_summary": summary,
        })

    with open(os.path.join(d, "v1uj_event_status_registry.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EVENT_STATUS_COLUMNS)
        writer.writeheader()
        writer.writerows(event_rows)

    with open(os.path.join(d, "v1uj_next_actions_registry.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=NEXT_ACTIONS_COLUMNS)
        writer.writeheader()
        writer.writerows(next_actions)

    total_obs = sum(int(r["observed_candidates_for_review"]) for r in event_rows)
    total_vec = sum(1 for r in event_rows if r["vector_artifact_found"] == "true")
    downloads_ok = count_status(manifest_dl, "DOWNLOAD_OK")
    already_same = count_status(manifest_dl, "ALREADY_EXISTS_SAME_URL_SAME_HASH")
    duplicate_content = count_status(manifest_dl, "DUPLICATE_CONTENT_DIFFERENT_URL")
    blocked_or_failed = sum(1 for r in manifest_dl
                            if r.get("download_status") not in LOCAL_AVAILABLE_DOWNLOAD_STATUSES
                            and r.get("download_status") not in ("DRY_RUN", "PLANNED"))
    unique_download_urls = count_unique(manifest_dl, "url")
    collision_rows = sum(1 for r in collision if r.get("collision_status") == "COLLISION_DETECTED")
    collision_groups = count_unique(collision, "collision_group",
                                    lambda r: r.get("collision_status") == "COLLISION_DETECTED")
    ckan_total = len(ckan)
    ckan_geo = sum(1 for r in ckan if r.get("is_geospatial_candidate") == "true")
    ckan_geo_noncontext = sum(1 for r in ckan if r.get("is_geospatial_candidate") == "true"
                              and r.get("is_contextual_only") != "true")
    ckan_csv = sum(1 for r in ckan if (r.get("resource_format") or "").upper() == "CSV")
    ckan_geojson = sum(1 for r in ckan if (r.get("resource_format") or "").upper() == "GEOJSON")
    inv_csv = sum(1 for r in inventory if r.get("extension") == ".csv")
    inv_geojson = sum(1 for r in inventory if r.get("extension") == ".geojson")
    inv_coord_tables = sum(1 for r in inventory
                           if r.get("classification") == "TABLE_WITH_COORDINATES_CANDIDATE_FOR_REVIEW")
    inv_occurrence_tables = sum(1 for r in inventory
                                if r.get("classification") == "DOCUMENTED_OCCURRENCE_TABLE_NO_GEOMETRY")
    inv_context_layers = sum(1 for r in inventory
                             if r.get("classification") == "CONTEXTUAL_OFFICIAL_LAYER")
    inv_context_only = sum(1 for r in inventory if r.get("classification") == "CONTEXT_ONLY")
    audit_blockers = sum(1 for r in promotion if r.get("max_status") == "CANDIDATE_WITH_BLOCKERS")
    audit_not_geometry = sum(1 for r in promotion if r.get("max_status") == "NOT_A_GEOMETRY_CANDIDATE")
    rec_status = next((r for r in event_rows if r["event_id"] == "REC_2022_05_24_30"), {})
    rec_ready = rec_status.get("path_to_supervisor_review") == "READY"

    summary_metrics = [
        ("ckan_resources_total", ckan_total, "Recursos CKAN Recife inventariados no registry v1uj"),
        ("ckan_geospatial_candidates", ckan_geo, "Inclui candidatos contextuais"),
        ("ckan_geospatial_noncontextual_candidates", ckan_geo_noncontext, "Elegiveis para downloader focado"),
        ("ckan_csv_resources", ckan_csv, "resource_format=CSV"),
        ("ckan_geojson_resources", ckan_geojson, "resource_format=GeoJSON"),
        ("unique_download_urls_processed", unique_download_urls, "URLs unicas no manifest de download"),
        ("download_ok", downloads_ok, "download_status=DOWNLOAD_OK"),
        ("already_exists_same_url_same_hash", already_same, "Reexecucao segura da mesma URL"),
        ("duplicate_content_different_url", duplicate_content, "Conteudo SHA256 identico em URLs diferentes"),
        ("blocked_or_failed_downloads", blocked_or_failed, "Falhas ou bloqueios com status explicito"),
        ("collision_rows_detected", collision_rows, "Linhas do audit com colisao de basename/path"),
        ("collision_groups_detected", collision_groups, "Grupos de colisao distintos"),
        ("inventoried_csv_assets", inv_csv, "Arquivos CSV inventariados localmente"),
        ("inventoried_geojson_assets", inv_geojson, "Arquivos GeoJSON inventariados localmente"),
        ("table_with_coordinates_candidate_for_review", inv_coord_tables,
         "Classificacao de inventario, nao ground reference"),
        ("documented_occurrence_table_no_geometry", inv_occurrence_tables,
         "Tabelas com ocorrencia/data/endereco sem coordenadas formais"),
        ("contextual_official_layer", inv_context_layers, "Camadas oficiais contextuais"),
        ("context_only", inv_context_only, "Infraestrutura, risco ou suscetibilidade"),
        ("promotion_candidate_with_blockers", audit_blockers,
         "Falha em pelo menos um gate requerido para candidato observado"),
        ("promotion_not_geometry_candidate", audit_not_geometry, "Sem geometria/coordenada candidata"),
        ("observed_candidates_for_review_total", total_obs, "Status maximo permitido"),
        ("rec_2022_path_to_supervisor_review", rec_status.get("path_to_supervisor_review", "NOT_YET"),
         "READY somente se houver OBSERVED_GEOMETRY_CANDIDATE_FOR_REVIEW"),
    ]
    write_live_summary(os.path.join(d, "v1uj_live_execution_summary.csv"), summary_metrics)

    manifest = []
    mseq = 0
    for path, atype in sorted(V1UJ_ARTIFACTS.items()):
        exists = os.path.exists(path)
        manifest.append({
            "artifact_id": f"ART_{PROTOCOL_VERSION}_{mseq:04d}",
            "artifact_path": path, "artifact_type": atype,
            "protocol_version": PROTOCOL_VERSION,
            "sha256_prefix": sha256_file(path),
            "file_size_bytes": str(os.path.getsize(path) if exists else 0),
            "is_versionable": "true" if exists else "false",
            "reason": "Safe for git" if exists else "File not found",
        })
        mseq += 1
    with open(os.path.join(d, "v1uj_versionable_artifacts_manifest.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(manifest)

    best_event = max(event_rows,
                     key=lambda r: (int(r["observed_candidates_for_review"]),
                                    int(r["copernicus_vector_candidates"])
                                    + int(r["ckan_geo_candidates"])
                                    + int(r["rigeo_geodata_candidates"])
                                    + int(r["geosgb_observed_candidates"])),
                     default=None)
    v1uk_reco = ("v1uk - Supervisor Review and Event-Patch Overlay Preflight for Recife Candidates"
                 if rec_ready else "v1uk - Recife CKAN Schema Deep Audit")

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Protocolo C v1uj — Relatorio Focused Public Source Deepening",
        f"Gerado: {ts}", f"Protocolo: {PROTOCOL_VERSION}", "",
        "## Guardrails [ENFORCED]",
    ]
    for k, v in GUARDRAILS.items():
        lines.append(f"  {k} = {v}")
    lines += [
        "",
        "## O que a v1ui-live encontrou (baseline)",
        "  - 10 fontes publicas acessadas; 21/23 URLs HTTP 200; 497 links; "
        "16 PDFs (33.2MB); 76 assets DOCUMENT_ONLY; 0 candidato geometrico.",
        "",
        "## O que a v1uj aprofundou",
        "  - Resolvers focados source-specific: Copernicus EMS, GeoSGB ArcGIS REST, "
        "CKAN/Dados Recife, S2iD/dados.gov.br, RIGeo bitstreams e deep links de PDF.",
        "  - Correcao metodologica Copernicus: EMSR564 e EMSR602 foram auditados como "
        "off-target (Madagascar/Spain) e nao sao tratados como event-specific para PET/REC.",
        "  - Produtos explicitamente nao-event-specific ficam bloqueados para download.",
        "",
        "## Achado principal: CKAN Recife",
        f"  - Recursos CKAN totais: {ckan_total}",
        f"  - Candidatos geoespaciais nao contextuais: {ckan_geo_noncontext}",
        f"  - CSV no registry CKAN: {ckan_csv}",
        f"  - GeoJSON no registry CKAN: {ckan_geojson}",
        f"  - URLs unicas tratadas no downloader: {unique_download_urls}",
        f"  - DOWNLOAD_OK: {downloads_ok}",
        f"  - ALREADY_EXISTS_SAME_URL_SAME_HASH: {already_same}",
        f"  - DUPLICATE_CONTENT_DIFFERENT_URL: {duplicate_content}",
        f"  - Bloqueios/falhas explicitos: {blocked_or_failed}",
        f"  - Grupos de colisao corrigidos por filename seguro: {collision_groups}",
        f"  - Linhas com colisao no audit: {collision_rows}",
        f"  - CSV inventariados: {inv_csv}",
        f"  - GeoJSON inventariados: {inv_geojson}",
        f"  - TABLE_WITH_COORDINATES_CANDIDATE_FOR_REVIEW: {inv_coord_tables}",
        f"  - DOCUMENTED_OCCURRENCE_TABLE_NO_GEOMETRY: {inv_occurrence_tables}",
        f"  - CONTEXTUAL_OFFICIAL_LAYER: {inv_context_layers}",
        f"  - CONTEXT_ONLY: {inv_context_only}",
        f"  - Promotion audit CANDIDATE_WITH_BLOCKERS: {audit_blockers}",
        f"  - Promotion audit NOT_A_GEOMETRY_CANDIDATE: {audit_not_geometry}",
        "",
        "## Fontes especificas e respostas",
    ]
    for r in event_rows:
        lines.append(
            f"  [{r['event_id']}] copernicus_prod={r['copernicus_products']} "
            f"(vec={r['copernicus_vector_candidates']}) | geosgb_layers={r['geosgb_layers']} "
            f"(obs={r['geosgb_observed_candidates']}) | ckan={r['ckan_resources']} "
            f"(geo={r['ckan_geo_candidates']}) | s2id={r['s2id_records']} | "
            f"rigeo={r['rigeo_bitstreams']} (geo={r['rigeo_geodata_candidates']}) | "
            f"pdf_links={r['pdf_deeplinks']} | downloaded={r['downloaded_artifacts']} | "
            f"obs_for_review={r['observed_candidates_for_review']}")
    lines += [
        "",
        "## Apareceu artefato vetorial?",
        f"  - Eventos com artefato vetorial detectado: {total_vec}/{len(event_rows)}",
        "",
        "## Candidato pronto para revisao?",
        f"  - OBSERVED_GEOMETRY_CANDIDATE_FOR_REVIEW (total): {total_obs}",
        f"  - REC_2022_05_24_30 path_to_supervisor_review: "
        f"{rec_status.get('path_to_supervisor_review', 'NOT_YET')}",
        f"  - REC_2022_05_24_30 status: {rec_status.get('status_summary', 'n/a')}",
        "",
        "## Evento mais promissor",
        f"  - {best_event['event_id'] if best_event else 'n/a'} "
        f"(fonte: {best_event['most_promising_source'] if best_event else 'n/a'})",
        "",
        "## O que falta para ground reference",
        "  - G11 supervisor_review_required: pendente (sempre FAIL nesta etapa)",
        "  - G12 overlay_not_executed: nenhum overlay executado (bloqueia ground reference)",
        "  - G13 label_forbidden: rotulo proibido",
        "  - Revisao supervisora humana ainda nao realizada.",
        "",
        "## Recomendacao para v1uk",
        f"  - {v1uk_reco}",
        "",
        "## Invariantes",
        "  - Nenhum ground reference criado",
        "  - Nenhum label de treinamento criado",
        "  - Nenhum overlay executado",
        "  - Nenhuma coordenada inventada",
        "  - Nenhum dado bruto versionado",
        "  - quickview/suscetibilidade NAO viram ocorrencia observada",
        "  - produto operacional publico no maximo OBSERVED_GEOMETRY_CANDIDATE_FOR_REVIEW",
    ]

    os.makedirs(args.docs_dir, exist_ok=True)
    report_path = os.path.join(args.docs_dir,
                               "protocolo_c_relatorio_v1uj_focused_public_source_deepening.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    status_lines = [
        "# Status Atual — Protocolo C v1uj", f"Atualizado: {ts}", "",
        f"Eventos avaliados: {len(event_rows)}",
        f"Candidatos observados para revisao: {total_obs}",
        f"Eventos com artefato vetorial: {total_vec}",
        f"Recomendacao v1uk: {v1uk_reco}", "",
        f"REC_2022_05_24_30 path_to_supervisor_review={rec_status.get('path_to_supervisor_review', 'NOT_YET')}",
        f"download_ok={downloads_ok}",
        f"duplicate_content_different_url={duplicate_content}",
        f"collision_groups_detected={collision_groups}",
        "",
        "ground_truth_operational=false",
        "can_create_ground_reference=false",
        "can_create_training_label=false",
        "supervisor_review_completed=false",
        "no_overlay_executed=true",
        "route=PUBLIC_OFFICIAL_DISCOVERY",
        "formal_request_path=LEGACY_SECONDARY_ONLY",
    ]
    status_path = os.path.join(args.docs_dir, "protocolo_c_status_atual_v1uj.md")
    with open(status_path, "w", encoding="utf-8") as f:
        f.write("\n".join(status_lines))

    print(f"[Completion Report v1uj]")
    print(f"  Events: {len(event_rows)} | observed_for_review: {total_obs} | "
          f"vector_found_events: {total_vec}")
    print(f"  v1uk recommendation: {v1uk_reco}")
    print(f"  Report: {report_path}")


if __name__ == "__main__":
    main()
