#!/usr/bin/env python3
"""
v1uf — Event Hydromet Scorecard

Builds a hydrometeorological scorecard per event, the v1ue->v1uf gate delta,
the next actions registry, and the v1uf closure report. Strong rainfall during
an event NEVER creates ground reference — only improves temporal plausibility.
"""

import argparse
import csv
import os
from datetime import datetime

PROTOCOL_VERSION = "v1uf"

SCORECARD_COLUMNS = [
    "event_id", "has_official_station_series", "has_precipitation_during_event",
    "has_pre_event_precipitation", "has_temporal_anchor", "has_station_coordinates",
    "has_spatial_event_geometry", "has_phenomenon_separation",
    "hydromet_evidence_level", "hydromet_summary", "remaining_blocker",
    "can_support_ground_reference_future", "can_create_ground_reference",
    "can_create_training_label", "next_best_action",
]

GATE_DELTA_COLUMNS = [
    "delta_id", "event_id", "gained_official_series", "gained_official_station",
    "gained_official_coordinate", "gained_precipitation_metric",
    "gained_temporal_coverage", "still_no_geometry", "still_no_supervisor_review",
    "still_no_label", "can_create_ground_reference", "notes",
]

NEXT_ACTIONS_COLUMNS = [
    "action_id", "event_id", "action_type", "priority", "description", "notes",
]

GUARDRAILS = {
    "ground_truth_operational": False,
    "can_create_ground_reference": False,
    "can_create_training_label": False,
    "can_reopen_protocol_b": False,
    "dino_usage": "SUPPORT_ONLY",
    "no_overlay_executed": True,
    "no_coordinates_invented": True,
    "supervisor_review_completed": False,
}

VALID_ACTIONS = [
    "SEND_FORMAL_REQUEST_SGB_FIELD_GEODATA",
    "SEND_FORMAL_REQUEST_DEFESA_CIVIL_OCCURRENCE_POINTS",
    "SEND_FORMAL_REQUEST_CEMADEN_EVENT_BULLETIN",
    "SEND_FORMAL_REQUEST_ANA_STATION_SERIES",
    "RETRY_INMET_STATION_EXTRACTION",
    "MANUAL_REVIEW_INMET_SERIES",
    "SEARCH_COPERNICUS_PRODUCT_BY_EVENT_ID",
    "SEARCH_CHARTER_PRODUCT_BY_ACTIVATION_ID",
    "PREPARE_HUMAN_REVIEW_PACKAGE",
    "DO_NOT_PROMOTE_GROUND_REFERENCE",
]


def load_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def classify_hydromet(has_series, has_event_precip, has_pre_precip, has_coord,
                      has_geometry, has_phenomenon_sep, hazard_scope, coverage_ok) -> tuple:
    if hazard_scope == "mixed" and not has_phenomenon_sep:
        return "BLOCKED_PHENOMENON_SEPARATION_REQUIRED", "PHENOMENON_SEPARATION_REQUIRED"
    if not has_geometry:
        geom_blocker = "EVENT_GEOMETRY_MISSING"
    else:
        geom_blocker = ""

    if not has_series:
        return "NO_STATION_DATA", "STATION_SERIES_MISSING"
    if has_series and not coverage_ok:
        return "BLOCKED_INSUFFICIENT_COVERAGE", "INSUFFICIENT_COVERAGE"
    if has_series and not has_coord:
        # series present but station coordinate unresolved
        if has_event_precip:
            return "OFFICIAL_WINDOW_DATA_AVAILABLE", "STATION_COORDINATES_MISSING"
        return "OFFICIAL_YEAR_DATA_AVAILABLE", "STATION_COORDINATES_MISSING"
    if has_series and has_coord and has_event_precip:
        return "TEMPORAL_HYDROMET_ANCHOR_CONFIRMED", geom_blocker or "EVENT_GEOMETRY_MISSING"
    if has_series and has_coord:
        return "TEMPORAL_ANCHOR_ONLY_NO_GEOMETRY", geom_blocker or "EVENT_GEOMETRY_MISSING"
    return "NO_STATION_DATA", "STATION_SERIES_MISSING"


def main():
    parser = argparse.ArgumentParser(description="v1uf — Event Hydromet Scorecard")
    parser.add_argument("--events", default="datasets/protocolo_c/event_candidate_registry.csv")
    parser.add_argument("--assets", default="datasets/protocolo_c/v1uf_station_series_asset_registry.csv")
    parser.add_argument("--catalog", default="datasets/protocolo_c/v1uf_official_station_catalog_registry.csv")
    parser.add_argument("--binding", default="datasets/protocolo_c/v1uf_station_binding_registry.csv")
    parser.add_argument("--metrics", default="datasets/protocolo_c/v1uf_hydromet_window_metrics_registry.csv")
    parser.add_argument("--v1ue-scorecard", default="datasets/protocolo_c/v1ue_event_evidence_scorecard.csv")
    parser.add_argument("--out-dir", default="datasets/protocolo_c")
    parser.add_argument("--out-report", default="docs/metodologia_cientifica/protocolo_c_relatorio_v1uf_station_resolved_acquisition.md")
    args = parser.parse_args()

    events = load_csv(args.events)
    assets = load_csv(args.assets)
    catalog = load_csv(args.catalog)
    binding = load_csv(args.binding)
    metrics = load_csv(args.metrics)
    v1ue_scorecard = {s["event_id"]: s for s in load_csv(args.v1ue_scorecard)}

    extracted_by_event = {}
    for a in assets:
        if a.get("extraction_status") == "EXTRACTED":
            extracted_by_event.setdefault(a["event_id"], []).append(a)

    coord_by_candidate = {c["station_candidate_id"]: c for c in catalog}
    bindings_by_event = {}
    for b in binding:
        bindings_by_event.setdefault(b["event_id"], []).append(b)

    metrics_by_event = {}
    for m in metrics:
        metrics_by_event.setdefault(m["event_id"], []).append(m)

    scorecard_rows = []
    delta_rows = []
    action_rows = []
    aseq = 0

    for event in events:
        event_id = event["event_id"]
        hazard_scope = event.get("hazard_scope", "")

        has_series = event_id in extracted_by_event
        ev_metrics = metrics_by_event.get(event_id, [])
        computed = [m for m in ev_metrics if m.get("metric_status") == "COMPUTED"]
        coverage_ok = len(computed) > 0

        # precipitation during event (core window) and pre-event
        has_event_precip = any(
            m.get("window_type") == "event_core_window" and m.get("metric_status") == "COMPUTED"
            and m.get("precipitation_total_mm") not in ("", "0.0")
            for m in ev_metrics
        )
        has_pre_precip = any(
            m.get("window_type", "").startswith("pre_event") and m.get("metric_status") == "COMPUTED"
            and m.get("precipitation_total_mm") not in ("", "0.0")
            for m in ev_metrics
        )

        # station coordinate resolved for this event
        has_coord = False
        for b in bindings_by_event.get(event_id, []):
            c = coord_by_candidate.get(b["station_candidate_id"], {})
            if c.get("coordinate_status") == "FROM_OFFICIAL_CATALOG":
                has_coord = True
                break

        has_geometry = False  # never in this stage
        has_phenomenon_sep = hazard_scope in ("flood", "inundation", "urban_flooding")
        has_temporal_anchor = coverage_ok

        level, blocker = classify_hydromet(
            has_series, has_event_precip, has_pre_precip, has_coord,
            has_geometry, has_phenomenon_sep, hazard_scope, coverage_ok,
        )

        # next best action
        if hazard_scope == "mixed" and not has_phenomenon_sep:
            next_action = "SEND_FORMAL_REQUEST_SGB_FIELD_GEODATA"
        elif not has_geometry:
            next_action = "SEND_FORMAL_REQUEST_DEFESA_CIVIL_OCCURRENCE_POINTS"
        elif not has_series:
            next_action = "RETRY_INMET_STATION_EXTRACTION"
        else:
            next_action = "PREPARE_HUMAN_REVIEW_PACKAGE"

        summary_parts = []
        if has_series:
            summary_parts.append(f"{len(extracted_by_event.get(event_id, []))} série(s) oficial(is) extraída(s)")
        if coverage_ok:
            summary_parts.append(f"{len(computed)} janela(s) com cobertura")
        if not has_geometry:
            summary_parts.append("sem geometria observada")
        summary = "; ".join(summary_parts) if summary_parts else "sem dados de estação oficial"

        scorecard_rows.append({
            "event_id": event_id,
            "has_official_station_series": str(has_series).lower(),
            "has_precipitation_during_event": str(has_event_precip).lower(),
            "has_pre_event_precipitation": str(has_pre_precip).lower(),
            "has_temporal_anchor": str(has_temporal_anchor).lower(),
            "has_station_coordinates": str(has_coord).lower(),
            "has_spatial_event_geometry": "false",
            "has_phenomenon_separation": str(has_phenomenon_sep).lower(),
            "hydromet_evidence_level": level,
            "hydromet_summary": summary,
            "remaining_blocker": blocker,
            "can_support_ground_reference_future": "true" if (has_series and coverage_ok) else "false",
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "next_best_action": next_action,
        })

        # Gate delta
        delta_rows.append({
            "delta_id": f"DEL_{PROTOCOL_VERSION}_{event_id}",
            "event_id": event_id,
            "gained_official_series": str(has_series).lower(),
            "gained_official_station": str(len(bindings_by_event.get(event_id, [])) > 0).lower(),
            "gained_official_coordinate": str(has_coord).lower(),
            "gained_precipitation_metric": str(coverage_ok).lower(),
            "gained_temporal_coverage": str(coverage_ok).lower(),
            "still_no_geometry": "true",
            "still_no_supervisor_review": "true",
            "still_no_label": "true",
            "can_create_ground_reference": "false",
            "notes": f"v1ue:{v1ue_scorecard.get(event_id, {}).get('classification', 'NA')} -> v1uf:{level}",
        })

        # Next actions (per event)
        actions_for_event = []
        if hazard_scope == "mixed" and not has_phenomenon_sep:
            actions_for_event.append(("SEND_FORMAL_REQUEST_SGB_FIELD_GEODATA", "1",
                                      "Solicitar geodata de campo SGB/CPRM com separação de fenômeno"))
            actions_for_event.append(("DO_NOT_PROMOTE_GROUND_REFERENCE", "1",
                                      "Evento misto sem separação — não promover"))
        actions_for_event.append(("SEND_FORMAL_REQUEST_DEFESA_CIVIL_OCCURRENCE_POINTS", "1",
                                  "Solicitar pontos de ocorrência georreferenciados (geometria)"))
        if not has_series:
            actions_for_event.append(("RETRY_INMET_STATION_EXTRACTION", "2",
                                      "Repetir download/extração da série INMET"))
        else:
            actions_for_event.append(("MANUAL_REVIEW_INMET_SERIES", "2",
                                      "Revisar manualmente a série INMET extraída"))
        if event_id.startswith("REC"):
            actions_for_event.append(("SEARCH_COPERNICUS_PRODUCT_BY_EVENT_ID", "2",
                                      "Buscar produto Copernicus EMS para o evento"))
        actions_for_event.append(("SEND_FORMAL_REQUEST_CEMADEN_EVENT_BULLETIN", "3",
                                  "Solicitar boletim Cemaden do evento"))
        actions_for_event.append(("SEND_FORMAL_REQUEST_ANA_STATION_SERIES", "3",
                                  "Solicitar série de estação ANA"))

        for atype, prio, desc in actions_for_event:
            action_rows.append({
                "action_id": f"ACT_{PROTOCOL_VERSION}_{aseq:04d}",
                "event_id": event_id,
                "action_type": atype,
                "priority": prio,
                "description": desc,
                "notes": "",
            })
            aseq += 1

    os.makedirs(args.out_dir, exist_ok=True)
    sc_path = os.path.join(args.out_dir, "v1uf_event_hydromet_scorecard.csv")
    with open(sc_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SCORECARD_COLUMNS)
        writer.writeheader()
        writer.writerows(scorecard_rows)

    delta_path = os.path.join(args.out_dir, "v1uf_gate_delta_registry.csv")
    with open(delta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=GATE_DELTA_COLUMNS)
        writer.writeheader()
        writer.writerows(delta_rows)

    na_path = os.path.join(args.out_dir, "v1uf_next_actions_registry.csv")
    with open(na_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=NEXT_ACTIONS_COLUMNS)
        writer.writeheader()
        writer.writerows(action_rows)

    generate_report(events, scorecard_rows, assets, catalog, metrics, action_rows, args.out_report)

    print(f"[Event Hydromet Scorecard v1uf] {len(scorecard_rows)} events")
    for r in scorecard_rows:
        print(f"  {r['event_id']}: {r['hydromet_evidence_level']} (blocker={r['remaining_blocker']})")
    print(f"\n  can_create_ground_reference=false (all)")
    print(f"  Scorecard: {sc_path}")
    print(f"  Gate delta: {delta_path}")
    print(f"  Next actions: {na_path}")
    print(f"  Report: {args.out_report}")


def generate_report(events, scorecard, assets, catalog, metrics, actions, out_md):
    sc_by_event = {s["event_id"]: s for s in scorecard}
    extracted = [a for a in assets if a.get("extraction_status") == "EXTRACTED"]
    coord_resolved = [c for c in catalog if c.get("coordinate_status") == "FROM_OFFICIAL_CATALOG"]
    computed_metrics = [m for m in metrics if m.get("metric_status") == "COMPUTED"]

    os.makedirs(os.path.dirname(out_md) or ".", exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(f"# Protocolo C — Relatório v1uf Station-Resolved Acquisition\n\n")
        f.write(f"**Gerado em:** {datetime.now().isoformat()}  \n")
        f.write(f"**Versão:** {PROTOCOL_VERSION}  \n\n")

        f.write("## Resumo\n\n")
        f.write(f"| Métrica | Valor |\n|---------|-------|\n")
        f.write(f"| Eventos | {len(events)} |\n")
        f.write(f"| Assets de série extraídos | {len(extracted)} |\n")
        f.write(f"| Coordenadas oficiais resolvidas | {len(coord_resolved)} |\n")
        f.write(f"| Métricas de janela computadas | {len(computed_metrics)} |\n")
        f.write(f"| Próximas ações | {len(actions)} |\n\n")

        f.write("## Tabela por Evento\n\n")
        f.write("| Evento | Nível Hidromet | Série | Coord | Precip Evento | Bloqueio |\n")
        f.write("|--------|---------------|-------|-------|---------------|----------|\n")
        for event in events:
            s = sc_by_event.get(event["event_id"], {})
            f.write(f"| {event['event_id']} | {s.get('hydromet_evidence_level', '')} | "
                    f"{s.get('has_official_station_series', '')} | {s.get('has_station_coordinates', '')} | "
                    f"{s.get('has_precipitation_during_event', '')} | {s.get('remaining_blocker', '')} |\n")

        f.write("\n## Perguntas-Chave\n\n")

        f.write("### Quais ZIPs oficiais foram baixados?\n")
        zips = set(a.get("zip_sha256", "") for a in extracted if a.get("zip_sha256"))
        if zips:
            f.write(f"- {len(zips)} ZIP(s) oficial(is) único(s) baixado(s) (por SHA256), em local_only/ (não versionado)\n")
        else:
            f.write("- Nenhum ZIP baixado nesta execução (download não autorizado, falhou, ou excedeu limite)\n")

        f.write("\n### Quais arquivos foram extraídos?\n")
        if extracted:
            for a in extracted:
                f.write(f"- {a['event_id']}: {a.get('station_name', '')} (code={a.get('station_code', '')})\n")
        else:
            f.write("- Nenhum arquivo de estação extraído nesta execução\n")

        f.write("\n### Quais estações foram associadas aos eventos?\n")
        f.write(f"- {len([c for c in catalog])} estação(ões) candidata(s) no catálogo v1uf\n")

        f.write("\n### Quais coordenadas oficiais foram resolvidas?\n")
        if coord_resolved:
            for c in coord_resolved:
                f.write(f"- {c['station_code']} ({c['municipality']}): lat={c['latitude']} lon={c['longitude']} [FROM_OFFICIAL_CATALOG]\n")
        else:
            f.write("- Nenhuma coordenada resolvida (catálogo oficial não acessível nesta execução) — permanecem MISSING\n")

        f.write("\n### Quais métricas de precipitação foram calculadas?\n")
        if computed_metrics:
            for m in computed_metrics[:20]:
                f.write(f"- {m['event_id']} / {m['window_type']}: total={m['precipitation_total_mm']}mm "
                        f"max_diário={m['precipitation_max_daily_mm']}mm (cobertura={m['coverage_ratio']})\n")
        else:
            f.write("- Nenhuma métrica com cobertura suficiente (séries não baixadas/extraídas nesta execução)\n")

        f.write("\n### Quais eventos ganharam âncora hidrometeorológica real?\n")
        for event in events:
            s = sc_by_event.get(event["event_id"], {})
            anchor = "SIM" if s.get("has_temporal_anchor") == "true" else "não"
            f.write(f"- {event['event_id']}: {anchor} (nível={s.get('hydromet_evidence_level', '')})\n")

        f.write("\n### Quais continuam só com portal genérico?\n")
        for event in events:
            s = sc_by_event.get(event["event_id"], {})
            if s.get("has_official_station_series") != "true":
                f.write(f"- {event['event_id']}: ainda sem série oficial extraída\n")

        f.write("\n### Quais continuam bloqueados por fenômeno misto?\n")
        for event in events:
            if event.get("hazard_scope") == "mixed":
                f.write(f"- {event['event_id']}: fenômeno MISTO — separação pendente\n")

        f.write("\n### Quais continuam bloqueados por ausência de geometria?\n")
        for event in events:
            f.write(f"- {event['event_id']}: SEM geometria observada (todos nesta etapa)\n")

        f.write("\n### O que falta para ground reference?\n")
        f.write("- Geometria observacional oficial (ausente em todos)\n")
        f.write("- Separação de fenômeno (eventos PET mistos)\n")
        f.write("- Revisão de supervisor (não executada)\n")
        f.write("- Overlay patch-evidência (não executado)\n")

        f.write("\n### Por que ainda não há ground truth operacional?\n")
        f.write("- `ground_truth_operational=false` é invariante do protocolo\n")
        f.write("- Chuva forte no evento melhora plausibilidade temporal, NÃO cria ground reference\n")
        f.write("- Estação é sensor pontual, não geometria de extensão de inundação\n")
        f.write("- Score/precipitação alto define apenas próxima ação\n")

        f.write("\n## Invariantes — Confirmação Explícita\n\n")
        for k, v in GUARDRAILS.items():
            f.write(f"- **{k}** = `{v}`\n")
        f.write(f"- **estação oficial não é geometria de inundação**\n")
        f.write(f"- **precipitação ancora plausibilidade temporal, não patch-level truth**\n")

        f.write(f"\n---\n*Relatório gerado por Protocol C {PROTOCOL_VERSION}.*\n")


if __name__ == "__main__":
    main()
