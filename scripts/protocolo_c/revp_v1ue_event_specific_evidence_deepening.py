#!/usr/bin/env python3
"""
v1ue — Event-Specific Evidence Deepening (orchestrator)

Binds source -> event -> temporal window -> station, consolidates the
event-specific evidence registry, generates next actions and the v1ue report.

Optionally downloads year-specific datasets when --allow-web --download and
the dataset is approved (domain allowlist + license + size limit).
Never creates ground truth, label, geometry, or invents coordinates.
"""

import argparse
import csv
import hashlib
import os
import sys
import time
from datetime import datetime
from urllib.parse import urlparse

try:
    import yaml
except ImportError:
    yaml = None

try:
    import requests
except ImportError:
    requests = None

PROTOCOL_VERSION = "v1ue"
USER_AGENT = (
    "REV-P-AcademicResearch/1.0 "
    "(TCC ground reference evidence acquisition; contact: academic use only)"
)

EVIDENCE_COLUMNS = [
    "evidence_id", "event_id", "source_id", "city", "region",
    "window_types_linked", "station_candidates_linked", "dataset_resolution_id",
    "dataset_specificity", "asset_acquired", "asset_path_hash",
    "evidence_dimension", "evidence_role", "is_generic_portal",
    "is_event_specific", "can_create_ground_reference", "can_create_training_label",
    "supervisor_review_completed", "notes",
]

NEXT_ACTIONS_COLUMNS = [
    "action_id", "event_id", "source_id", "action_type", "priority",
    "description", "target", "estimated_effort", "notes",
]

GUARDRAILS = {
    "ground_truth_operational": False,
    "can_create_training_label": False,
    "can_reopen_protocol_b": False,
    "dino_usage": "SUPPORT_ONLY",
    "no_overlay_executed": True,
    "no_coordinates_invented": True,
}

VALID_ACTIONS = [
    "DOWNLOAD_YEAR_SPECIFIC_INMET_SERIES", "IDENTIFY_NEAREST_OFFICIAL_STATIONS",
    "REQUEST_ANA_STATION_SERIES", "REQUEST_CEMADEN_EVENT_REPORT",
    "REQUEST_SGB_FIELD_GEODATA", "REQUEST_DEFESA_CIVIL_OCCURRENCE_POINTS",
    "MANUAL_PDF_REVIEW", "SEARCH_COPERNICUS_EVENT_PRODUCT",
    "SEARCH_CHARTER_PRODUCT_NOT_QUICKVIEW", "KEEP_CONTEXT_ONLY",
    "READY_FOR_HUMAN_REVIEW_PACKAGE", "DO_NOT_PROMOTE",
]


def load_yaml(path: str) -> dict:
    if yaml is None:
        print("[ERROR] pyyaml not installed")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha256_file(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def maybe_download(url: str, dest_dir: str, max_mb: float, timeout: int) -> dict:
    if requests is None:
        return {"status": "DEPENDENCY_MISSING", "path": "", "hash": ""}
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout, stream=True, allow_redirects=True)
        resp.raise_for_status()
        cl = resp.headers.get("Content-Length")
        if cl and int(cl) > max_mb * 1024 * 1024:
            return {"status": "REJECTED_TOO_LARGE", "path": "", "hash": ""}
        os.makedirs(dest_dir, exist_ok=True)
        fname = os.path.basename(urlparse(url).path) or "dataset.bin"
        dest = os.path.join(dest_dir, fname[:200])
        limit = int(max_mb * 1024 * 1024)
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                downloaded += len(chunk)
                if downloaded > limit:
                    f.close()
                    os.remove(dest)
                    return {"status": "REJECTED_TOO_LARGE", "path": "", "hash": ""}
                f.write(chunk)
        return {"status": "DOWNLOADED", "path": dest, "hash": sha256_file(dest), "size": downloaded}
    except requests.exceptions.RequestException as e:
        return {"status": "NETWORK_ERROR", "path": "", "hash": "", "error": str(e)[:200]}


def determine_dimension(source_id: str, dataset_specificity: str, is_generic: bool) -> tuple[str, str]:
    if source_id in ("INMET_BDMEP", "ANA_HIDROWEB", "CEMADEN_PLUVIOMETROS"):
        if is_generic:
            return "temporal_generic", "context_then_temporal"
        return "temporal", "temporal_anchor"
    if source_id == "SGB_CPRM_CARTOGRAFIA":
        return "phenomenological_context", "phenomenon_typing_context"
    if source_id == "COPERNICUS_EMS":
        return "operational", "operational_product_candidate"
    if source_id == "INTERNATIONAL_CHARTER":
        return "operational_contextual", "contextual_only"
    return "generic", "context_only"


def build_next_actions(events, scorecard, resolutions, stations) -> list[dict]:
    actions = []
    seq = 0
    score_by_event = {s["event_id"]: s for s in scorecard}

    for event in events:
        event_id = event["event_id"]
        hazard = event.get("hazard_scope", "")
        score = score_by_event.get(event_id, {})
        classification = score.get("classification", "")

        ev_res = [r for r in resolutions if r["event_id"] == event_id]

        # INMET year-specific download
        inmet_year = [r for r in ev_res if r["source_id"] == "INMET_BDMEP" and r.get("is_year_specific") == "true"]
        if inmet_year:
            actions.append(_mk(seq, event_id, "INMET_BDMEP", "DOWNLOAD_YEAR_SPECIFIC_INMET_SERIES", "1",
                               "Baixar série anual INMET específica do ano do evento", inmet_year[0]["candidate_url"]))
            seq += 1

        # Station identification
        actions.append(_mk(seq, event_id, "INMET_BDMEP", "IDENTIFY_NEAREST_OFFICIAL_STATIONS", "1",
                           "Resolver estações oficiais mais próximas (coordenada do catálogo)", event.get("city", "")))
        seq += 1

        # ANA series
        if any(r["source_id"] == "ANA_HIDROWEB" for r in ev_res):
            actions.append(_mk(seq, event_id, "ANA_HIDROWEB", "REQUEST_ANA_STATION_SERIES", "2",
                               "Solicitar/baixar série de estação fluviométrica ANA", event.get("city", "")))
            seq += 1

        # Cemaden
        if any(r["source_id"] == "CEMADEN_PLUVIOMETROS" for r in ev_res):
            actions.append(_mk(seq, event_id, "CEMADEN_PLUVIOMETROS", "REQUEST_CEMADEN_EVENT_REPORT", "2",
                               "Solicitar boletim/relatório Cemaden do evento", event.get("city", "")))
            seq += 1

        # SGB geodata
        if any(r["source_id"] == "SGB_CPRM_CARTOGRAFIA" for r in ev_res):
            actions.append(_mk(seq, event_id, "SGB_CPRM_CARTOGRAFIA", "REQUEST_SGB_FIELD_GEODATA", "1",
                               "Solicitar geodata de campo SGB/CPRM (separar fenômeno)", event.get("city", "")))
            seq += 1

        # Defesa Civil occurrence points (always useful for geometry)
        actions.append(_mk(seq, event_id, "DEFESA_CIVIL", "REQUEST_DEFESA_CIVIL_OCCURRENCE_POINTS", "1",
                           "Solicitar pontos de ocorrência georreferenciados à Defesa Civil", event.get("city", "")))
        seq += 1

        # Copernicus
        if any(r["source_id"] == "COPERNICUS_EMS" for r in ev_res):
            actions.append(_mk(seq, event_id, "COPERNICUS_EMS", "SEARCH_COPERNICUS_EVENT_PRODUCT", "2",
                               "Buscar produto Copernicus EMS específico do evento", event.get("city", "")))
            seq += 1

        # Charter
        if any(r["source_id"] == "INTERNATIONAL_CHARTER" for r in ev_res):
            actions.append(_mk(seq, event_id, "INTERNATIONAL_CHARTER", "SEARCH_CHARTER_PRODUCT_NOT_QUICKVIEW", "3",
                               "Buscar produto validado Charter (NAO quickview)", event.get("city", "")))
            seq += 1

        # Phenomenon block for mixed
        if hazard == "mixed":
            actions.append(_mk(seq, event_id, "ALL", "DO_NOT_PROMOTE", "1",
                               "Evento misto sem separação de fenômeno — não promover", "phenomenon_separation"))
            seq += 1

        # Terminal action by classification
        if classification == "READY_FOR_HUMAN_REVIEW":
            actions.append(_mk(seq, event_id, "ALL", "READY_FOR_HUMAN_REVIEW_PACKAGE", "1",
                               "Montar pacote para revisão humana (sem promoção automática)", event_id))
            seq += 1
        else:
            actions.append(_mk(seq, event_id, "ALL", "KEEP_CONTEXT_ONLY", "2",
                               f"Manter como contexto — classificação: {classification}", event_id))
            seq += 1

    return actions


def _mk(seq, event_id, source_id, action_type, priority, desc, target):
    return {
        "action_id": f"ACT_{PROTOCOL_VERSION}_{seq:04d}",
        "event_id": event_id,
        "source_id": source_id,
        "action_type": action_type,
        "priority": priority,
        "description": desc,
        "target": target,
        "estimated_effort": "LOW" if action_type in ("KEEP_CONTEXT_ONLY", "DO_NOT_PROMOTE") else "MEDIUM",
        "notes": "",
    }


def build_evidence_registry(events, resolutions, windows, stations, observations) -> list[dict]:
    rows = []
    seq = 0
    for event in events:
        event_id = event["event_id"]
        ev_windows = [w["window_type"] for w in windows if w["event_id"] == event_id]
        ev_stations = [s["station_candidate_id"] for s in stations if s["event_id"] == event_id]
        ev_res = [r for r in resolutions if r["event_id"] == event_id]
        ev_obs = {o["source_id"]: o for o in observations if o["event_id"] == event_id}

        for res in ev_res:
            source_id = res["source_id"]
            is_generic = res.get("resolution_status") == "GENERIC_PORTAL"
            is_event_specific = res.get("is_event_specific") == "true"
            dimension, role = determine_dimension(source_id, res.get("resolution_status", ""), is_generic)

            obs = ev_obs.get(source_id, {})

            rows.append({
                "evidence_id": f"EVE_{PROTOCOL_VERSION}_{seq:04d}",
                "event_id": event_id,
                "source_id": source_id,
                "city": event.get("city", ""),
                "region": event.get("region", ""),
                "window_types_linked": "|".join(sorted(set(ev_windows))),
                "station_candidates_linked": "|".join(ev_stations),
                "dataset_resolution_id": res.get("dataset_resolution_id", ""),
                "dataset_specificity": res.get("resolution_status", ""),
                "asset_acquired": "true" if obs.get("asset_path_hash") else "false",
                "asset_path_hash": obs.get("asset_path_hash", ""),
                "evidence_dimension": dimension,
                "evidence_role": role,
                "is_generic_portal": str(is_generic).lower(),
                "is_event_specific": str(is_event_specific).lower(),
                "can_create_ground_reference": "false",
                "can_create_training_label": "false",
                "supervisor_review_completed": "false",
                "notes": res.get("blocking_reason", ""),
            })
            seq += 1
    return rows


def generate_report(events, scorecard, evidence, resolutions, stations, next_actions, out_md):
    score_by_event = {s["event_id"]: s for s in scorecard}

    os.makedirs(os.path.dirname(out_md) or ".", exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(f"# Protocolo C — Relatório v1ue Event-Specific Evidence Deepening\n\n")
        f.write(f"**Gerado em:** {datetime.now().isoformat()}  \n")
        f.write(f"**Versão:** {PROTOCOL_VERSION}  \n\n")

        generic_count = sum(1 for e in evidence if e["is_generic_portal"] == "true")
        specific_count = sum(1 for e in evidence if e["is_event_specific"] == "true")

        f.write("## Visão Geral\n\n")
        f.write(f"| Métrica | Valor |\n|---------|-------|\n")
        f.write(f"| Eventos | {len(events)} |\n")
        f.write(f"| Estações candidatas | {len(stations)} |\n")
        f.write(f"| Datasets resolvidos | {len(resolutions)} |\n")
        f.write(f"| Evidências evento-específicas | {specific_count} |\n")
        f.write(f"| Evidências ainda genéricas (portal) | {generic_count} |\n")
        f.write(f"| Próximas ações | {len(next_actions)} |\n\n")

        f.write("## Tabela por Evento\n\n")
        f.write("| Evento | Classificação | Temporal | Geometria | Fenômeno | Bloqueios |\n")
        f.write("|--------|---------------|----------|-----------|----------|----------|\n")
        for event in events:
            s = score_by_event.get(event["event_id"], {})
            f.write(f"| {event['event_id']} | {s.get('classification', '')} | "
                    f"{s.get('temporal_evidence_score', '')} | {s.get('geometry_score', '')} | "
                    f"{s.get('phenomenon_typing_score', '')} | {s.get('blocking_summary', '')} |\n")

        f.write("\n## Perguntas-Chave\n\n")

        f.write("### Quais evidências deixaram de ser homepage genérica?\n")
        if specific_count:
            for e in evidence:
                if e["is_event_specific"] == "true":
                    f.write(f"- {e['source_id']} / {e['event_id']}: {e['dataset_specificity']}\n")
        else:
            f.write("- Nenhuma ainda totalmente evento-específica. Datasets ano-específicos (INMET) resolvidos mas dependem de download/parse de série.\n")

        f.write("\n### Quais eventos ganharam âncora temporal?\n")
        for event in events:
            s = score_by_event.get(event["event_id"], {})
            t = float(s.get("temporal_evidence_score", 0) or 0)
            status = "SIM" if t >= 0.3 else "não"
            f.write(f"- {event['event_id']}: {status} (score={s.get('temporal_evidence_score', 0)})\n")

        f.write("\n### Quais eventos ganharam estação candidata oficial?\n")
        for event in events:
            n = sum(1 for st in stations if st["event_id"] == event["event_id"])
            f.write(f"- {event['event_id']}: {n} estação(ões) candidata(s)\n")

        f.write("\n### Quais eventos continuam sem geometria?\n")
        for event in events:
            s = score_by_event.get(event["event_id"], {})
            g = float(s.get("geometry_score", 0) or 0)
            if g == 0:
                f.write(f"- {event['event_id']}: SEM geometria observacional\n")

        f.write("\n### Quais eventos continuam sem separação de fenômeno?\n")
        for event in events:
            if event.get("hazard_scope") == "mixed":
                f.write(f"- {event['event_id']}: fenômeno MISTO — separação pendente\n")

        f.write("\n### Quais fontes exigem pedido formal?\n")
        formal = set()
        for a in next_actions:
            if a["action_type"] in ("REQUEST_SGB_FIELD_GEODATA", "REQUEST_ANA_STATION_SERIES",
                                     "REQUEST_CEMADEN_EVENT_REPORT", "REQUEST_DEFESA_CIVIL_OCCURRENCE_POINTS"):
                formal.add(a["source_id"])
        for src in sorted(formal):
            f.write(f"- {src}\n")

        f.write("\n### Qual é o próximo melhor alvo de aquisição?\n")
        f.write("1. Série anual INMET específica (download direto possível)\n")
        f.write("2. Pontos de ocorrência georreferenciados da Defesa Civil (geometria observacional)\n")
        f.write("3. Geodata de campo SGB/CPRM (com separação de fenômeno)\n")

        f.write("\n### O que falta para qualquer evento virar ground reference?\n")
        f.write("- Geometria observacional oficial (ausente em todos)\n")
        f.write("- Separação de fenômeno para eventos PET (mistos)\n")
        f.write("- Revisão de supervisor (não executada)\n")
        f.write("- Overlay patch-evidência (não executado nesta etapa)\n")

        f.write("\n### Por que ainda não há ground truth operacional?\n")
        f.write("- `ground_truth_operational=false` é invariante do protocolo nesta fase\n")
        f.write("- Score alto define apenas próxima ação, nunca cria label\n")
        f.write("- Gates G10 (overlay) e G11 (supervisor) permanecem FAIL\n")

        f.write("\n## Invariantes — Confirmação Explícita\n\n")
        for k, v in GUARDRAILS.items():
            f.write(f"- **{k}** = `{v}`\n")
        f.write(f"- **can_create_ground_reference** = `false` (todos os eventos)\n")
        f.write(f"- **supervisor_review_completed** = `false` (todos os eventos)\n")

        f.write(f"\n---\n*Relatório gerado por Protocol C {PROTOCOL_VERSION}.*\n")


def main():
    parser = argparse.ArgumentParser(description="v1ue — Event-Specific Evidence Deepening")
    parser.add_argument("--events", default="datasets/protocolo_c/event_candidate_registry.csv")
    parser.add_argument("--windows", default="datasets/protocolo_c/v1ue_event_temporal_window_registry.csv")
    parser.add_argument("--stations", default="datasets/protocolo_c/v1ue_station_candidate_registry.csv")
    parser.add_argument("--resolutions", default="datasets/protocolo_c/v1ue_official_dataset_resolution_registry.csv")
    parser.add_argument("--observations", default="datasets/protocolo_c/v1ue_observation_series_registry.csv")
    parser.add_argument("--scorecard", default="datasets/protocolo_c/v1ue_event_evidence_scorecard.csv")
    parser.add_argument("--domains-config", default="configs/protocolo_c/v1ud_allowed_domains.yaml")
    parser.add_argument("--out-dir", default="datasets/protocolo_c")
    parser.add_argument("--local-only-dir", default="local_only/protocolo_c")
    parser.add_argument("--out-report", default="docs/metodologia_cientifica/protocolo_c_relatorio_v1ue_event_specific_evidence_deepening.md")
    parser.add_argument("--allow-web", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-download-mb", type=float, default=50.0)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    events = load_csv(args.events)
    windows = load_csv(args.windows)
    stations = load_csv(args.stations)
    resolutions = load_csv(args.resolutions)
    observations = load_csv(args.observations)
    scorecard = load_csv(args.scorecard)

    print(f"[Event Deepening v1ue] Events: {len(events)} | Windows: {len(windows)} | "
          f"Stations: {len(stations)} | Resolutions: {len(resolutions)}")
    print(f"  dry_run={args.dry_run} allow_web={args.allow_web} download={args.download}")
    print(f"  ground_truth_operational=false can_create_training_label=false")

    # Optional year-specific download for approved INMET datasets
    downloaded = 0
    if args.allow_web and args.download and not args.dry_run:
        domains_config = load_yaml(args.domains_config)
        domain_map = {d["domain"]: d for d in domains_config.get("allowed_domains", [])}
        for res in resolutions:
            if res.get("is_downloadable") != "true":
                continue
            if res.get("license_status") == "UNKNOWN_NEEDS_REVIEW":
                continue
            url = res["candidate_url"]
            host = urlparse(url).hostname or ""
            dinfo = domain_map.get(host, {})
            if not dinfo.get("download_allowed", False):
                continue
            max_mb = min(dinfo.get("max_download_mb", args.max_download_mb), args.max_download_mb)
            dest_dir = os.path.join(args.local_only_dir, "evidence_raw", "v1ue",
                                    res["source_id"], res["event_id"])
            result = maybe_download(url, dest_dir, max_mb, args.timeout)
            if result["status"] == "DOWNLOADED":
                downloaded += 1
            time.sleep(1.5)
        print(f"  Year-specific downloads completed: {downloaded}")

    evidence = build_evidence_registry(events, resolutions, windows, stations, observations)
    next_actions = build_next_actions(events, scorecard, resolutions, stations)

    os.makedirs(args.out_dir, exist_ok=True)
    ev_path = os.path.join(args.out_dir, "v1ue_event_specific_evidence_registry.csv")
    with open(ev_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EVIDENCE_COLUMNS)
        writer.writeheader()
        writer.writerows(evidence)

    na_path = os.path.join(args.out_dir, "v1ue_next_actions_registry.csv")
    with open(na_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=NEXT_ACTIONS_COLUMNS)
        writer.writeheader()
        writer.writerows(next_actions)

    generate_report(events, scorecard, evidence, resolutions, stations, next_actions, args.out_report)

    print(f"\n[Results]")
    print(f"  Evidence registry: {len(evidence)} rows -> {ev_path}")
    print(f"  Next actions: {len(next_actions)} -> {na_path}")
    print(f"  Report: {args.out_report}")
    print(f"  can_create_ground_reference=false (all)")


if __name__ == "__main__":
    main()
