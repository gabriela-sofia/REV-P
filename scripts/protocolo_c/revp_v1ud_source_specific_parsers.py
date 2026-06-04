#!/usr/bin/env python3
"""
v1ud — Source-Specific Parsers

Defensive metadata parsers for each source type.
Produces next_actions_registry.csv and updates evidence extraction registry.
Also generates the v1ud closure report.
"""

import argparse
import csv
import json
import os
from datetime import datetime

try:
    import yaml
except ImportError:
    yaml = None

PROTOCOL_VERSION = "v1ud"

NEXT_ACTIONS_COLUMNS = [
    "action_id", "source_id", "event_id", "evidence_id",
    "action_type", "priority", "description", "target_artifact",
    "estimated_effort", "notes",
]

ACTION_TYPES = [
    "DOWNLOAD_RETRY", "MANUAL_REVIEW", "FORMAL_REQUEST_REQUIRED",
    "LICENSE_REVIEW_REQUIRED", "PDF_TEXT_REVIEW", "GEOMETRY_AUDIT_REQUIRED",
    "PHENOMENON_SEPARATION_REQUIRED", "EVENT_SPECIFIC_SOURCE_NEEDED",
    "KEEP_CONTEXT_ONLY", "REJECT_AS_GROUND_REFERENCE",
]

GUARDRAILS = {
    "ground_truth_operational": False,
    "can_create_training_label": False,
    "can_reopen_protocol_b": False,
    "dino_usage": "SUPPORT_ONLY",
    "no_overlay_executed": True,
    "no_coordinates_invented": True,
}


def load_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_yaml(path: str) -> dict:
    if yaml is None:
        return {}
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_inmet(entry: dict) -> dict:
    status = entry.get("acquisition_status", "")
    if status == "DOWNLOADED":
        return {
            "parsed": True,
            "source_type": "INMET_HISTORICAL",
            "data_format": entry.get("mime_type", ""),
            "event_specific": False,
            "portal_generic": "portal.inmet" in entry.get("url", ""),
            "has_temporal_data": True,
            "action": "PDF_TEXT_REVIEW" if ".pdf" in entry.get("url", "").lower() else "MANUAL_REVIEW",
        }
    return {"parsed": False, "action": "DOWNLOAD_RETRY" if status in ("NETWORK_ERROR",) else "METADATA_ONLY"}


def parse_ana(entry: dict) -> dict:
    url = entry.get("url", "").lower()
    if "serieshistoricas" in url or "hidroweb" in url:
        return {
            "parsed": True,
            "source_type": "ANA_HIDROWEB",
            "portal_generic": "apresentacao" in url,
            "has_temporal_data": True,
            "action": "FORMAL_REQUEST_REQUIRED" if entry.get("acquisition_status") != "DOWNLOADED" else "MANUAL_REVIEW",
        }
    return {"parsed": False, "action": "METADATA_ONLY"}


def parse_cemaden(entry: dict) -> dict:
    return {
        "parsed": True,
        "source_type": "CEMADEN_PUBLIC",
        "portal_generic": True,
        "has_temporal_data": True,
        "action": "KEEP_CONTEXT_ONLY",
    }


def parse_sgb_cprm(entry: dict) -> dict:
    url = entry.get("url", "").lower()
    is_specific = "setorizacao" in url or "risco" in url
    return {
        "parsed": True,
        "source_type": "SGB_CPRM_PRODUCT",
        "portal_generic": not is_specific,
        "event_specific": is_specific,
        "has_geometry": False,
        "action": "FORMAL_REQUEST_REQUIRED",
    }


def parse_copernicus(entry: dict) -> dict:
    return {
        "parsed": True,
        "source_type": "COPERNICUS_EMS",
        "portal_generic": True,
        "event_specific": False,
        "action": "EVENT_SPECIFIC_SOURCE_NEEDED",
    }


def parse_charter(entry: dict) -> dict:
    url = entry.get("url", "").lower()
    is_activation = "activation" in url or "756" in url
    return {
        "parsed": True,
        "source_type": "CHARTER_ACTIVATION" if is_activation else "CHARTER_PORTAL",
        "portal_generic": not is_activation,
        "quickview_only": True,
        "action": "KEEP_CONTEXT_ONLY",
        "note": "Quickview = contextual only, never ground reference",
    }


def parse_maxar(entry: dict) -> dict:
    return {
        "parsed": True,
        "source_type": "MAXAR_OPEN_DATA",
        "portal_generic": True,
        "action": "LICENSE_REVIEW_REQUIRED",
    }


def parse_planet(entry: dict) -> dict:
    return {
        "parsed": True,
        "source_type": "PLANET_DISASTER",
        "portal_generic": True,
        "action": "LICENSE_REVIEW_REQUIRED",
    }


def parse_emdat(entry: dict) -> dict:
    return {
        "parsed": True,
        "source_type": "EMDAT_INVENTORY",
        "portal_generic": True,
        "action": "KEEP_CONTEXT_ONLY",
        "note": "EM-DAT = context only, never patch-level geometry",
    }


SOURCE_PARSERS = {
    "INMET_BDMEP": parse_inmet,
    "ANA_HIDROWEB": parse_ana,
    "CEMADEN_PLUVIOMETROS": parse_cemaden,
    "SGB_CPRM_CARTOGRAFIA": parse_sgb_cprm,
    "COPERNICUS_EMS": parse_copernicus,
    "INTERNATIONAL_CHARTER": parse_charter,
    "MAXAR_OPEN_DATA": parse_maxar,
    "PLANET_DISASTER_DATA": parse_planet,
    "EMDAT": parse_emdat,
}


def generate_next_actions(
    extractions: list[dict],
    resolutions: list[dict],
    integrity: list[dict],
    gate_delta: list[dict],
) -> list[dict]:
    actions = []
    seq = 0

    int_map = {r.get("extraction_id", ""): r for r in integrity}
    delta_map = {r.get("evidence_id", ""): r for r in gate_delta}

    for ext in extractions:
        source_id = ext.get("source_id", "")
        event_id = ext.get("event_id", "")
        ext_id = ext.get("extraction_id", "")
        status = ext.get("acquisition_status", "")

        parser = SOURCE_PARSERS.get(source_id)
        if parser:
            parsed = parser(ext)
        else:
            parsed = {"parsed": False, "action": "MANUAL_REVIEW"}

        action_type = parsed.get("action", "MANUAL_REVIEW")

        if status == "NETWORK_ERROR":
            action_type = "DOWNLOAD_RETRY"
        elif status == "REJECTED_TOO_LARGE":
            action_type = "MANUAL_REVIEW"
        elif status == "DEPENDENCY_MISSING":
            action_type = "DOWNLOAD_RETRY"
        elif status in ("SKIPPED", "DRY_RUN"):
            if source_id in ("EMDAT", "INTERNATIONAL_CHARTER"):
                action_type = "KEEP_CONTEXT_ONLY"
            elif source_id in ("MAXAR_OPEN_DATA", "PLANET_DISASTER_DATA"):
                action_type = "LICENSE_REVIEW_REQUIRED"

        int_entry = int_map.get(ext_id, {})
        if int_entry.get("file_category") == "PDF" and int_entry.get("probe_status") == "PDF_PARSED":
            if not int_entry.get("pdf_text_sample"):
                action_type = "PDF_TEXT_REVIEW"

        delta = delta_map.get(ext_id, {})
        if delta.get("gained_geometry") == "true":
            action_type = "GEOMETRY_AUDIT_REQUIRED"

        description = f"{action_type} for {source_id}/{event_id}"
        if parsed.get("portal_generic"):
            description += " (portal generic — may need event-specific URL)"
        if parsed.get("quickview_only"):
            description += " (quickview = contextual only)"

        actions.append({
            "action_id": f"ACT_{PROTOCOL_VERSION}_{seq:04d}",
            "source_id": source_id,
            "event_id": event_id,
            "evidence_id": ext_id,
            "action_type": action_type,
            "priority": "1" if action_type in ("FORMAL_REQUEST_REQUIRED", "DOWNLOAD_RETRY") else "2",
            "description": description,
            "target_artifact": ext.get("url", ""),
            "estimated_effort": "LOW" if action_type in ("KEEP_CONTEXT_ONLY", "REJECT_AS_GROUND_REFERENCE") else "MEDIUM",
            "notes": parsed.get("note", ""),
        })
        seq += 1

    return actions


def generate_report(
    extractions: list[dict],
    resolutions: list[dict],
    integrity: list[dict],
    gate_delta: list[dict],
    next_actions: list[dict],
    sources_config: dict,
    out_md: str,
):
    total = len(extractions)
    downloaded = sum(1 for e in extractions if e.get("acquisition_status") == "DOWNLOADED")
    skipped = sum(1 for e in extractions if e.get("acquisition_status") == "SKIPPED")
    dry_run = sum(1 for e in extractions if e.get("acquisition_status") == "DRY_RUN")
    errors = sum(1 for e in extractions if e.get("acquisition_status") == "NETWORK_ERROR")
    dep_missing = sum(1 for e in extractions if e.get("acquisition_status") == "DEPENDENCY_MISSING")

    gained_hash = sum(1 for d in gate_delta if d.get("gained_hash") == "true")
    gained_pdf = sum(1 for d in gate_delta if d.get("gained_pdf_text") == "true")
    gained_geo = sum(1 for d in gate_delta if d.get("gained_geometry") == "true")
    gained_html = sum(1 for d in gate_delta if d.get("gained_html_links") == "true")
    still_blocked = sum(1 for d in gate_delta if d.get("still_blocked") == "true")

    action_counts = {}
    for a in next_actions:
        t = a.get("action_type", "UNKNOWN")
        action_counts[t] = action_counts.get(t, 0) + 1

    generic_count = sum(
        1 for e in extractions
        if SOURCE_PARSERS.get(e.get("source_id", ""), lambda x: {})({}).get("portal_generic", False)
    )

    os.makedirs(os.path.dirname(out_md) or ".", exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(f"# Protocolo C — Relatório v1ud Real Source Acquisition\n\n")
        f.write(f"**Gerado em:** {datetime.now().isoformat()}  \n")
        f.write(f"**Versão:** {PROTOCOL_VERSION}  \n\n")

        f.write("## Resumo de Aquisição\n\n")
        f.write(f"| Métrica | Valor |\n|---------|-------|\n")
        f.write(f"| Total de entradas | {total} |\n")
        f.write(f"| Downloads completados | {downloaded} |\n")
        f.write(f"| Skipped (não permitido) | {skipped} |\n")
        f.write(f"| Dry-run | {dry_run} |\n")
        f.write(f"| Erros de rede | {errors} |\n")
        f.write(f"| Dependência ausente | {dep_missing} |\n\n")

        f.write("## Gate Delta (v1uc → v1ud)\n\n")
        f.write(f"| Ganho | Contagem |\n|-------|----------|\n")
        f.write(f"| Hash SHA256 | {gained_hash} |\n")
        f.write(f"| Texto de PDF | {gained_pdf} |\n")
        f.write(f"| Geometria | {gained_geo} |\n")
        f.write(f"| Links HTML | {gained_html} |\n")
        f.write(f"| Ainda bloqueados | {still_blocked} |\n\n")

        f.write("## Classificação de Fontes\n\n")
        f.write("| Tipo | Descrição |\n|------|-----------|")
        f.write("\n| Fonte genérica (portal homepage) | Página inicial de portal — precisa de URL específica |")
        f.write("\n| Fonte específica de evento | URL que referencia evento ou período específico |")
        f.write("\n| Potencial de referência | Fonte oficial com dados observacionais e geometria |")
        f.write("\n| Apenas contexto | Inventário macro, quickview, suscetibilidade |\n\n")

        f.write("## Próximas Ações\n\n")
        f.write(f"| Ação | Contagem |\n|------|----------|\n")
        for t, c in sorted(action_counts.items()):
            f.write(f"| {t} | {c} |\n")

        f.write("\n## Prioridades de Solicitação Formal\n\n")
        formal = [a for a in next_actions if a.get("action_type") == "FORMAL_REQUEST_REQUIRED"]
        if formal:
            for a in formal:
                f.write(f"1. **{a['source_id']}** / {a['event_id']}: {a.get('description', '')}\n")
        else:
            f.write("Nenhuma solicitação formal pendente nesta execução.\n")

        f.write("\n## Por Que Ground Truth Continua Bloqueado\n\n")
        f.write("1. **G10 (patch_overlay_possible):** Overlay não implementado em v1ud\n")
        f.write("2. **G11 (supervisor_review_completed):** Revisão humana não executada\n")
        f.write("3. **Fenômeno misto:** Eventos PET ainda com separação pendente\n")
        f.write("4. **Geometria ausente:** Maioria das fontes não fornece geometria diretamente\n")
        f.write("5. **Licenças pendentes:** Algumas fontes requerem revisão de termos\n\n")

        f.write("## Invariantes — Confirmação Explícita\n\n")
        for k, v in GUARDRAILS.items():
            f.write(f"- **{k}** = `{v}`\n")

        f.write(f"\n---\n*Relatório gerado por Protocol C {PROTOCOL_VERSION}.*\n")


def main():
    parser = argparse.ArgumentParser(description="v1ud — Source-Specific Parsers & Next Actions")
    parser.add_argument("--extraction-registry", default="datasets/protocolo_c/v1ud_evidence_extraction_registry.csv")
    parser.add_argument("--resolution-registry", default="datasets/protocolo_c/v1ud_source_resolution_registry.csv")
    parser.add_argument("--integrity-registry", default="datasets/protocolo_c/v1ud_raw_asset_integrity_registry.csv")
    parser.add_argument("--gate-delta", default="datasets/protocolo_c/v1ud_gate_delta_registry.csv")
    parser.add_argument("--sources-config", default="configs/protocolo_c/ground_reference_evidence_sources.yaml")
    parser.add_argument("--out-actions", default="datasets/protocolo_c/v1ud_next_actions_registry.csv")
    parser.add_argument("--out-report", default="docs/metodologia_cientifica/protocolo_c_relatorio_v1ud_real_source_acquisition.md")
    args = parser.parse_args()

    extractions = load_csv(args.extraction_registry)
    resolutions = load_csv(args.resolution_registry)
    integrity = load_csv(args.integrity_registry)
    gate_delta = load_csv(args.gate_delta)
    sources_config = load_yaml(args.sources_config)

    print(f"[Parsers v1ud] Extractions: {len(extractions)} | Resolutions: {len(resolutions)}")

    next_actions = generate_next_actions(extractions, resolutions, integrity, gate_delta)

    os.makedirs(os.path.dirname(args.out_actions) or ".", exist_ok=True)
    with open(args.out_actions, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=NEXT_ACTIONS_COLUMNS)
        writer.writeheader()
        writer.writerows(next_actions)

    generate_report(
        extractions, resolutions, integrity, gate_delta,
        next_actions, sources_config, args.out_report,
    )

    print(f"  Next actions: {len(next_actions)}")
    print(f"  Actions registry: {args.out_actions}")
    print(f"  Report: {args.out_report}")
    print(f"  ground_truth_operational = false")
    print(f"  can_create_training_label = false")


if __name__ == "__main__":
    main()
