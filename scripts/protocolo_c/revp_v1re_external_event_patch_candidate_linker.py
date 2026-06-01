"""REV-P v1re — External event/patch candidate linker.

Links external event candidates (v1rd) to existing patches as REVIEW-ONLY
link candidates, matched by region/hazard. A link is never proof of an event
and never a label. DINO never validates the link.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from revp_v1ra_v1rf_external_intake_common import (
    DATASETS,
    DOCS,
    SCHEMAS,
    _p,
    assert_clean_rows,
    guardrail_row,
    hash_short,
    normalize_patch_id,
    normalize_region,
    read_csv_safe,
    write_csv_with_header,
    write_doc,
    write_schema_safe,
)

ROOT = Path(__file__).resolve().parents[2]

IN_CANDIDATES = _p("REVP_V1RE_IN_CANDIDATES", DATASETS / "protocol_c_external_event_candidates_v1rd.csv")
IN_PATCHES = _p("REVP_V1RE_IN_PATCHES", DATASETS / "recife_event_patch_linkage_registry_v1ox.csv")
OUT_LINKS = _p("REVP_V1RE_OUT_LINKS", DATASETS / "protocol_c_external_event_patch_link_candidates_v1re.csv")
OUT_SUMMARY = _p("REVP_V1RE_OUT_SUMMARY", DATASETS / "protocol_c_external_event_patch_link_candidates_summary_v1re.csv")
SCHEMA_LINKS = _p("REVP_V1RE_SCHEMA_LINKS", SCHEMAS / "protocol_c_external_event_patch_link_candidates_v1re_schema.csv")
SCHEMA_SUMMARY = _p("REVP_V1RE_SCHEMA_SUMMARY", SCHEMAS / "protocol_c_external_event_patch_link_candidates_summary_v1re_schema.csv")
DOC = _p("REVP_V1RE_DOC", DOCS / "revp_v1re_external_event_patch_candidate_linker.md")

LINK_FIELDS = [
    "link_candidate_id", "event_candidate_id", "patch_id", "region",
    "hazard_type", "link_basis", "link_confidence", "link_status",
    "review_only", "dino_validates_event", "can_create_operational_label",
    "can_train_model", "target_created", "ground_truth_operational",
    "absence_as_negative", "notes",
]

SUMMARY_FIELDS = ["stat_key", "stat_value"]

NO_CANDIDATES = "EXTERNAL_INTAKE_WAITING_MANUAL_DOCUMENTS"
LINKS_READY = "EXTERNAL_LINK_CANDIDATES_READY_REVIEW_ONLY"


def _patches_by_region(patches: list[dict[str, str]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for r in patches:
        region = normalize_region(r.get("region", ""))
        pid = normalize_patch_id(r.get("patch_id", ""))
        if pid and pid != "UNKNOWN_PATCH":
            out.setdefault(region, [])
            if pid not in out[region]:
                out[region].append(pid)
    return out


def run(datasets: Path | None = None) -> dict[str, Any]:
    candidates = read_csv_safe(IN_CANDIDATES)
    patches = read_csv_safe(IN_PATCHES)
    by_region = _patches_by_region(patches)

    rows: list[dict[str, Any]] = []
    for c in candidates:
        region = normalize_region(c.get("region", ""))
        ev_id = c.get("event_candidate_id", "")
        region_patches = by_region.get(region, [])
        if not region_patches:
            row = {
                "link_candidate_id": f"V1RE_LNK_{hash_short(ev_id, 10)}_NONE",
                "event_candidate_id": ev_id, "patch_id": "",
                "region": region, "hazard_type": c.get("hazard_type", ""),
                "link_basis": "NO_PATCH_IN_REGION", "link_confidence": "NONE",
                "link_status": "NO_PATCH_AVAILABLE_REVIEW_ONLY", "notes": "",
            }
            row.update(guardrail_row())
            rows.append(row)
            continue
        for pid in region_patches:
            row = {
                "link_candidate_id": f"V1RE_LNK_{hash_short(ev_id + pid, 10)}",
                "event_candidate_id": ev_id, "patch_id": pid,
                "region": region, "hazard_type": c.get("hazard_type", ""),
                "link_basis": "REGION_HAZARD_MATCH",
                "link_confidence": "REVIEW_ONLY_LOW",
                "link_status": "LINK_CANDIDATE_REVIEW_ONLY", "notes": "not_proof_of_event",
            }
            row.update(guardrail_row())
            rows.append(row)

    assert_clean_rows(rows, "v1re_links")
    write_csv_with_header(OUT_LINKS, rows, LINK_FIELDS)
    write_schema_safe(SCHEMA_LINKS, LINK_FIELDS, "v1re_links")

    status = LINKS_READY if candidates else NO_CANDIDATES
    real_links = sum(1 for r in rows if r["link_status"] == "LINK_CANDIDATE_REVIEW_ONLY")
    summary = [
        {"stat_key": "link_status", "stat_value": status},
        {"stat_key": "event_candidates_in", "stat_value": str(len(candidates))},
        {"stat_key": "link_candidates", "stat_value": str(len(rows))},
        {"stat_key": "review_only_links", "stat_value": str(real_links)},
        {"stat_key": "stage", "stat_value": "v1re"},
    ]
    write_csv_with_header(OUT_SUMMARY, summary, SUMMARY_FIELDS)
    write_schema_safe(SCHEMA_SUMMARY, SUMMARY_FIELDS, "v1re_summary")

    write_doc(
        DOC,
        "v1re — External Event/Patch Candidate Linker",
        [
            "## Objetivo",
            "Ligar candidatos de evento externos (v1rd) a patches existentes como "
            "candidatos de link REVIEW-ONLY, por correspondencia de regiao/ameaca.",
            "## Resultado",
            f"Status: {status}. Candidatos de link: {len(rows)} ({real_links} review-only).",
            "## Guardrails",
            "Um link nunca e prova de evento e nunca e label. dino_validates_event=false. "
            "absence_as_negative=false. Nenhum target/ground truth operacional.",
        ],
    )
    print(f"[v1re] status={status} links={len(rows)}")
    return {"status": status, "links": len(rows)}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1re external event patch linker").parse_args()
    run()
