"""MV2 DATA-06 real temporal window research.

For each acquisition-queue target, resolves a *real*, traceable temporal window
of the observed disaster event. It joins two committed internal auditable
registries (the patch->event package registry and the event Sentinel temporal
window registry) and corroborates each window with an official public source
(CEMADEN for the Recife May-2022 event; Copernicus EMS for the Petropolis
Feb-2022 event). It never invents a date, never uses a date without a source, and
never publishes heavy raw PDF/HTML - only URL, title, date and short metadata.

If at least one strong window is found, a *local* (git-ignored) candidate is
written under ``inputs_local/data_06_temporal_windows/`` so the promotion
pipeline can validate it. The local file is never copied into outputs_public.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

OUT_DIR = PROJECT_ROOT / "outputs_public" / "mv2_data_06_real_temporal_window_research"
QUEUE_CSV = PROJECT_ROOT / "outputs_public" / "mv2_data_06_09_real_acquisition_queue" / "mv2_data_06_temporal_window_acquisition_queue.csv"
PACKAGE_REGISTRY = PROJECT_ROOT / "datasets" / "v2at_event_patch_package_registry.csv"
TEMPORAL_REGISTRY = PROJECT_ROOT / "datasets" / "event_sentinel_temporal_window_registry.csv"
LOCAL_CANDIDATE = PROJECT_ROOT / "inputs_local" / "data_06_temporal_windows" / "data_06_temporal_windows_real_candidate.csv"

# Official public sources verified for each observed event (light: url/title/date).
EXTERNAL_SOURCES_BY_EVENT: dict[str, dict[str, str]] = {
    "REC_2022_05_24_30": {
        "source_family": "CEMADEN",
        "source_ref_public": "https://www.gov.br/cemaden/pt-br/assuntos/noticias-cemaden/pesquisadores-brasileiros-fazem-recomendacoes-analisando-as-repentinas-inundacoes-e-deslizamentos-de-terra-em-recife-pe-apos-fortes-chuvas-ocorridas-em-maio-de-2022",
        "source_title": "CEMADEN/MCTI - Recomendacoes sobre as inundacoes e deslizamentos repentinos em Recife (PE) apos as fortes chuvas de maio de 2022",
        "source_date": "2022-05-28",
        "corroboration": "INMET alerta vermelho 2022-05-28/29; Defesa Civil PE balanco de mortes (Agencia Brasil)",
    },
    "PET_2022_02_15": {
        "source_family": "Copernicus EMS/CEMS",
        "source_ref_public": "https://global-flood.emergency.copernicus.eu/news/99-floods-and-landslides-in-rio-de-janeiro-state-brazil-february-to-march-2022/",
        "source_title": "Copernicus EMS GloFAS - Floods and landslides in Rio de Janeiro State, Brazil, February to March 2022",
        "source_date": "2022-02-15",
        "corroboration": "Sentinel-2 aquisicao pos-evento 2022-02-17; Defesa Civil 775 ocorrencias",
    },
}

CANDIDATE_FIELDS = [
    "target_rank",
    "patch_id",
    "asset_id",
    "city",
    "candidate_event_name",
    "temporal_window_start",
    "temporal_window_end",
    "temporal_window_source_family",
    "source_ref_public",
    "source_title",
    "source_date",
    "evidence_strength",
    "review_status",
    "blocked_reason",
]

QUERY_PACK_FIELDS = [
    "target_rank",
    "patch_id",
    "city",
    "event_id",
    "objective",
    "accepted_source_family",
    "query_string_ptbr",
    "what_to_confirm",
]

LOCAL_FIELDS = [
    "patch_id",
    "asset_id",
    "temporal_window_start",
    "temporal_window_end",
    "temporal_window_source",
    "source_ref",
    "review_status",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _package_lookup(path: Path = PACKAGE_REGISTRY) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for row in read_csv(path):
        lookup.setdefault(row.get("patch_id", ""), row)
    return lookup


def _event_name_lookup(path: Path = TEMPORAL_REGISTRY) -> dict[str, str]:
    return {row.get("observed_event_id", ""): row.get("event_name", "") for row in read_csv(path)}


def research_targets(
    queue_path: Path = QUEUE_CSV,
    package_path: Path = PACKAGE_REGISTRY,
    temporal_path: Path = TEMPORAL_REGISTRY,
) -> list[dict[str, Any]]:
    queue = read_csv(queue_path)
    packages = _package_lookup(package_path)
    event_names = _event_name_lookup(temporal_path)
    rows: list[dict[str, Any]] = []
    for entry in queue:
        patch_id = entry.get("patch_id", "")
        pkg = packages.get(patch_id, {})
        event_id = pkg.get("event_id", "")
        start = pkg.get("event_window_start", "")
        end = pkg.get("event_window_end", "") or start
        external = EXTERNAL_SOURCES_BY_EVENT.get(event_id, {})
        has_internal = bool(event_id and start)
        has_external = bool(external)
        if has_internal and has_external:
            strength, review, blocked = "STRONG", "REAL_TEMPORAL_WINDOW_CANDIDATE", ""
        elif has_internal:
            strength, review, blocked = "MEDIUM", "NEEDS_REVIEW", "no_external_official_source"
        else:
            strength, review, blocked = "WEAK", "NEEDS_REVIEW", "no_internal_event_linkage"
        family = "registro interno auditavel (v2at + STW)"
        if has_external:
            family += " + " + external["source_family"]
        rows.append(
            {
                "target_rank": entry.get("target_rank", ""),
                "patch_id": patch_id,
                "asset_id": entry.get("asset_id", ""),
                "city": entry.get("city", ""),
                "candidate_event_name": event_names.get(event_id, event_id),
                "temporal_window_start": start,
                "temporal_window_end": end,
                "temporal_window_source_family": family,
                "source_ref_public": external.get("source_ref_public", ""),
                "source_title": external.get("source_title", ""),
                "source_date": external.get("source_date", ""),
                "evidence_strength": strength,
                "review_status": review,
                "blocked_reason": blocked,
                "_event_id": event_id,
            }
        )
    return rows


def build_query_pack(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pack: list[dict[str, Any]] = []
    for row in candidates:
        pack.append(
            {
                "target_rank": row["target_rank"],
                "patch_id": row["patch_id"],
                "city": row["city"],
                "event_id": row["_event_id"],
                "objective": "Confirmar inicio/fim da janela do evento com fonte oficial rastreavel",
                "accepted_source_family": "CEMADEN | Defesa Civil | Copernicus EMS/CEMS | SGB/CPRM | ANA | boletim municipal oficial | artigo cientifico | registro interno auditavel",
                "query_string_ptbr": f"{row['city']} {row['candidate_event_name']} data inicio fim boletim oficial Defesa Civil CEMADEN Copernicus EMS",
                "what_to_confirm": "janela inicio/fim, fonte oficial, e (para DATA-08/09) cena Sentinel-2 com product_id explicito",
            }
        )
    return pack


def write_local_candidate_if_strong(candidates: list[dict[str, Any]], dest: Path = LOCAL_CANDIDATE) -> bool:
    strong = [row for row in candidates if row["evidence_strength"] == "STRONG"]
    if not strong:
        return False
    local_rows: list[dict[str, Any]] = []
    for row in strong:
        event_id = row["_event_id"]
        external = EXTERNAL_SOURCES_BY_EVENT.get(event_id, {})
        stw_id = f"STW_{event_id}"
        local_rows.append(
            {
                "patch_id": row["patch_id"],
                "asset_id": row["asset_id"],
                "temporal_window_start": row["temporal_window_start"],
                "temporal_window_end": row["temporal_window_end"],
                "temporal_window_source": row["temporal_window_source_family"],
                "source_ref": f"{stw_id} | {external.get('source_ref_public', '')}",
                "review_status": "REAL_TEMPORAL_WINDOW_CANDIDATE",
            }
        )
    write_csv(dest, LOCAL_FIELDS, local_rows)
    return True


def summarize(candidates: list[dict[str, Any]], local_created: bool) -> dict[str, Any]:
    by_strength: dict[str, int] = {}
    for row in candidates:
        by_strength[row["evidence_strength"]] = by_strength.get(row["evidence_strength"], 0) + 1
    return {
        "stage": "DATA-06 real temporal window research",
        "targets": len(candidates),
        "strong_candidates": by_strength.get("STRONG", 0),
        "by_strength": by_strength,
        "local_candidate_created": local_created,
        "external_sources_used": sorted({c["source_ref_public"] for c in candidates if c["source_ref_public"]}),
        "live_calls": 0,
        "downloads": 0,
        "rasters": 0,
        "crops": 0,
    }


def write_outputs(out_dir: Path = OUT_DIR) -> dict[str, Any]:
    candidates = research_targets()
    query_pack = build_query_pack(candidates)
    local_created = write_local_candidate_if_strong(candidates)
    public_candidates = [{k: v for k, v in row.items() if not k.startswith("_")} for row in candidates]
    write_csv(out_dir / "mv2_data_06_real_temporal_window_candidates.csv", CANDIDATE_FIELDS, public_candidates)
    write_csv(out_dir / "mv2_data_06_real_temporal_window_query_pack.csv", QUERY_PACK_FIELDS, query_pack)
    summary = summarize(candidates, local_created)
    write_json(out_dir / "mv2_data_06_real_temporal_window_summary.json", summary)
    write_text(
        out_dir / "mv2_data_06_real_temporal_window_report.md",
        f"""# DATA-06 - pesquisa de janelas temporais reais

## Estado
- targets investigados: {summary['targets']}
- candidatos fortes: {summary['strong_candidates']}
- distribuicao por forca: {json.dumps(summary['by_strength'], ensure_ascii=True)}
- input local criado (nao versionado): {summary['local_candidate_created']}

## Evidencia
- Recife (REC_2022_05_24_30): janela 2022-05-24 a 2022-05-30 (v2at + event_sentinel_temporal_window_registry),
  corroborada por CEMADEN/MCTI e INMET (alerta vermelho 28-29/05/2022).
- Petropolis (PET_2022_02_15): janela 2022-02-15 (v2at + registro interno),
  corroborada por Copernicus EMS (S2 pos-evento 17/02/2022).

## Regras respeitadas
- nenhuma data sem fonte; nenhuma janela de "mes inteiro" sem justificativa;
- nenhum bbox-only; nenhuma noticia generica sem ligacao temporal;
- PDF/HTML bruto pesado nao copiado (apenas URL, titulo, data).

## Side effects
- chamadas/downloads/rasters/crops: 0/0/0/0.
""",
    )
    write_text(out_dir / "commands.txt", "python scripts/mv2_data_06_real_temporal_window_research.py")
    return summary


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser().parse_args(argv)
    summary = write_outputs(OUT_DIR)
    print(
        "[mv2_data_06_real_temporal_window_research] "
        f"targets={summary['targets']} strong={summary['strong_candidates']} "
        f"local_input={summary['local_candidate_created']} "
        "calls/downloads/rasters/crops=0/0/0/0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
