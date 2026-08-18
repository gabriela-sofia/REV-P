"""Núcleo compartilhado do marco MV2-DATA-05 — Temporal Window Intake & Validation.

Cria o intake, a validação e o gate de promoção para janelas temporais por target do corpus,
a partir do template gerado no MV2-DATA-04. Produz a ponte válida
target_id+patch_id+asset_id+bbox+CRS+temporal_window_start/end+source_ref+review_status que
DATA-06 usará para liberar metadata probe.

Fail-closed:
- nunca inventa data; janela vazia/sem fonte/sem review nunca é promovida;
- data futura, start>end e janela ampla demais sem justificativa são rejeitadas;
- canary Protocolo-C nunca entra como corpus; T23KPR nunca auto-vinculado;
- NÃO executa GEE/STAC/OData, NÃO baixa raster, NÃO cria crop/feature, NÃO desbloqueia Dia 10.

Reusa IO/classificadores do MV2-13 e o batch/template do MV2-DATA-04 (uma só fonte de verdade).
"""

from __future__ import annotations

from pathlib import Path

from mv2_13_sentinel_binding_common import (  # noqa: F401  (reuso/reexport intencional)
    ISO_DATE_RE,
    PROJECT_ROOT,
    classify_datetime,
    clean,
    crs_status,
    read_csv_rows,
    write_csv,
    write_json,
)

PUBLIC_DIR = PROJECT_ROOT / "outputs_public" / "mv2_data_temporal_window_intake"
SCHEMA_DIR = PROJECT_ROOT / "datasets" / "schemas"

DATA04_DIR = PROJECT_ROOT / "outputs_public" / "mv2_data_corpus_metadata_probe"
DATA04_TEMPLATE = DATA04_DIR / "mv2_data_04_temporal_window_template.csv"
DATA04_BATCH = DATA04_DIR / "mv2_data_04_corpus_metadata_batch.csv"
DATA01_TARGETS = PROJECT_ROOT / "outputs_public" / "mv2_data_gee_lineage_targets" / "mv2_data_01_lineage_targets.csv"

BRANCH_EXECUTADA = "analysis/temporal-asset-readiness-mv1"

# Locais privados (git-ignored) onde o humano deposita o template preenchido.
FILLED_TEMPLATE_RELPATHS = [
    "local_only/mv2_data_temporal_window/mv2_data_05_temporal_window_filled.csv",
    "data_local/mv2_data_temporal_window/mv2_data_05_temporal_window_filled.csv",
    "private_outputs/mv2_data_temporal_window/mv2_data_05_temporal_window_filled.csv",
]

# Janela "ampla demais" sem justificativa: acima deste número de dias exige justificativa.
MAX_WINDOW_DAYS = 45
# review_status que contam como revisão aprovada.
APPROVED_REVIEW = {"APPROVED", "REVIEWED", "APROVADO", "REVISADO", "CONFIRMED"}

TEMPORAL_EVIDENCE_STATUSES = {
    "TEMPORAL_EVIDENCE_VALID_STRONG",
    "TEMPORAL_EVIDENCE_VALID_PARTIAL",
    "TEMPORAL_EVIDENCE_REVIEW_REQUIRED",
    "TEMPORAL_EVIDENCE_EMPTY",
    "TEMPORAL_EVIDENCE_CONFLICT",
    "TEMPORAL_EVIDENCE_INVALID",
}

TEMPORAL_PROMOTION_STATUSES = {
    "TEMPORAL_WINDOW_PROMOTED_STRONG",
    "TEMPORAL_WINDOW_PROMOTED_PARTIAL",
    "TEMPORAL_WINDOW_REVIEW_REQUIRED",
    "TEMPORAL_WINDOW_BLOCKED_EMPTY",
    "TEMPORAL_WINDOW_BLOCKED_INVALID",
    "TEMPORAL_WINDOW_BLOCKED_CONFLICT",
}

RISK_TYPES = [
    "RISK_TEMPORAL_WINDOW_INVENTED",
    "RISK_TEMPORAL_WINDOW_TOO_BROAD",
    "RISK_FUTURE_DATE",
    "RISK_START_AFTER_END",
    "RISK_SOURCE_MISSING",
    "RISK_REVIEW_STATUS_MISSING",
    "RISK_EVENT_DATE_AS_SCENE_DATE",
    "RISK_EXPORT_DATE_AS_ACQUISITION_DATE",
    "RISK_STAC_PROBE_WITH_WEAK_TEMPORALITY",
    "RISK_DAY10_FALSE_UNLOCK",
    "RISK_SANDBOX_FALSE_UNLOCK",
]


def ensure_public_dir() -> Path:
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    return PUBLIC_DIR


def find_filled_template() -> tuple[Path | None, str]:
    """Localiza o template preenchido privado. (path|None, location_type)."""
    for rel in FILLED_TEMPLATE_RELPATHS:
        p = PROJECT_ROOT / rel
        if p.exists():
            return p, "PRIVATE_FILLED"
    return None, "NONE"


def load_intake_source() -> tuple[list[dict[str, str]], bool, str]:
    """Carrega a fonte do intake: template preenchido privado, ou o template público vazio.

    Retorna (linhas, filled_template_found, origem).
    """
    path, _ = find_filled_template()
    if path is not None:
        return read_csv_rows(path), True, path.relative_to(PROJECT_ROOT).as_posix()
    # Fallback: template público DATA-04 (vazio) -> nenhuma promoção.
    return read_csv_rows(DATA04_TEMPLATE), False, DATA04_TEMPLATE.relative_to(PROJECT_ROOT).as_posix()


def corpus_index() -> dict[str, dict[str, str]]:
    """Targets do corpus (DATA-01) por target_id, para validar existência/asset/patch."""
    return {clean(r.get("target_id")): r for r in read_csv_rows(DATA01_TARGETS)}


def batch_index() -> dict[str, dict[str, str]]:
    """Batch DATA-04 por target_id (traz bbox/CRS/region)."""
    return {clean(r.get("target_id")): r for r in read_csv_rows(DATA04_BATCH)}


def window_days(start: str, end: str) -> int | None:
    """Número de dias entre start e end (ISO). None se não parseável."""
    from datetime import date

    s, e = clean(start), clean(end)
    if not (ISO_DATE_RE.match(s) and ISO_DATE_RE.match(e)):
        return None
    try:
        ds = date(int(s[:4]), int(s[5:7]), int(s[8:10]))
        de = date(int(e[:4]), int(e[5:7]), int(e[8:10]))
    except ValueError:
        return None
    return (de - ds).days
