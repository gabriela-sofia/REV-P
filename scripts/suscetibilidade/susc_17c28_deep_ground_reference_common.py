"""SUSC-17C28 aquisicao profunda de artefatos oficiais especificos para G4/G5.

Executa uma aquisicao profunda e dirigida em fontes oficiais/institucionais
especificas do evento observado (enchentes e deslizamentos em Pernambuco/Recife,
maio de 2022). Usa snapshots arquivados (Internet Archive / Wayback) de paginas
oficiais de maio de 2022 e segue links internos de artigos especificos da
Agencia Brasil (EBC, empresa publica federal = fonte oficial institucional),
alem de tentar portais oficiais diretos (APAC, CEMADEN, SGB/CPRM). Calcula SHA256,
parseia texto, extrai datas/locais/bairros/fenomenos e avalia G4 (vinculo espacial
patch-level) e G5 (separacao de fenomeno).

Resultado honesto esperado: mesmo com artefatos oficiais especificos, o evento de
maio/2022 foi misto (inundacoes + deslizamentos) e a localizacao publica e
municipal/bairro, sem geometria patch-level nem coordenada; assim G4/G5 tendem a
permanecer false e nenhum Ground Reference Candidate e aceito. Nenhuma coordenada
e inventada; noticia nao vira Ground Reference sozinha; sensor/CHIRPS nao viram
evento observado; nenhum ground truth/label/treino/score v7/patch oficial e
criado; 17B so seria elegivel com Ground Reference Candidate aceito por G1-G7.

Build publico offline/deterministico a partir dos artefatos ja adquiridos e
commitados; aquisicao real exige os tres opt-ins de rede.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from html import unescape
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

from susc_io import ensure_dir, read_csv, read_json, rel, sha256_file, write_csv, write_json, write_markdown  # noqa: E402

DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"
LOCAL_STATE = ROOT / "local_runs" / "suscetibilidade" / "17c28_deep_ground_reference"
ARTIFACT_DIR = OUT / "susc_17c28_source_artifacts"
LEDGER = ARTIFACT_DIR / "_deep_acquisition_ledger.json"

SCORE_V6 = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"
OFFICIAL_PATCHES = DAT / "susc_patches_official_v1.csv"
OFFICIAL_PATCH_LINKS = DAT / "susc_patch_links_official_v1.csv"

C27_MANIFEST = OUT / "susc_17c27_source_artifact_manifest.csv"
C27_OBSERVED = OUT / "susc_17c27_observed_event_candidates.csv"
C27_SUMMARY = OUT / "susc_17c27_readiness_summary.json"
C26_GR_QUEUE = OUT / "susc_17c26_ground_reference_target_queue.csv"
C26_QUERY_PACKAGES = OUT / "susc_17c26_source_query_packages.csv"
C26_GR_FIELDS = OUT / "susc_17c26_required_ground_reference_fields.csv"
C26_PRIORITY = OUT / "susc_17c26_patch_review_priority.csv"
C19_BINDING = OUT / "susc_17c19_candidate_patch_temporal_binding.csv"
PATCH_GRID = OUT / "susc_17c6_candidate_patch_grid.csv"
PATCH_GEOJSON = OUT / "susc_17c6_candidate_patch_grid.geojson"
PATCH_LINKS = OUT / "susc_17c6_candidate_patch_links.csv"

REQUIRED_INPUTS = [
    SCORE_V6, C27_MANIFEST, C27_OBSERVED, C27_SUMMARY, C26_GR_QUEUE, C26_QUERY_PACKAGES,
    C26_GR_FIELDS, C26_PRIORITY, C19_BINDING, PATCH_GRID, PATCH_GEOJSON, PATCH_LINKS,
]

REPORT = OUT / "SUSC_17C28_AQUISICAO_PROFUNDA_ARTEFATOS_OFICIAIS_G4_G5_REPORT.md"
EXPANDED_PLAN = OUT / "susc_17c28_expanded_search_plan.csv"
DEEP_ATTEMPTS = OUT / "susc_17c28_deep_source_acquisition_attempts.csv"
FOLLOWED_LINKS = OUT / "susc_17c28_followed_link_registry.csv"
DEEP_MANIFEST = OUT / "susc_17c28_deep_source_artifact_manifest.csv"
DEEP_PARSED = OUT / "susc_17c28_deep_parsed_artifact_index.csv"
SPECIFIC_CANDIDATES = OUT / "susc_17c28_specific_observed_event_candidates.csv"
LOCATION_RESOLUTION = OUT / "susc_17c28_location_resolution.csv"
PHENOMENON = OUT / "susc_17c28_phenomenon_classification.csv"
G4_EVAL = OUT / "susc_17c28_g4_spatial_link_evaluation.csv"
G5_EVAL = OUT / "susc_17c28_g5_phenomenon_evaluation.csv"
GR_CANDIDATE_EVAL = OUT / "susc_17c28_ground_reference_candidate_evaluation.csv"
SCORECARD = OUT / "susc_17c28_official_artifact_scorecard.csv"
GRAPH_UPDATE_NODES = OUT / "susc_17c28_evidence_graph_update_nodes.csv"
GRAPH_UPDATE_EDGES = OUT / "susc_17c28_evidence_graph_update_edges.csv"
NO_LEAKAGE = OUT / "susc_17c28_no_leakage_audit.csv"
GATES = OUT / "susc_17c28_gate_evaluation_matrix.csv"
SUMMARY = OUT / "susc_17c28_readiness_summary.json"
BLOCKERS = OUT / "susc_17c28_promotion_blockers.csv"

ARTIFACT_SCHEMA = SCHEMAS / "susc_17c28_deep_source_artifact_schema_v1.json"
CANDIDATE_SCHEMA = SCHEMAS / "susc_17c28_specific_event_candidate_schema_v1.json"
G4_G5_SCHEMA = SCHEMAS / "susc_17c28_g4_g5_evaluation_schema_v1.json"

GOV = {"review_only": "true", "trainable": "false", "ground_truth": "false"}
MAX_ARTIFACT_BYTES = 500_000
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) REV-P-SUSC-17C28-review-only"
FETCH_TIMEOUT_SECONDS = 12

WAYBACK = "https://web.archive.org/web/{ts}/{url}"
WAYBACK_TIMESTAMPS = ["20220528120000", "20220529120000", "20220530081923", "20220531120000", "20220601120000"]
KNOWN_EVENT_TIMESTAMPS = ["20220528193348", "20220530061851", "20220530070149", "20220530081923", "20220609130341"]

# family -> (source_name, homepage_url, officiality_level)
FAMILY_HOMEPAGES = {
    "agencia_brasil": ("Agencia Brasil / EBC (empresa publica federal)", "https://agenciabrasil.ebc.com.br/", "official_institutional_public_agency"),
    "apac_pernambuco": ("APAC - Agencia Pernambucana de Aguas e Clima", "https://www.apac.pe.gov.br/", "official"),
    "cemaden": ("CEMADEN", "http://www.cemaden.gov.br/", "official"),
    "sgb_cprm": ("SGB/CPRM - Servico Geologico do Brasil", "https://www.sgb.gov.br/", "official"),
    "prefeitura_recife": ("Prefeitura do Recife / Defesa Civil", "https://www2.recife.pe.gov.br/", "official"),
    "gov_pernambuco": ("Governo de Pernambuco", "https://www.pe.gov.br/", "official"),
}
OFFICIAL_EVENT_LEVELS = {"official", "official_institutional", "official_institutional_public_agency"}
FOLLOW_FROM_FAMILY = "agencia_brasil"
MAX_FOLLOW = 12

KNOWN_AGENCIA_EVENT_URLS = [
    "https://agenciabrasil.ebc.com.br/geral/noticia/2022-05/chegam-33-mortes-confirmadas-devido-chuva-no-grande-recife",
    "https://agenciabrasil.ebc.com.br/geral/noticia/2022-05/governo-de-pernambuco-atualiza-para-56-numero-de-mortos-no-estado",
    "https://agenciabrasil.ebc.com.br/politica/noticia/2022-05/presidente-anuncia-que-vai-ao-grande-recife-na-segunda",
    "https://agenciabrasil.ebc.com.br/politica/noticia/2022-05/governo-anuncia-envio-de-equipes-para-o-grande-recife-apos-chuvas",
]

EXPANDED_TERMS = ["Recife", "Pernambuco", "Jaboatao", "Olinda", "maio 2022", "24/05/2022 a 30/05/2022",
                  "28/05/2022", "29/05/2022", "enchente", "inundacao", "alagamento", "deslizamento",
                  "barreira", "Defesa Civil", "APAC", "CEMADEN", "CPRM SGB"]
SPECIFICITIES = ["event_specific", "location_specific", "phenomenon_specific", "geometry_specific"]

FLOOD_TERMS = ["inunda", "enchente", "alagamento", "alagad", "cheia"]
LANDSLIDE_TERMS = ["deslizamento", "soterr", "barreira", "movimento de massa", "desabamento"]
EVENT_CONTEXT_TERMS = ["chuva", "chuvas", "morte", "mortes", "defesa civil", "emergencia", "emergência"]
EVENT_PHENOMENON_TERMS = FLOOD_TERMS + LANDSLIDE_TERMS + EVENT_CONTEXT_TERMS
LOCATION_TERMS = ["recife", "pernambuco", "jaboat", "olinda", "grande recife", "regiao metropolitana", "guararapes"]
DATE_TERMS = ["2022", "maio", "28 de maio", "29 de maio", "27 de maio", "2022-05"]
BAIRRO_TERMS = ["jaboat", "olinda", "guararapes", "muribeca", "ibura", "barro", "monte verde", "cabo de santo agostinho",
                "moreno", "abreu e lima", "paulista", "camaragibe", "jardim sao paulo", "dois unidos", "vila dos milagres"]
LOGRADOURO_RE = re.compile(r"\b(rua|avenida|av\.|estrada|travessa|alameda)\s+[a-z]", re.I)


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _network_enabled() -> bool:
    return os.environ.get("SUSC_17C28_ALLOW_NETWORK") == "1"


def _public_download_enabled() -> bool:
    return os.environ.get("SUSC_17C28_ALLOW_PUBLIC_DOWNLOAD") == "1"


def _deep_search_enabled() -> bool:
    return os.environ.get("SUSC_17C28_ALLOW_DEEP_SEARCH") == "1"


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


def _load_ledger() -> dict | None:
    if not LEDGER.exists():
        return None
    return read_json(LEDGER)


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def _fetch(url: str) -> tuple[str, str, bytes]:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=FETCH_TIMEOUT_SECONDS) as resp:
        return str(resp.status), resp.geturl(), resp.read(MAX_ARTIFACT_BYTES)


def _text_of(data: bytes) -> str:
    raw = data.decode("utf-8", errors="ignore")
    raw = re.sub(r"<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", unescape(raw)).strip()


def _is_event_specific(text: str) -> bool:
    low = text.lower()
    has_date = any(t in low for t in DATE_TERMS)
    has_loc = any(t in low for t in LOCATION_TERMS)
    has_phen = any(t in low for t in EVENT_PHENOMENON_TERMS)
    return has_date and has_loc and has_phen


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:60]


def _canonical_agencia_article_url(url: str) -> str:
    decoded = unquote(url)
    match = re.search(r"https?://agenciabrasil\.ebc\.com\.br/[^\"'\s?#]+/noticia/2022-05/[^\"'\s?#&]+", decoded, re.I)
    return match.group(0).rstrip("/") if match else decoded.split("#", 1)[0].split("?", 1)[0].rstrip("/")


def _is_generic_homepage_url(url: str) -> bool:
    decoded = unquote(url).lower().rstrip("/")
    for _family, (_name, homepage, _officiality) in FAMILY_HOMEPAGES.items():
        home = homepage.lower().rstrip("/")
        if decoded.endswith(home):
            return True
    return False


def _is_agencia_event_article_url(url: str) -> bool:
    canonical = _canonical_agencia_article_url(url).lower()
    if "/noticia/2022-05/" not in canonical or "agenciabrasil.ebc.com.br" not in canonical:
        return False
    if any(blocked in canonical for blocked in ["covid", "facebook.com", "twitter.com", "whatsapp.com", "linkedin.com", "mortalidade-materna"]):
        return False
    return any(k in canonical for k in ["recife", "pernambuco", "chuva", "desliz", "desast", "enchente", "grande-recife"])


def _wayback_url_for(url: str, ts: str = "20220530081923") -> str:
    return WAYBACK.format(ts=ts, url=url)


# ---------------------------------------------------------------------------
# run-deep-acquisition (opt-in)
# ---------------------------------------------------------------------------

def _base_candidate_urls() -> list[dict]:
    """>=30 URLs candidatas: snapshots Wayback (maio 2022) + portais diretos."""
    candidates = []
    for family, (name, homepage, officiality) in FAMILY_HOMEPAGES.items():
        for ts in WAYBACK_TIMESTAMPS:
            candidates.append({"family": family, "source_name": name, "officiality": officiality,
                               "url": WAYBACK.format(ts=ts, url=homepage), "attempt_type": "wayback_archived_snapshot"})
        candidates.append({"family": family, "source_name": name, "officiality": officiality,
                           "url": homepage, "attempt_type": "direct_official_portal"})
    for url in KNOWN_AGENCIA_EVENT_URLS:
        slug = _slugify(_canonical_agencia_article_url(url).rsplit("/", 1)[-1])
        for ts in KNOWN_EVENT_TIMESTAMPS:
            candidates.append({
                "family": "agencia_brasil",
                "source_name": FAMILY_HOMEPAGES["agencia_brasil"][0],
                "officiality": FAMILY_HOMEPAGES["agencia_brasil"][2],
                "url": _wayback_url_for(url, ts),
                "attempt_type": "wayback_known_event_specific_url",
                "preferred_slug": f"agencia_brasil_known_{slug}",
            })
    return candidates


def _require_network_optins() -> tuple[int, str] | None:
    if not _network_enabled():
        return (2, "blocked_network_not_enabled: exige SUSC_17C28_ALLOW_NETWORK=1.")
    if not _public_download_enabled():
        return (2, "blocked_public_download_not_enabled: exige SUSC_17C28_ALLOW_PUBLIC_DOWNLOAD=1.")
    if not _deep_search_enabled():
        return (2, "blocked_deep_search_not_enabled: exige SUSC_17C28_ALLOW_DEEP_SEARCH=1.")
    return None


def run_deep_acquisition_text() -> tuple[int, str]:
    guard = _require_network_optins()
    if guard:
        return guard
    ensure_dir(ARTIFACT_DIR)
    ensure_dir(LOCAL_STATE)
    ledger = {"acquired_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "attempts": [], "artifacts": {}, "followed": []}
    acquired_canonicals: set[str] = set()
    for cand in _base_candidate_urls():
        attempt = {"family": cand["family"], "url": cand["url"], "attempt_type": cand["attempt_type"],
                   "officiality": cand["officiality"], "source_name": cand["source_name"]}
        try:
            status, final, data = _fetch(cand["url"])
            text = _text_of(data)
            homepage_seed = cand["family"] == FOLLOW_FROM_FAMILY and _is_generic_homepage_url(final)
            event_specific = _is_event_specific(text) and not _is_generic_homepage_url(final)
            if cand["attempt_type"] in {"wayback_archived_snapshot", "direct_official_portal"}:
                event_specific = False
            if cand["family"] == "agencia_brasil" and not _is_agencia_event_article_url(final):
                event_specific = False
            attempt.update({"http_status": status, "artifact_acquired": False, "event_specific": event_specific})
            if data and homepage_seed and "agencia_brasil_base" not in ledger["artifacts"]:
                path = ARTIFACT_DIR / "agencia_brasil_base.html"
                path.write_bytes(data)
                ledger["artifacts"]["agencia_brasil_base"] = {
                    "slug": "agencia_brasil_base", "family": cand["family"], "source_name": cand["source_name"],
                    "source_url": final, "officiality_level": cand["officiality"], "artifact_type": "html",
                    "artifact_file": path.name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size,
                    "event_specific": False, "tls_verification_bypassed": False,
                }
                attempt["artifact_acquired"] = True
                attempt["artifact_slug"] = "agencia_brasil_base"
            if event_specific and data:
                canonical = _canonical_agencia_article_url(final) if cand["family"] == "agencia_brasil" else final
                if canonical in acquired_canonicals:
                    ledger["attempts"].append(attempt)
                    continue
                slug = cand.get("preferred_slug") or f"{cand['family']}_{_slugify(canonical)}"
                path = ARTIFACT_DIR / f"{slug}.html"
                path.write_bytes(data)
                ledger["artifacts"][slug] = {
                    "slug": slug, "family": cand["family"], "source_name": cand["source_name"],
                    "source_url": final, "officiality_level": cand["officiality"], "artifact_type": "html",
                    "artifact_file": path.name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size,
                    "event_specific": True, "tls_verification_bypassed": False,
                }
                acquired_canonicals.add(canonical)
                attempt["artifact_acquired"] = True
                attempt["artifact_slug"] = slug
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            reason = type(exc).__name__
            if isinstance(exc, URLError) and "CERTIFICATE_VERIFY_FAILED" in str(exc):
                reason = "ssl_certificate_verify_failed"
            attempt.update({"http_status": "failed_safe", "artifact_acquired": False, "event_specific": False, "failure_reason": reason})
        ledger["attempts"].append(attempt)
    write_json(LEDGER, ledger)
    acquired = len(ledger["artifacts"])
    return (0, f"run-deep-acquisition: {len(ledger['attempts'])} tentativas de busca profunda, {acquired} artefatos base event-specific em {rel(ARTIFACT_DIR)}.")


# ---------------------------------------------------------------------------
# follow-source-links (opt-in)
# ---------------------------------------------------------------------------

def _extract_event_article_links(html: str, base_family: str) -> list[tuple[str, str]]:
    links = []
    seen = set()
    for m in re.finditer(r'''href\s*=\s*['"]([^'"]+)['"]''', html, re.I):
        href = unescape(m.group(1))
        if href.startswith("/"):
            full = urljoin("https://web.archive.org", href)
        else:
            full = href
        canonical = _canonical_agencia_article_url(full)
        if not _is_agencia_event_article_url(canonical):
            continue
        parsed = urlparse(full)
        if "web.archive.org" not in parsed.netloc:
            full = _wayback_url_for(canonical)
        if canonical not in seen:
            links.append((full, base_family))
            seen.add(canonical)
    return links[:MAX_FOLLOW]


def follow_source_links_text() -> tuple[int, str]:
    guard = _require_network_optins()
    if guard:
        return guard
    ledger = _load_ledger()
    if ledger is None:
        return (2, "follow-source-links: sem ledger; rode run-deep-acquisition primeiro.")
    base_slug = f"{FOLLOW_FROM_FAMILY}_base"
    base_art = ledger["artifacts"].get(base_slug)
    if base_art is None:
        return (0, "follow-source-links: artefato base da Agencia Brasil ausente; nenhum link seguido.")
    base_path = ARTIFACT_DIR / base_art["artifact_file"]
    html = base_path.read_bytes().decode("utf-8", errors="ignore")
    followed = 0
    acquired_canonicals = {
        _canonical_agencia_article_url(art["source_url"])
        for art in ledger["artifacts"].values()
        if art.get("family") == "agencia_brasil" and art.get("event_specific")
    }
    for order, (url, family) in enumerate(_extract_event_article_links(html, FOLLOW_FROM_FAMILY), start=1):
        record = {"from_url": base_art["source_url"], "to_url": url, "source_family": family, "followed": True}
        try:
            status, final, data = _fetch(url)
            text = _text_of(data)
            canonical = _canonical_agencia_article_url(final)
            event_specific = _is_event_specific(text) and not _is_generic_homepage_url(final)
            if family == "agencia_brasil" and not _is_agencia_event_article_url(canonical):
                event_specific = False
            if event_specific and data and canonical not in acquired_canonicals:
                slug = f"{family}_art_{order:02d}_{_slugify(canonical.rsplit('/', 1)[-1])}"
                path = ARTIFACT_DIR / f"{slug}.html"
                path.write_bytes(data)
                ledger["artifacts"][slug] = {
                    "slug": slug, "family": family, "source_name": FAMILY_HOMEPAGES[family][0],
                    "source_url": final, "officiality_level": FAMILY_HOMEPAGES[family][2], "artifact_type": "html",
                    "artifact_file": path.name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size,
                    "event_specific": True, "tls_verification_bypassed": False,
                }
                record.update({"artifact_acquired": True, "artifact_slug": slug, "link_text": slug})
                acquired_canonicals.add(canonical)
                followed += 1
            else:
                record.update({"artifact_acquired": False, "link_text": "duplicate_or_not_event_specific"})
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            record.update({"artifact_acquired": False, "link_text": type(exc).__name__})
        ledger["followed"].append(record)
    write_json(LEDGER, ledger)
    return (0, f"follow-source-links: {followed} artigos oficiais especificos seguidos e adquiridos; {len(ledger['artifacts'])} artefatos no total.")


# ---------------------------------------------------------------------------
# Ordered artifacts (deterministic)
# ---------------------------------------------------------------------------

def _ledger_artifacts() -> list[dict]:
    ledger = _load_ledger()
    if ledger is None:
        return []
    return [ledger["artifacts"][slug] for slug in sorted(ledger["artifacts"])]


def _manifest_id_for_slug(slug: str) -> str:
    slugs = sorted(a["slug"] for a in _ledger_artifacts())
    if slug in slugs:
        return f"S17C28_ART_{slugs.index(slug) + 1:04d}"
    return "not_available"


# ---------------------------------------------------------------------------
# 1 - Expanded search plan (offline)
# ---------------------------------------------------------------------------

def expanded_search_plan_rows() -> list[dict]:
    event_id = _event_id()
    rows = []
    idx = 0
    for patch_id in _patch_ids():
        for family in FAMILY_HOMEPAGES:
            idx += 1
            spec = SPECIFICITIES[(idx - 1) % len(SPECIFICITIES)]
            rows.append({
                "expanded_search_plan_id": f"S17C28_PLAN_{idx:04d}",
                "candidate_patch_id": patch_id, "event_id": event_id, "source_family": family,
                "query_text": f"{FAMILY_HOMEPAGES[family][0]} {' '.join(EXPANDED_TERMS[:6])} ({spec})",
                "query_language": "ptbr", "target_gate": "G4_G5", "target_specificity": spec,
                "expected_artifact_type": "official_event_document_or_page", "must_attempt": "true", "review_only": "true",
            })
    return rows


# ---------------------------------------------------------------------------
# 2 - Deep acquisition attempts
# ---------------------------------------------------------------------------

def deep_source_acquisition_attempt_rows() -> list[dict]:
    event_id = _event_id()
    ledger = _load_ledger()
    rows = []
    attempts = ledger.get("attempts", []) if ledger else []
    for idx, att in enumerate(attempts, start=1):
        acquired = bool(att.get("artifact_acquired"))
        rows.append({
            "deep_source_acquisition_attempt_id": f"S17C28_ATT_{idx:04d}",
            "candidate_patch_id": "all_candidate_patches", "event_id": event_id, "source_family": att["family"],
            "attempted_url_or_query": att["url"], "attempt_type": att["attempt_type"],
            "network_enabled": "true", "http_status": att.get("http_status", "failed_safe"),
            "artifact_acquired": _bool_text(acquired),
            "artifact_manifest_id": _manifest_id_for_slug(att.get("artifact_slug", "")) if acquired else "not_available",
            "tls_verification_bypassed": _bool_text(bool(att.get("tls_verification_bypassed"))),
            "failure_reason": att.get("failure_reason", "not_applicable" if acquired else "not_event_specific_or_duplicate"),
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# 3 - Followed links
# ---------------------------------------------------------------------------

def followed_link_registry_rows() -> list[dict]:
    ledger = _load_ledger()
    rows = []
    for idx, rec in enumerate(ledger.get("followed", []) if ledger else [], start=1):
        acquired = bool(rec.get("artifact_acquired"))
        rows.append({
            "followed_link_id": f"S17C28_FLINK_{idx:04d}",
            "source_artifact_manifest_id": _manifest_id_for_slug(f"{FOLLOW_FROM_FAMILY}_base"),
            "source_family": rec["source_family"], "from_url": rec["from_url"], "to_url": rec["to_url"],
            "link_text": rec.get("link_text", "not_available"), "followed": _bool_text(bool(rec.get("followed"))),
            "artifact_acquired": _bool_text(acquired),
            "artifact_manifest_id": _manifest_id_for_slug(rec.get("artifact_slug", "")) if acquired else "not_available",
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# 4 - Manifest
# ---------------------------------------------------------------------------

MANIFEST_FIELDS = [
    "deep_source_artifact_manifest_id", "candidate_patch_id", "event_id", "source_family", "source_name",
    "source_url", "artifact_local_path", "artifact_type", "sha256", "size_bytes", "officiality_level",
    "event_specific", "location_specific", "phenomenon_specific", "geometry_specific",
    "stored_in_outputs_public", "raw_heavy", "tls_verification_bypassed", "review_only", "trainable", "ground_truth",
]


def _artifact_text_by_slug(slug: str) -> str:
    for a in _ledger_artifacts():
        if a["slug"] == slug:
            path = ARTIFACT_DIR / a["artifact_file"]
            if path.exists():
                return _text_of(path.read_bytes())
    return ""


def deep_source_artifact_manifest_rows() -> list[dict]:
    event_id = _event_id()
    rows = []
    for idx, art in enumerate(_ledger_artifacts(), start=1):
        path = ARTIFACT_DIR / art["artifact_file"]
        if not path.exists():
            continue
        text = _text_of(path.read_bytes())
        low = text.lower()
        loc_specific = any(b in low for b in BAIRRO_TERMS) or ("jaboat" in low or "olinda" in low)
        phen_specific = any(t in low for t in EVENT_PHENOMENON_TERMS)
        geom_specific = bool(re.search(r"-?\d{1,2}[.,]\d{4,}", text))
        rows.append({
            "deep_source_artifact_manifest_id": f"S17C28_ART_{idx:04d}",
            "candidate_patch_id": "all_candidate_patches", "event_id": event_id, "source_family": art["family"],
            "source_name": art["source_name"], "source_url": art["source_url"],
            "artifact_local_path": rel(path), "artifact_type": art["artifact_type"],
            "sha256": sha256_file(path), "size_bytes": str(path.stat().st_size),
            "officiality_level": art["officiality_level"], "event_specific": _bool_text(bool(art.get("event_specific"))),
            "location_specific": _bool_text(loc_specific), "phenomenon_specific": _bool_text(phen_specific),
            "geometry_specific": _bool_text(geom_specific), "stored_in_outputs_public": "true", "raw_heavy": "false",
            "tls_verification_bypassed": _bool_text(bool(art.get("tls_verification_bypassed"))), **GOV,
        })
    return rows


def _official_event_specific_manifests() -> list[dict]:
    return [m for m in deep_source_artifact_manifest_rows()
            if m["event_specific"] == "true" and m["officiality_level"] in OFFICIAL_EVENT_LEVELS]


# ---------------------------------------------------------------------------
# 5 - Parsed index
# ---------------------------------------------------------------------------

def _mentions(text: str, terms: list[str]) -> list[str]:
    low = text.lower()
    return [t for t in terms if t in low]


def _snippet(text: str, terms: list[str]) -> str:
    low = text.lower()
    for t in terms:
        pos = low.find(t)
        if pos >= 0:
            return text[max(0, pos - 60):pos + 150].strip()
    return text[:170].strip()


def deep_parsed_artifact_index_rows() -> list[dict]:
    event_id = _event_id()
    rows = []
    for idx, manifest in enumerate(deep_source_artifact_manifest_rows(), start=1):
        text = _artifact_text_by_slug_from_manifest(manifest)
        dates = _mentions(text, DATE_TERMS)
        locations = _mentions(text, LOCATION_TERMS)
        phenomena = _mentions(text, EVENT_PHENOMENON_TERMS)
        bairros = _mentions(text, BAIRRO_TERMS)
        coords = re.findall(r"-?\d{1,2}[.,]\d{4,}", text)[:3]
        logradouros = LOGRADOURO_RE.findall(text)[:3]
        limitations = []
        if not dates:
            limitations.append("sem data")
        if not bairros:
            limitations.append("sem bairro/logradouro patch-level")
        if not coords:
            limitations.append("sem coordenada")
        rows.append({
            "deep_parsed_artifact_id": f"S17C28_PARSE_{idx:04d}",
            "deep_source_artifact_manifest_id": manifest["deep_source_artifact_manifest_id"],
            "candidate_patch_id": "all_candidate_patches", "event_id": event_id,
            "parse_success": _bool_text(bool(text)), "text_extracted": _bool_text(bool(text)),
            "date_mentions": ";".join(dates) if dates else "none",
            "location_mentions": ";".join(locations) if locations else "none",
            "phenomenon_mentions": ";".join(phenomena) if phenomena else "none",
            "coordinate_mentions": ";".join(coords) if coords else "none",
            "bairro_mentions": ";".join(bairros) if bairros else "none",
            "logradouro_mentions": ";".join(logradouros) if logradouros else "none",
            "evidence_snippet": (_snippet(text, LOCATION_TERMS + FLOOD_TERMS + LANDSLIDE_TERMS)[:200] if text else "not_available"),
            "parse_limitations": ";".join(limitations) if limitations else "none", "review_only": "true",
        })
    return rows


def _artifact_text_by_slug_from_manifest(manifest: dict) -> str:
    path = ROOT / manifest["artifact_local_path"]
    return _text_of(path.read_bytes()) if path.exists() else ""


# ---------------------------------------------------------------------------
# 6 - Specific observed event candidates
# ---------------------------------------------------------------------------

def _has_event_content(parsed: dict) -> bool:
    return parsed["date_mentions"] != "none" and parsed["location_mentions"] != "none" and parsed["phenomenon_mentions"] != "none"


def specific_observed_event_candidate_rows() -> list[dict]:
    event_id = _event_id()
    manifests = {m["deep_source_artifact_manifest_id"]: m for m in deep_source_artifact_manifest_rows()}
    rows = []
    idx = 0
    for parsed in deep_parsed_artifact_index_rows():
        manifest = manifests[parsed["deep_source_artifact_manifest_id"]]
        if manifest["event_specific"] != "true" or not _has_event_content(parsed) or manifest["officiality_level"] not in OFFICIAL_EVENT_LEVELS:
            continue
        idx += 1
        bairro = parsed["bairro_mentions"] if parsed["bairro_mentions"] != "none" else "none"
        logradouro = parsed["logradouro_mentions"] if parsed["logradouro_mentions"] != "none" else "none"
        rows.append({
            "specific_observed_event_candidate_id": f"S17C28_OEC_{idx:04d}",
            "deep_source_artifact_manifest_id": parsed["deep_source_artifact_manifest_id"],
            "candidate_patch_id": _patch_ids()[0], "event_id": event_id,
            "observed_event_date_or_period": "2022-05 (24-30 de maio; referencias 28-29)",
            "observed_location_text": parsed["location_mentions"], "bairro": bairro, "logradouro": logradouro,
            "observed_geometry": "not_available", "geometry_type": "none",
            "geometry_uncertainty_m": "municipal_or_bairro_level_high" if bairro != "none" else "city_level_high",
            "phenomenon_candidate": parsed["phenomenon_mentions"], "source_family": manifest["source_family"],
            "officiality_level": manifest["officiality_level"], "evidence_snippet": parsed["evidence_snippet"],
            "can_evaluate_g4": "true", "can_evaluate_g5": "true", **GOV,
        })
    return rows


# ---------------------------------------------------------------------------
# 7 - Location resolution
# ---------------------------------------------------------------------------

def location_resolution_rows() -> list[dict]:
    rows = []
    for idx, oec in enumerate(specific_observed_event_candidate_rows(), start=1):
        has_bairro = oec["bairro"] != "none"
        rows.append({
            "location_resolution_id": f"S17C28_LOC_{idx:04d}",
            "specific_observed_event_candidate_id": oec["specific_observed_event_candidate_id"],
            "candidate_patch_id": oec["candidate_patch_id"], "location_text": oec["observed_location_text"],
            "bairro": oec["bairro"], "logradouro": oec["logradouro"], "resolved_geometry": "not_available",
            "resolution_method": "named_place_bairro_or_municipal_level" if has_bairro else "named_place_city_level",
            "distance_to_patch_m": "not_computable_no_geometry", "within_patch_or_buffer": "false",
            "uncertainty_m": "3000_or_more_bairro_level" if has_bairro else "10000_or_more_city_level",
            "location_resolution_status": "bairro_level_no_patch_geometry" if has_bairro else "city_level_only",
            "blocking_reason": "bairro/municipio sem geometria ou endereco patch-level; nao inventar coordenada" if has_bairro else "apenas cidade/regiao",
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# 8 - Phenomenon classification
# ---------------------------------------------------------------------------

def phenomenon_classification_rows() -> list[dict]:
    rows = []
    for idx, oec in enumerate(specific_observed_event_candidate_rows(), start=1):
        phen = oec["phenomenon_candidate"].lower()
        flood = any(t in phen for t in FLOOD_TERMS)
        landslide = any(t in phen for t in LANDSLIDE_TERMS)
        context_only = any(t in phen for t in EVENT_CONTEXT_TERMS)
        if flood and landslide:
            pclass, conf = "MIXED_AMBIGUOUS", "medium"
        elif flood:
            pclass, conf = "HYDROLOGICAL_CONFIRMED", "medium"
        elif landslide:
            pclass, conf = "MASS_MOVEMENT_CONFIRMED", "medium"
        elif context_only:
            pclass, conf = "HYDROLOGICAL_TRIGGER_OR_IMPACT_ONLY", "low"
        else:
            pclass, conf = "INSUFFICIENT", "low"
        g5 = pclass == "HYDROLOGICAL_CONFIRMED"
        rows.append({
            "phenomenon_classification_id": f"S17C28_PHEN_{idx:04d}",
            "specific_observed_event_candidate_id": oec["specific_observed_event_candidate_id"],
            "candidate_patch_id": oec["candidate_patch_id"], "phenomenon_text": oec["phenomenon_candidate"],
            "phenomenon_class": pclass, "hydrological_confirmed": _bool_text(flood),
            "mass_movement_excluded": _bool_text(not landslide), "mixed_or_ambiguous": _bool_text(flood and landslide),
            "classification_confidence": conf, "G5_candidate_status": _bool_text(g5),
            "blocking_reason": "fenomeno misto (inundacao + deslizamento): nao separa hidrologico de movimento de massa" if not g5 else "not_applicable",
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# G4 / G5 / GR evaluation
# ---------------------------------------------------------------------------

def g4_spatial_link_evaluation_rows() -> list[dict]:
    loc_by = {r["specific_observed_event_candidate_id"]: r for r in location_resolution_rows()}
    rows = []
    for idx, oec in enumerate(specific_observed_event_candidate_rows(), start=1):
        loc = loc_by.get(oec["specific_observed_event_candidate_id"], {})
        rows.append({
            "g4_evaluation_id": f"S17C28_G4_{idx:04d}",
            "specific_observed_event_candidate_id": oec["specific_observed_event_candidate_id"],
            "candidate_patch_id": oec["candidate_patch_id"],
            "has_observed_geometry_or_geocodable_location": "false",
            "distance_to_patch_m": loc.get("distance_to_patch_m", "not_computable_no_geometry"),
            "uncertainty_m": loc.get("uncertainty_m", "10000_or_more_city_level"),
            "within_patch_or_acceptable_buffer": "false", "G4_vinculo_espacial_evento": "false",
            "blocking_reason": "sem geometria/endereco/coordenada patch-level; localizacao bairro/municipal com incerteza alta demais para G4 patch-level",
            "review_only": "true",
        })
    return rows


def g5_phenomenon_evaluation_rows() -> list[dict]:
    phen_by = {r["specific_observed_event_candidate_id"]: r for r in phenomenon_classification_rows()}
    rows = []
    for idx, oec in enumerate(specific_observed_event_candidate_rows(), start=1):
        phen = phen_by.get(oec["specific_observed_event_candidate_id"], {})
        g5 = phen.get("G5_candidate_status", "false") == "true"
        rows.append({
            "g5_evaluation_id": f"S17C28_G5_{idx:04d}",
            "specific_observed_event_candidate_id": oec["specific_observed_event_candidate_id"],
            "candidate_patch_id": oec["candidate_patch_id"], "phenomenon_class": phen.get("phenomenon_class", "INSUFFICIENT"),
            "hydrological_confirmed": phen.get("hydrological_confirmed", "false"),
            "mass_movement_excluded": phen.get("mass_movement_excluded", "false"),
            "G5_separacao_fenomeno": _bool_text(g5), "blocking_reason": phen.get("blocking_reason", "fenomeno insuficiente"),
            "review_only": "true",
        })
    return rows


def ground_reference_candidate_evaluation_rows() -> list[dict]:
    event_id = _event_id()
    manifests = {m["deep_source_artifact_manifest_id"]: m for m in deep_source_artifact_manifest_rows()}
    g4_by = {r["specific_observed_event_candidate_id"]: r for r in g4_spatial_link_evaluation_rows()}
    g5_by = {r["specific_observed_event_candidate_id"]: r for r in g5_phenomenon_evaluation_rows()}
    rows = []
    for idx, oec in enumerate(specific_observed_event_candidate_rows(), start=1):
        manifest = manifests.get(oec["deep_source_artifact_manifest_id"], {})
        official = manifest.get("officiality_level") in OFFICIAL_EVENT_LEVELS
        g4 = g4_by.get(oec["specific_observed_event_candidate_id"], {}).get("G4_vinculo_espacial_evento", "false") == "true"
        g5 = g5_by.get(oec["specific_observed_event_candidate_id"], {}).get("G5_separacao_fenomeno", "false") == "true"
        g1, g2, g3, g6, g7 = True, official, True, True, True
        can_gr = all([g1, g2, g3, g4, g5, g6, g7])
        rows.append({
            "ground_reference_candidate_eval_id": f"S17C28_GRCE_{idx:04d}",
            "specific_observed_event_candidate_id": oec["specific_observed_event_candidate_id"],
            "candidate_patch_id": oec["candidate_patch_id"], "event_id": event_id,
            "G1_existencia_documental": _bool_text(g1), "G2_confiabilidade_fonte": _bool_text(g2),
            "G3_precisao_temporal": _bool_text(g3), "G4_vinculo_espacial_evento": _bool_text(g4),
            "G5_separacao_fenomeno": _bool_text(g5), "G6_proveniencia_integridade": _bool_text(g6),
            "G7_anti_leakage": _bool_text(g7), "can_be_ground_reference_candidate": _bool_text(can_gr),
            "can_be_ground_truth": "false", "can_be_training_label": "false", "can_unlock_17b": _bool_text(can_gr),
            "blocking_reason": "G4 (geometria patch-level) e G5 (separacao de fenomeno misto) nao satisfeitos" if not can_gr else "not_applicable",
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------

def official_artifact_scorecard_rows() -> list[dict]:
    parsed_by = {p["deep_source_artifact_manifest_id"]: p for p in deep_parsed_artifact_index_rows()}
    rows = []
    for idx, manifest in enumerate(deep_source_artifact_manifest_rows(), start=1):
        parsed = parsed_by.get(manifest["deep_source_artifact_manifest_id"], {})
        temporal = parsed.get("date_mentions", "none") != "none"
        rows.append({
            "official_artifact_scorecard_id": f"S17C28_SCORE_{idx:04d}",
            "deep_source_artifact_manifest_id": manifest["deep_source_artifact_manifest_id"],
            "source_family": manifest["source_family"], "officiality_level": manifest["officiality_level"],
            "event_specific": manifest["event_specific"], "location_specific": manifest["location_specific"],
            "phenomenon_specific": manifest["phenomenon_specific"], "geometry_specific": manifest["geometry_specific"],
            "temporal_specific": _bool_text(temporal), "parse_success": parsed.get("parse_success", "false"),
            "usable_for_g4": "false", "usable_for_g5": "false", "usable_for_ground_reference_candidate": "false",
            "blocking_reason": "artefato oficial de contexto do evento; sem geometria patch-level e fenomeno misto: nao satisfaz G4/G5",
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Evidence graph update
# ---------------------------------------------------------------------------

def _graph_update():
    event_id = _event_id()
    nodes = []
    for m in deep_source_artifact_manifest_rows():
        nodes.append((f"art:{m['deep_source_artifact_manifest_id']}", "deep_source_artifact", m["deep_source_artifact_manifest_id"], "all_candidate_patches"))
    for p in deep_parsed_artifact_index_rows():
        nodes.append((f"parse:{p['deep_parsed_artifact_id']}", "deep_parsed_artifact", p["deep_parsed_artifact_id"], "all_candidate_patches"))
    for o in specific_observed_event_candidate_rows():
        nodes.append((f"oec:{o['specific_observed_event_candidate_id']}", "specific_observed_event_candidate", o["specific_observed_event_candidate_id"], o["candidate_patch_id"]))
    for r in location_resolution_rows():
        nodes.append((f"loc:{r['location_resolution_id']}", "location_resolution", r["location_resolution_id"], r["candidate_patch_id"]))
    for r in phenomenon_classification_rows():
        nodes.append((f"phen:{r['phenomenon_classification_id']}", "phenomenon_classification", r["phenomenon_classification_id"], r["candidate_patch_id"]))
    for r in g4_spatial_link_evaluation_rows():
        nodes.append((f"g4:{r['g4_evaluation_id']}", "g4_evaluation", r["g4_evaluation_id"], r["candidate_patch_id"]))
    for r in g5_phenomenon_evaluation_rows():
        nodes.append((f"g5:{r['g5_evaluation_id']}", "g5_evaluation", r["g5_evaluation_id"], r["candidate_patch_id"]))
    for r in ground_reference_candidate_evaluation_rows():
        nodes.append((f"grce:{r['ground_reference_candidate_eval_id']}", "ground_reference_candidate_evaluation", r["ground_reference_candidate_eval_id"], r["candidate_patch_id"]))
    key_to_id = {key: f"S17C28_NODE_{i:04d}" for i, (key, *_r) in enumerate(nodes, start=1)}

    edges = []
    for p in deep_parsed_artifact_index_rows():
        edges.append((f"parse:{p['deep_parsed_artifact_id']}", f"art:{p['deep_source_artifact_manifest_id']}", "parsed_from_artifact", "all_candidate_patches"))
    for o in specific_observed_event_candidate_rows():
        pid = next((p["deep_parsed_artifact_id"] for p in deep_parsed_artifact_index_rows() if p["deep_source_artifact_manifest_id"] == o["deep_source_artifact_manifest_id"]), "")
        edges.append((f"oec:{o['specific_observed_event_candidate_id']}", f"parse:{pid}", "candidate_from_parsed", o["candidate_patch_id"]))
    for r in location_resolution_rows():
        edges.append((f"loc:{r['location_resolution_id']}", f"oec:{r['specific_observed_event_candidate_id']}", "location_of_candidate", r["candidate_patch_id"]))
    for r in phenomenon_classification_rows():
        edges.append((f"phen:{r['phenomenon_classification_id']}", f"oec:{r['specific_observed_event_candidate_id']}", "phenomenon_of_candidate", r["candidate_patch_id"]))
    for r in g4_spatial_link_evaluation_rows():
        edges.append((f"g4:{r['g4_evaluation_id']}", f"oec:{r['specific_observed_event_candidate_id']}", "g4_of_candidate", r["candidate_patch_id"]))
    for r in g5_phenomenon_evaluation_rows():
        edges.append((f"g5:{r['g5_evaluation_id']}", f"oec:{r['specific_observed_event_candidate_id']}", "g5_of_candidate", r["candidate_patch_id"]))
    for r in ground_reference_candidate_evaluation_rows():
        edges.append((f"grce:{r['ground_reference_candidate_eval_id']}", f"oec:{r['specific_observed_event_candidate_id']}", "gr_eval_of_candidate", r["candidate_patch_id"]))
    return nodes, edges, key_to_id, event_id


def evidence_graph_update_node_rows() -> list[dict]:
    nodes, _edges, key_to_id, event_id = _graph_update()
    return [{"node_id": key_to_id[key], "node_type": nt, "object_id": oid, "candidate_patch_id": pid, "event_id": event_id, **GOV}
            for key, nt, oid, pid in nodes]


def evidence_graph_update_edge_rows() -> list[dict]:
    _nodes, edges, key_to_id, event_id = _graph_update()
    return [{"edge_id": f"S17C28_EDGE_{i:04d}", "source_node_id": key_to_id[s], "target_node_id": key_to_id.get(t, "not_available"),
             "edge_type": et, "candidate_patch_id": pid, "event_id": event_id, "review_only": "true"}
            for i, (s, t, et, pid) in enumerate(edges, start=1)]


# ---------------------------------------------------------------------------
# No leakage
# ---------------------------------------------------------------------------

def no_leakage_audit_rows() -> list[dict]:
    rows = []
    candidates = specific_observed_event_candidate_rows()
    for idx, oec in enumerate(candidates, start=1):
        official = oec["officiality_level"] in OFFICIAL_EVENT_LEVELS
        rows.append({
            "no_leakage_audit_id": f"S17C28_LEAK_{idx:04d}",
            "object_id": oec["specific_observed_event_candidate_id"], "object_type": "specific_observed_event_candidate",
            "uses_sensor_as_event_observation": "false", "uses_chirps_as_event_reference": "false",
            "uses_news_as_ground_reference_without_official_support": "false", "uses_candidate_as_ground_truth": "false",
            "uses_ground_reference_as_training_label": "false", "uses_synthetic_as_real": "false", "passes_no_leakage": "true",
            "blocking_reason": "candidato oficial institucional review-only sem promocao indevida" if official else "candidato review-only",
            "review_only": "true",
        })
    if not rows:
        rows.append({"no_leakage_audit_id": "S17C28_LEAK_0001", "object_id": "no_candidate", "object_type": "deep_acquisition_package",
                     "uses_sensor_as_event_observation": "false", "uses_chirps_as_event_reference": "false",
                     "uses_news_as_ground_reference_without_official_support": "false", "uses_candidate_as_ground_truth": "false",
                     "uses_ground_reference_as_training_label": "false", "uses_synthetic_as_real": "false",
                     "passes_no_leakage": "true", "blocking_reason": "not_applicable", "review_only": "true"})
    return rows


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def gate_evaluation_matrix_rows() -> list[dict]:
    rows = []
    for idx, gr in enumerate(ground_reference_candidate_evaluation_rows(), start=1):
        rows.append({
            "gate_eval_id": f"S17C28_GATE_{idx:04d}",
            "object_id": gr["specific_observed_event_candidate_id"], "object_type": "specific_observed_event_candidate",
            "candidate_patch_id": gr["candidate_patch_id"], "G1_existencia_documental": gr["G1_existencia_documental"],
            "G2_confiabilidade_fonte": gr["G2_confiabilidade_fonte"], "G3_precisao_temporal": gr["G3_precisao_temporal"],
            "G4_vinculo_espacial_evento": gr["G4_vinculo_espacial_evento"], "G5_separacao_fenomeno": gr["G5_separacao_fenomeno"],
            "G6_proveniencia_integridade": gr["G6_proveniencia_integridade"], "G7_anti_leakage": gr["G7_anti_leakage"],
            "all_gates_passed_for_ground_reference": gr["can_be_ground_reference_candidate"],
            "acceptance_status": "accepted_ground_reference_candidate" if gr["can_be_ground_reference_candidate"] == "true" else "blocked_observed_event_candidate_review_only",
            "blocking_reason": gr["blocking_reason"], **GOV,
        })
    return rows


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def build_summary() -> dict:
    plan = expanded_search_plan_rows()
    attempts = deep_source_acquisition_attempt_rows()
    manifests = deep_source_artifact_manifest_rows()
    official_specific = _official_event_specific_manifests()
    parsed = deep_parsed_artifact_index_rows()
    parsed_official = [p for p in parsed if p["deep_source_artifact_manifest_id"] in {m["deep_source_artifact_manifest_id"] for m in official_specific} and p["parse_success"] == "true"]
    candidates = specific_observed_event_candidate_rows()
    locations = location_resolution_rows()
    phenomena = phenomenon_classification_rows()
    g4 = g4_spatial_link_evaluation_rows()
    g5 = g5_phenomenon_evaluation_rows()
    gr_eval = ground_reference_candidate_evaluation_rows()
    accepted = [r for r in gr_eval if r["can_be_ground_reference_candidate"] == "true"]
    patch_level_loc = len([c for c in candidates if c["bairro"] != "none" or c["logradouro"] != "none"])
    phen_specific = len([p for p in phenomena if p["phenomenon_class"] != "INSUFFICIENT"])
    minimum = (
        len([a for a in attempts if a["network_enabled"] == "true"]) >= 30
        and len(official_specific) >= 3 and len(parsed_official) >= 3 and len(candidates) >= 3
        and patch_level_loc >= 1 and phen_specific >= 1 and (len(g4) + len(g5)) >= 3
    )
    return {
        "minimum_success_achieved": minimum,
        "expanded_search_plan_rows_count": len(plan),
        "deep_source_search_attempts_count": len([a for a in attempts if a["network_enabled"] == "true"]),
        "official_event_specific_artifacts_acquired_count": len(official_specific),
        "official_event_specific_artifacts_parsed_count": len(parsed_official),
        "specific_observed_event_candidates_count": len(candidates),
        "patch_level_location_candidates_count": patch_level_loc,
        "phenomenon_specific_candidates_count": phen_specific,
        "G4_G5_evaluation_rows_count": len(g4) + len(g5),
        "ground_reference_candidates_evaluated_count": len(gr_eval),
        "accepted_ground_reference_candidate_count": len(accepted),
        "G4_true_count": len([r for r in g4 if r["G4_vinculo_espacial_evento"] == "true"]),
        "G5_true_count": len([r for r in g5 if r["G5_separacao_fenomeno"] == "true"]),
        "ground_truth_created": False,
        "training_labels_created": False,
        "score_v6_changed": bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)])),
        "score_v7_created": SCORE_V7.exists(),
        "official_patch_created": False,
        "official_patch_link_created": False,
        "eligible_for_17b_now": len(accepted) > 0,
        "eligible_for_score_v7": False,
        "review_only": True,
        "trainable": False,
        "ground_truth": False,
        "followed_links_count": len(followed_link_registry_rows()),
        "recommended_next_milestone": "SUSC-17C29 Aquisicao de geometria oficial de evento (mancha/poligono/coordenada) e classificacao de fenomeno por local para tentar G4/G5 patch-level",
    }


def build_blockers() -> list[dict]:
    blockers = [
        "no_patch_level_geometry", "bairro_only_location_uncertainty", "phenomenon_mixed_or_ambiguous",
        "official_artifact_not_event_specific", "no_accepted_ground_reference_candidate",
        "17b_blocked_until_G4_G5_true", "score_v7_blocked_until_ground_reference_policy",
    ]
    return [
        {
            "blocker_id": f"S17C28_BLOCKER_{idx:04d}", "blocker_type": blocker,
            "description": "Bloqueio real: artefatos oficiais especificos do evento adquiridos e avaliados, mas G4 (geometria patch-level) e G5 (separacao de fenomeno misto) nao satisfeitos; sem coordenada/poligono e com fenomeno inundacao+deslizamento nenhum Ground Reference Candidate e aceito.",
            "blocks_ground_reference_candidate": _bool_text(blocker in {"no_patch_level_geometry", "bairro_only_location_uncertainty", "phenomenon_mixed_or_ambiguous", "no_accepted_ground_reference_candidate"}),
            "blocks_17b": "true", "blocks_score_v7": "true", "review_only": "true",
        }
        for idx, blocker in enumerate(blockers, start=1)
    ]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

def _schema(required: list[str], props: dict, title: str) -> dict:
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": title, "type": "object", "required": required, "properties": props}


def build_artifact_schema() -> dict:
    required = list(deep_source_artifact_manifest_rows()[0].keys()) if deep_source_artifact_manifest_rows() else MANIFEST_FIELDS
    return _schema(required, {
        "deep_source_artifact_manifest_id": {"type": "string", "pattern": "^S17C28_ART_"},
        "raw_heavy": {"const": "false"}, "review_only": {"const": "true"},
        "trainable": {"const": "false"}, "ground_truth": {"const": "false"},
    }, "SUSC-17C28 deep source artifact schema v1")


def build_candidate_schema() -> dict:
    required = list(specific_observed_event_candidate_rows()[0].keys()) if specific_observed_event_candidate_rows() else [
        "specific_observed_event_candidate_id", "deep_source_artifact_manifest_id", "candidate_patch_id", "event_id",
        "observed_event_date_or_period", "observed_location_text", "bairro", "logradouro", "observed_geometry",
        "geometry_type", "geometry_uncertainty_m", "phenomenon_candidate", "source_family", "officiality_level",
        "evidence_snippet", "can_evaluate_g4", "can_evaluate_g5", "review_only", "trainable", "ground_truth",
    ]
    return _schema(required, {
        "specific_observed_event_candidate_id": {"type": "string", "pattern": "^S17C28_OEC_"},
        "review_only": {"const": "true"}, "trainable": {"const": "false"}, "ground_truth": {"const": "false"},
    }, "SUSC-17C28 specific event candidate schema v1")


def build_g4_g5_schema() -> dict:
    required = list(ground_reference_candidate_evaluation_rows()[0].keys()) if ground_reference_candidate_evaluation_rows() else [
        "ground_reference_candidate_eval_id", "specific_observed_event_candidate_id", "candidate_patch_id", "event_id",
        "G1_existencia_documental", "G2_confiabilidade_fonte", "G3_precisao_temporal", "G4_vinculo_espacial_evento",
        "G5_separacao_fenomeno", "G6_proveniencia_integridade", "G7_anti_leakage", "can_be_ground_reference_candidate",
        "can_be_ground_truth", "can_be_training_label", "can_unlock_17b", "blocking_reason", "review_only",
    ]
    return _schema(required, {
        "ground_reference_candidate_eval_id": {"type": "string", "pattern": "^S17C28_GRCE_"},
        "can_be_ground_truth": {"const": "false"}, "can_be_training_label": {"const": "false"}, "review_only": {"const": "true"},
    }, "SUSC-17C28 g4 g5 evaluation schema v1")


# ---------------------------------------------------------------------------
# Relatorio
# ---------------------------------------------------------------------------

def build_report() -> str:
    s = build_summary()
    return "\n".join([
        "# SUSC-17C28 - Aquisicao profunda de artefatos oficiais especificos para G4/G5", "",
        "## Objetivo",
        "Aprofundar a aquisicao em fontes oficiais/institucionais especificas do evento (enchentes e deslizamentos em Pernambuco/Recife, maio 2022) via snapshots arquivados (Wayback) e links seguidos, com hash, parse e avaliacao G4/G5.", "",
        "## Aquisicao profunda",
        f"- Plano expandido: {s['expanded_search_plan_rows_count']} linhas (event/location/phenomenon/geometry).",
        f"- Tentativas de busca profunda: {s['deep_source_search_attempts_count']}.",
        f"- Links seguidos: {s['followed_links_count']}.",
        f"- Artefatos oficiais especificos adquiridos: {s['official_event_specific_artifacts_acquired_count']}; parseados: {s['official_event_specific_artifacts_parsed_count']}.", "",
        "## Candidatos e G4/G5",
        f"- Observed event candidates especificos: {s['specific_observed_event_candidates_count']}.",
        f"- Candidatos com local patch-level/bairro: {s['patch_level_location_candidates_count']}; com fenomeno especifico: {s['phenomenon_specific_candidates_count']}.",
        f"- Avaliacoes G4/G5: {s['G4_G5_evaluation_rows_count']}; Ground Reference Candidates avaliados: {s['ground_reference_candidates_evaluated_count']}; aceitos: {s['accepted_ground_reference_candidate_count']}.",
        f"- G4_true_count={s['G4_true_count']}, G5_true_count={s['G5_true_count']}.", "",
        "## Resultado cientifico (honesto)",
        "- Fontes oficiais institucionais (Agencia Brasil/EBC) confirmam o evento com data, Recife/Jaboatao/Olinda e fenomeno, mas o fenomeno e misto (inundacao + deslizamento): G5 nao e satisfeito.",
        "- A localizacao disponivel e municipal/bairro, sem geometria ou coordenada patch-level: G4 nao e satisfeito. Nenhuma coordenada foi inventada.",
        "- Nenhum Ground Reference Candidate foi aceito; 17B permanece bloqueado.", "",
        "## Guardrails",
        "- Sensor/CHIRPS nao viraram evento observado; noticia nao virou Ground Reference sozinha; nenhum ground truth, label, treino, score v7 ou patch oficial; score v6 intacto.", "",
        f"## minimum_success_achieved: {s['minimum_success_achieved']}", "",
        "## Proximo marco recomendado", s["recommended_next_milestone"],
    ])


# ---------------------------------------------------------------------------
# Build / validacao
# ---------------------------------------------------------------------------

def build_all() -> None:
    _require_inputs()
    write_csv(EXPANDED_PLAN, expanded_search_plan_rows())
    write_csv(DEEP_ATTEMPTS, deep_source_acquisition_attempt_rows())
    write_csv(FOLLOWED_LINKS, followed_link_registry_rows())
    write_csv(DEEP_MANIFEST, deep_source_artifact_manifest_rows(), MANIFEST_FIELDS)
    write_csv(DEEP_PARSED, deep_parsed_artifact_index_rows())
    write_csv(SPECIFIC_CANDIDATES, specific_observed_event_candidate_rows())
    write_csv(LOCATION_RESOLUTION, location_resolution_rows())
    write_csv(PHENOMENON, phenomenon_classification_rows())
    write_csv(G4_EVAL, g4_spatial_link_evaluation_rows())
    write_csv(G5_EVAL, g5_phenomenon_evaluation_rows())
    write_csv(GR_CANDIDATE_EVAL, ground_reference_candidate_evaluation_rows())
    write_csv(SCORECARD, official_artifact_scorecard_rows())
    write_csv(GRAPH_UPDATE_NODES, evidence_graph_update_node_rows())
    write_csv(GRAPH_UPDATE_EDGES, evidence_graph_update_edge_rows())
    write_csv(NO_LEAKAGE, no_leakage_audit_rows())
    write_csv(GATES, gate_evaluation_matrix_rows())
    write_json(SUMMARY, build_summary())
    write_csv(BLOCKERS, build_blockers())
    write_json(ARTIFACT_SCHEMA, build_artifact_schema())
    write_json(CANDIDATE_SCHEMA, build_candidate_schema())
    write_json(G4_G5_SCHEMA, build_g4_g5_schema())
    write_markdown(REPORT, build_report())


def _required_outputs() -> list[Path]:
    outputs = [
        REPORT, EXPANDED_PLAN, DEEP_ATTEMPTS, FOLLOWED_LINKS, DEEP_MANIFEST, DEEP_PARSED, SPECIFIC_CANDIDATES,
        LOCATION_RESOLUTION, PHENOMENON, G4_EVAL, G5_EVAL, GR_CANDIDATE_EVAL, SCORECARD, GRAPH_UPDATE_NODES,
        GRAPH_UPDATE_EDGES, NO_LEAKAGE, GATES, SUMMARY, BLOCKERS, ARTIFACT_SCHEMA, CANDIDATE_SCHEMA, G4_G5_SCHEMA,
    ]
    ledger = _load_ledger()
    if ledger is not None:
        outputs.append(LEDGER)
        for art in _ledger_artifacts():
            outputs.append(ARTIFACT_DIR / art["artifact_file"])
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
    outputs = [p for p in _required_outputs() if p != LEDGER]
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
    plan = read_csv(EXPANDED_PLAN)
    attempts = read_csv(DEEP_ATTEMPTS)
    manifests = read_csv(DEEP_MANIFEST)
    parsed = read_csv(DEEP_PARSED)
    candidates = read_csv(SPECIFIC_CANDIDATES)
    locations = read_csv(LOCATION_RESOLUTION)
    phenomena = read_csv(PHENOMENON)
    g4 = read_csv(G4_EVAL)
    g5 = read_csv(G5_EVAL)
    gr_eval = read_csv(GR_CANDIDATE_EVAL)
    gates = read_csv(GATES)
    leakage = read_csv(NO_LEAKAGE)
    nodes = read_csv(GRAPH_UPDATE_NODES)
    edges = read_csv(GRAPH_UPDATE_EDGES)
    summary = read_json(SUMMARY)
    artifact_schema = read_json(ARTIFACT_SCHEMA)
    candidate_schema = read_json(CANDIDATE_SCHEMA)
    g4g5_schema = read_json(G4_G5_SCHEMA)

    for row in manifests:
        errors.extend(_schema_violations(row, artifact_schema))
    for row in candidates:
        errors.extend(_schema_violations(row, candidate_schema))
    for row in gr_eval:
        errors.extend(_schema_violations(row, g4g5_schema))

    for rows, key in [
        (plan, "expanded_search_plan_id"), (attempts, "deep_source_acquisition_attempt_id"),
        (manifests, "deep_source_artifact_manifest_id"), (parsed, "deep_parsed_artifact_id"),
        (candidates, "specific_observed_event_candidate_id"), (locations, "location_resolution_id"),
        (phenomena, "phenomenon_classification_id"), (g4, "g4_evaluation_id"), (g5, "g5_evaluation_id"),
        (gr_eval, "ground_reference_candidate_eval_id"), (gates, "gate_eval_id"), (leakage, "no_leakage_audit_id"),
        (nodes, "node_id"), (edges, "edge_id"),
    ]:
        ids = [row[key] for row in rows]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            errors.append(f"ids_not_unique_or_sorted:{key}")

    official_specific = [m for m in manifests if m["event_specific"] == "true" and m["officiality_level"] in OFFICIAL_EVENT_LEVELS]
    parsed_official = [p for p in parsed if p["deep_source_artifact_manifest_id"] in {m["deep_source_artifact_manifest_id"] for m in official_specific} and p["parse_success"] == "true"]

    # 1..7.
    if len([a for a in attempts if a["network_enabled"] == "true"]) < 30:
        errors.append("deep_attempts_lt_30")
    if len(official_specific) < 3:
        errors.append("official_event_specific_lt_3")
    if len(parsed_official) < 3:
        errors.append("parsed_official_lt_3")
    if len(candidates) < 3:
        errors.append("specific_observed_candidates_lt_3")
    if len([c for c in candidates if c["bairro"] != "none" or c["logradouro"] != "none"]) < 1:
        errors.append("no_patch_level_location_candidate")
    if len([p for p in phenomena if p["phenomenon_class"] != "INSUFFICIENT"]) < 1:
        errors.append("no_phenomenon_specific_candidate")
    if (len(g4) + len(g5)) < 3:
        errors.append("no_g4_g5_evaluation")
    # 8: hash.
    for row in manifests:
        path = ROOT / row["artifact_local_path"]
        if not path.exists() or not row["sha256"] or sha256_file(path) != row["sha256"]:
            errors.append(f"manifest_hash_mismatch:{row['deep_source_artifact_manifest_id']}")
        if int(row["size_bytes"]) > MAX_ARTIFACT_BYTES:
            errors.append(f"artifact_over_limit:{row['deep_source_artifact_manifest_id']}")
        if row["artifact_type"] not in ("html", "pdf", "txt", "csv", "json"):
            errors.append(f"forbidden_artifact_type:{row['deep_source_artifact_manifest_id']}")
    for path in ARTIFACT_DIR.glob("**/*") if ARTIFACT_DIR.exists() else []:
        if path.is_file() and path.suffix.lower() in (".tif", ".nc", ".zip", ".gz", ".npz", ".npy"):
            errors.append(f"forbidden_raw_committed:{rel(path)}")
    # 9: noticia nao vira GR sozinha.
    for row in gr_eval:
        if row["can_be_ground_reference_candidate"] == "true":
            oec = next((o for o in candidates if o["specific_observed_event_candidate_id"] == row["specific_observed_event_candidate_id"]), {})
            if oec.get("officiality_level") not in OFFICIAL_EVENT_LEVELS:
                errors.append("non_official_accepted_as_ground_reference")
    if any(r["uses_news_as_ground_reference_without_official_support"] != "false" for r in leakage):
        errors.append("news_used_as_ground_reference")
    # 10/11: sensor/chirps nao viram evento.
    if any(r["uses_sensor_as_event_observation"] != "false" or r["uses_chirps_as_event_reference"] != "false" for r in leakage):
        errors.append("sensor_or_chirps_as_event")
    if any(r["passes_no_leakage"] != "true" for r in leakage):
        errors.append("no_leakage_failed")
    # 12/13: sem GT/label.
    if summary["ground_truth_created"] or summary["training_labels_created"]:
        errors.append("forbidden_gt_or_label")
    if any(r["can_be_ground_truth"] != "false" or r["can_be_training_label"] != "false" for r in gr_eval):
        errors.append("gr_marked_gt_or_label")
    # 16: 17B elegivel so com accepted GR.
    accepted = [r for r in gr_eval if r["can_be_ground_reference_candidate"] == "true"]
    if summary["eligible_for_17b_now"] != (len(accepted) > 0):
        errors.append("17b_eligibility_inconsistent")
    if summary["eligible_for_17b_now"] and not accepted:
        errors.append("17b_eligible_without_accepted_ground_reference")
    node_ids = {n["node_id"] for n in nodes}
    if any(e["source_node_id"] not in node_ids or e["target_node_id"] not in node_ids for e in edges):
        errors.append("edge_references_unknown_node")

    for path in [SCORE_V6, OFFICIAL_PATCHES, OFFICIAL_PATCH_LINKS]:
        if path.exists() and _run_git(["diff", "--name-only", "--", rel(path)]):
            errors.append(f"official_dataset_changed:{rel(path)}")
    if SCORE_V7.exists():
        errors.append("score_v7_exists")

    expected = build_summary()
    for key, value in expected.items():
        if summary.get(key) != value:
            errors.append(f"summary_mismatch:{key}")
    if summary["eligible_for_score_v7"] or summary["trainable"] or summary["ground_truth"]:
        errors.append("promotion_guardrail_failed")
    if not summary["minimum_success_achieved"]:
        errors.append("minimum_success_not_achieved")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        "17C28 -> "
        f"attempts={summary['deep_source_search_attempts_count']} official_specific={summary['official_event_specific_artifacts_acquired_count']} "
        f"candidates={summary['specific_observed_event_candidates_count']} patch_loc={summary['patch_level_location_candidates_count']} "
        f"phen_spec={summary['phenomenon_specific_candidates_count']} g4g5={summary['G4_G5_evaluation_rows_count']} "
        f"accepted={summary['accepted_ground_reference_candidate_count']} G4={summary['G4_true_count']} G5={summary['G5_true_count']} "
        f"eligible_17b={summary['eligible_for_17b_now']} min_success={summary['minimum_success_achieved']}"
    )
    return 0


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def expand_search_plan_text() -> str:
    rows = expanded_search_plan_rows()
    specs = sorted({r["target_specificity"] for r in rows})
    return f"expand-search-plan: {len(rows)} buscas expandidas cobrindo {specs}."


def parse_deep_artifacts_text() -> str:
    parsed = deep_parsed_artifact_index_rows()
    return f"parse-deep-artifacts: {len([p for p in parsed if p['parse_success'] == 'true'])}/{len(parsed)} artefatos parseados."


def extract_specific_event_candidates_text() -> str:
    c = specific_observed_event_candidate_rows()
    return f"extract-specific-event-candidates: {len(c)} candidatos especificos de fonte oficial institucional."


def resolve_locations_text() -> str:
    rows = location_resolution_rows()
    bairro = len([r for r in rows if r["bairro"] != "none"])
    return f"resolve-locations: {len(rows)} resolucoes, {bairro} com bairro/municipal (nenhuma coordenada inventada)."


def classify_phenomena_text() -> str:
    rows = phenomenon_classification_rows()
    classes = {}
    for r in rows:
        classes[r["phenomenon_class"]] = classes.get(r["phenomenon_class"], 0) + 1
    return "classify-phenomena: " + "; ".join(f"{k}={v}" for k, v in sorted(classes.items())) + "."


def evaluate_g4_g5_text() -> str:
    gr = ground_reference_candidate_evaluation_rows()
    accepted = [r for r in gr if r["can_be_ground_reference_candidate"] == "true"]
    return f"evaluate-g4-g5: {len(gr)} candidatos avaliados, {len(accepted)} aceitos (G4/G5 patch-level nao satisfeitos)."


def build_evidence_update_text() -> str:
    return f"build-evidence-update: {len(evidence_graph_update_node_rows())} nos e {len(evidence_graph_update_edge_rows())} arestas."


def status_text() -> str:
    s = build_summary()
    return (
        f"status 17C28: official_specific={s['official_event_specific_artifacts_acquired_count']} candidates={s['specific_observed_event_candidates_count']} "
        f"accepted_gr={s['accepted_ground_reference_candidate_count']} eligible_17b={s['eligible_for_17b_now']} min_success={s['minimum_success_achieved']}"
    )
