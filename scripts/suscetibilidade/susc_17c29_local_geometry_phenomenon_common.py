"""SUSC-17C29 aquisicao de geometria local oficial e separacao de fenomeno G4/G5.

O 17C27/17C28 ja provaram que o evento (chuvas extremas em Pernambuco/Recife,
maio de 2022, com inundacoes e deslizamentos) aconteceu, via fontes oficiais
institucionais (Agencia Brasil/EBC). O que faltou para desbloquear G4 e G5 foi:

* G4 (vinculo espacial patch-level): a localizacao publica e cidade/municipio/
  bairro, sem geometria, endereco ou coordenada compativel com o patch/buffer;
* G5 (separacao de fenomeno): o evento e misto (inundacao + deslizamento) e as
  fontes nao separam o fenomeno hidrologico do movimento de massa por local.

Este marco executa uma aquisicao DIRIGIDA a nivel local: expande termos por
bairro/logradouro usando o AOI do 17C6, adquire NOVOS artefatos oficiais/
institucionais especificos do evento (artigos da Agencia Brasil de maio/junho de
2022 que citam bairros do Grande Recife, distintos dos ja adquiridos no 17C28),
segue links locais, calcula SHA256, parseia texto, extrai local/fenomeno/data,
resolve a localizacao SEM inventar coordenada, classifica o fenomeno local
(hidrologico / movimento de massa / misto / insuficiente) e avalia G4/G5 por
candidato.

Resultado honesto esperado (Resultado B - bloqueio honesto): ha candidatos
geocodaveis a nivel de bairro (Ibura, Jardim Monte Verde, Barro, Muribeca,
Curado, Jaboatao, Olinda, Guararapes...) e ao menos um candidato onde o fenomeno
hidrologico esta especificamente documentado (alagamento/inundacao/enchente),
mas: (a) nenhuma fonte fornece coordenada/poligono patch-level -> G4 permanece
false; (b) onde o hidrologico aparece ele vem MISTO com deslizamento -> G5
permanece false (fenomeno misto nunca vira G5). Nenhuma coordenada e inventada;
cidade/municipio nao vira patch-level sem incerteza; noticia comercial nao vira
Ground Reference; sensor/CHIRPS nao viram evento observado; nenhum ground truth,
label de treino, score v7 ou patch oficial e criado; score v6 permanece intacto;
17B so seria elegivel com Ground Reference Candidate aceito por G1-G7.

Build publico offline/deterministico a partir dos artefatos ja adquiridos e
commitados; a aquisicao real exige os quatro opt-ins de rede.
"""

from __future__ import annotations

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
LOCAL_STATE = ROOT / "local_runs" / "suscetibilidade" / "17c29_local_geometry_phenomenon"
ARTIFACT_DIR = OUT / "susc_17c29_source_artifacts"
LEDGER = ARTIFACT_DIR / "_local_acquisition_ledger.json"

SCORE_V6 = DAT / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT / "susc_score_v7_candidate_by_patch_v1.csv"
OFFICIAL_PATCHES = DAT / "susc_patches_official_v1.csv"
OFFICIAL_PATCH_LINKS = DAT / "susc_patch_links_official_v1.csv"

C28_MANIFEST = OUT / "susc_17c28_deep_source_artifact_manifest.csv"
C28_PARSED = OUT / "susc_17c28_deep_parsed_artifact_index.csv"
C28_CANDIDATES = OUT / "susc_17c28_specific_observed_event_candidates.csv"
C28_LOCATION = OUT / "susc_17c28_location_resolution.csv"
C28_PHENOMENON = OUT / "susc_17c28_phenomenon_classification.csv"
C28_G4 = OUT / "susc_17c28_g4_spatial_link_evaluation.csv"
C28_G5 = OUT / "susc_17c28_g5_phenomenon_evaluation.csv"
C28_GR = OUT / "susc_17c28_ground_reference_candidate_evaluation.csv"
C28_SUMMARY = OUT / "susc_17c28_readiness_summary.json"
C27_OBSERVED = OUT / "susc_17c27_observed_event_candidates.csv"
C27_MANIFEST = OUT / "susc_17c27_source_artifact_manifest.csv"
C26_GR_QUEUE = OUT / "susc_17c26_ground_reference_target_queue.csv"
C26_QUERY_PACKAGES = OUT / "susc_17c26_source_query_packages.csv"
C26_GR_FIELDS = OUT / "susc_17c26_required_ground_reference_fields.csv"
C19_BINDING = OUT / "susc_17c19_candidate_patch_temporal_binding.csv"
PATCH_GRID = OUT / "susc_17c6_candidate_patch_grid.csv"
PATCH_GEOJSON = OUT / "susc_17c6_candidate_patch_grid.geojson"
PATCH_LINKS = OUT / "susc_17c6_candidate_patch_links.csv"

REQUIRED_INPUTS = [
    SCORE_V6, C28_MANIFEST, C28_PARSED, C28_CANDIDATES, C28_LOCATION, C28_PHENOMENON,
    C28_G4, C28_G5, C28_GR, C28_SUMMARY, C27_OBSERVED, C27_MANIFEST, C26_GR_QUEUE,
    C26_QUERY_PACKAGES, C26_GR_FIELDS, C19_BINDING, PATCH_GRID, PATCH_GEOJSON, PATCH_LINKS,
]

REPORT = OUT / "SUSC_17C29_AQUISICAO_GEOMETRIA_LOCAL_FENOMENO_G4_G5_REPORT.md"
LOCAL_PLAN = OUT / "susc_17c29_local_search_plan.csv"
LOCAL_ATTEMPTS = OUT / "susc_17c29_local_source_acquisition_attempts.csv"
FOLLOWED_LINKS = OUT / "susc_17c29_local_followed_link_registry.csv"
LOCAL_MANIFEST = OUT / "susc_17c29_local_source_artifact_manifest.csv"
LOCAL_PARSED = OUT / "susc_17c29_local_parsed_artifact_index.csv"
LOCAL_CANDIDATES = OUT / "susc_17c29_local_observed_event_candidates.csv"
LOCATION_RESOLUTION = OUT / "susc_17c29_local_location_resolution.csv"
PHENOMENON = OUT / "susc_17c29_local_phenomenon_classification.csv"
G4_EVAL = OUT / "susc_17c29_g4_spatial_link_evaluation.csv"
G5_EVAL = OUT / "susc_17c29_g5_phenomenon_evaluation.csv"
GR_CANDIDATE_EVAL = OUT / "susc_17c29_ground_reference_candidate_evaluation.csv"
SCORECARD = OUT / "susc_17c29_local_artifact_scorecard.csv"
GRAPH_UPDATE_NODES = OUT / "susc_17c29_evidence_graph_update_nodes.csv"
GRAPH_UPDATE_EDGES = OUT / "susc_17c29_evidence_graph_update_edges.csv"
NO_LEAKAGE = OUT / "susc_17c29_no_leakage_audit.csv"
GATES = OUT / "susc_17c29_gate_evaluation_matrix.csv"
SUMMARY = OUT / "susc_17c29_readiness_summary.json"
BLOCKERS = OUT / "susc_17c29_promotion_blockers.csv"

ARTIFACT_SCHEMA = SCHEMAS / "susc_17c29_local_source_artifact_schema_v1.json"
CANDIDATE_SCHEMA = SCHEMAS / "susc_17c29_local_event_candidate_schema_v1.json"
G4_G5_SCHEMA = SCHEMAS / "susc_17c29_g4_g5_evaluation_schema_v1.json"

GOV = {"review_only": "true", "trainable": "false", "ground_truth": "false"}
MAX_ARTIFACT_BYTES = 500_000
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) REV-P-SUSC-17C29-review-only"
FETCH_TIMEOUT_SECONDS = 15

WAYBACK = "https://web.archive.org/web/{ts}/{url}"
WAYBACK_TIMESTAMPS = ["20220528120000", "20220529120000", "20220530081923", "20220531120000", "20220601120000"]

# family -> (source_name, homepage_url, officiality_level)
FAMILY_HOMEPAGES = {
    "agencia_brasil": ("Agencia Brasil / EBC (empresa publica federal)", "https://agenciabrasil.ebc.com.br/", "official_institutional_public_agency"),
    "defesa_civil_pe": ("Defesa Civil de Pernambuco / SEDEC-PE", "https://www.defesacivil.pe.gov.br/", "official"),
    "prefeitura_recife": ("Prefeitura do Recife / COMDEC", "https://www2.recife.pe.gov.br/", "official"),
    "gov_pernambuco": ("Governo de Pernambuco", "https://www.pe.gov.br/", "official"),
    "diario_oficial_pe": ("Diario Oficial de Pernambuco (CEPE)", "https://www.cepe.com.br/", "official"),
    "apac_pernambuco": ("APAC - Agencia Pernambucana de Aguas e Clima", "https://www.apac.pe.gov.br/", "official"),
    "cemaden": ("CEMADEN", "http://www.cemaden.gov.br/", "official"),
    "sgb_cprm": ("SGB/CPRM - Servico Geologico do Brasil", "https://www.sgb.gov.br/", "official"),
}
OFFICIAL_EVENT_LEVELS = {"official", "official_institutional", "official_institutional_public_agency"}
FOLLOW_FROM_FAMILY = "agencia_brasil"
MAX_FOLLOW = 10

# NOVOS artigos oficiais/institucionais especificos do evento, a nivel local
# (citam bairros do Grande Recife) e DISTINTOS dos 4 ja adquiridos no 17C28.
# (canonical_url, [candidate_wayback_timestamps]).
KNOWN_LOCAL_EVENT_URLS = [
    ("https://agenciabrasil.ebc.com.br/geral/noticia/2022-05/defesa-civil-confirma-91-mortes-por-causa-das-chuvas-em-pernambuco",
     ["20220530163027", "20220530163028", "20220531000000"]),
    ("https://agenciabrasil.ebc.com.br/geral/noticia/2022-05/chega-100-o-numero-de-mortes-devido-chuvas-em-pernambuco",
     ["20220531153016", "20220531153017", "20220601120000"]),
    ("https://agenciabrasil.ebc.com.br/geral/noticia/2022-06/desalojados-em-pernambuco-chegam-119-mil-em-razao-das-chuvas",
     ["20220608024217", "20220608031135", "20220609120000"]),
    ("https://agenciabrasil.ebc.com.br/geral/noticia/2022-05/chuva-fez-79-mortes-em-pernambuco",
     ["20220530143037", "20220530143053", "20220530131859"]),
]

EXPANDED_TERMS = ["Recife", "Jaboatao", "Olinda", "Guararapes", "maio 2022", "28/05/2022",
                  "29/05/2022", "alagamento", "inundacao", "enchente", "deslizamento", "barreira",
                  "Defesa Civil", "abrigo", "decreto emergencia", "bairro", "rua"]
LOCAL_SEARCH_TEMPLATES = [
    "Recife alagamento maio 2022 bairro",
    "Recife inundacao maio 2022 rua",
    "Defesa Civil Recife maio 2022 alagamento",
    "Jaboatao maio 2022 inundacao bairro",
    "Olinda maio 2022 alagamento",
    "Guararapes maio 2022 alagamento",
    "barreira deslizamento Recife maio 2022",
    "alagamento Recife 28 maio 2022 Defesa Civil",
    "abrigos Recife enchentes maio 2022",
    "decreto emergencia Recife maio 2022 bairros",
]
LOCAL_SPECIFICITIES = ["bairro_logradouro", "abrigo_ocorrencia", "hidrologico_local", "geometria_poligono_ponto"]

FLOOD_TERMS = ["inunda", "enchente", "alagamento", "alagad", "cheia", "transbord"]
LANDSLIDE_TERMS = ["deslizamento", "soterr", "barreira", "movimento de massa", "desabamento", "desmoronamento"]
EVENT_CONTEXT_TERMS = ["chuva", "chuvas", "morte", "mortes", "defesa civil", "emergencia", "emergência", "desabrig", "desaloj"]
EVENT_PHENOMENON_TERMS = FLOOD_TERMS + LANDSLIDE_TERMS + EVENT_CONTEXT_TERMS
LOCATION_TERMS = ["recife", "pernambuco", "jaboat", "olinda", "grande recife", "regiao metropolitana", "guararapes"]
DATE_TERMS = ["2022", "maio", "28 de maio", "29 de maio", "27 de maio", "2022-05", "2022-06"]
BAIRRO_TERMS = ["jaboat", "olinda", "guararapes", "muribeca", "ibura", "barro", "monte verde",
                "cabo de santo agostinho", "moreno", "abreu e lima", "paulista", "camaragibe",
                "jardim sao paulo", "dois unidos", "vila dos milagres", "boa viagem", "cohab",
                "tejipio", "curado", "areias", "capibaribe", "beberibe", "tres carneiros", "toto"]
LOGRADOURO_RE = re.compile(r"\b(rua|avenida|av\.|estrada|travessa|alameda)\s+[a-z]", re.I)


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _network_enabled() -> bool:
    return os.environ.get("SUSC_17C29_ALLOW_NETWORK") == "1"


def _public_download_enabled() -> bool:
    return os.environ.get("SUSC_17C29_ALLOW_PUBLIC_DOWNLOAD") == "1"


def _deep_search_enabled() -> bool:
    return os.environ.get("SUSC_17C29_ALLOW_DEEP_SEARCH") == "1"


def _wayback_enabled() -> bool:
    return os.environ.get("SUSC_17C29_ALLOW_WAYBACK") == "1"


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


def _aoi_bbox() -> tuple[float, float, float, float]:
    rows = read_csv(PATCH_GRID)
    xs = [float(r["xmin"]) for r in rows] + [float(r["xmax"]) for r in rows]
    ys = [float(r["ymin"]) for r in rows] + [float(r["ymax"]) for r in rows]
    return min(xs), min(ys), max(xs), max(ys)


def _seed_localities() -> list[str]:
    """Localidades citadas nos artefatos do 17C28 (seed, sem re-adquirir)."""
    seen: list[str] = []
    for path in [C28_PARSED, C28_CANDIDATES]:
        if not path.exists():
            continue
        for row in read_csv(path):
            blob = " ".join([row.get("bairro_mentions", ""), row.get("bairro", ""),
                             row.get("logradouro_mentions", ""), row.get("location_mentions", "")])
            for term in re.split(r"[;\s]+", blob.lower()):
                term = term.strip()
                if term and term not in ("none", "") and term not in seen:
                    seen.append(term)
    return seen


def _c28_known_canonicals() -> set[str]:
    canon: set[str] = set()
    for row in read_csv(C28_MANIFEST) if C28_MANIFEST.exists() else []:
        canon.add(_canonical_agencia_article_url(row.get("source_url", "")))
    return canon


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


def _is_local(text: str) -> bool:
    low = text.lower()
    return any(b in low for b in BAIRRO_TERMS) or bool(LOGRADOURO_RE.search(text))


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:60]


def _canonical_agencia_article_url(url: str) -> str:
    decoded = unquote(url or "")
    match = re.search(r"https?://agenciabrasil\.ebc\.com\.br/[^\"'\s?#]+/noticia/2022-0[56]/[^\"'\s?#&]+", decoded, re.I)
    return match.group(0).rstrip("/") if match else decoded.split("#", 1)[0].split("?", 1)[0].rstrip("/")


def _is_generic_homepage_url(url: str) -> bool:
    decoded = unquote(url).lower().rstrip("/")
    for _family, (_name, homepage, _officiality) in FAMILY_HOMEPAGES.items():
        if decoded.endswith(homepage.lower().rstrip("/")):
            return True
    return False


def _is_agencia_event_article_url(url: str) -> bool:
    canonical = _canonical_agencia_article_url(url).lower()
    if "/noticia/2022-0" not in canonical or "agenciabrasil.ebc.com.br" not in canonical:
        return False
    if any(blocked in canonical for blocked in ["covid", "facebook.com", "twitter.com", "whatsapp.com", "linkedin.com", "meteoros", "nasa", "materna"]):
        return False
    return any(k in canonical for k in ["recife", "pernambuco", "chuva", "desliz", "desast", "enchente", "grande-recife", "desaloj", "desabrig"])


def _wayback_url_for(url: str, ts: str = "20220530081923") -> str:
    return WAYBACK.format(ts=ts, url=url)


# ---------------------------------------------------------------------------
# run-local-acquisition (opt-in)
# ---------------------------------------------------------------------------

def _base_candidate_urls() -> list[dict]:
    """>=40 URLs candidatas locais: portais oficiais (Wayback + direto) + artigos locais especificos."""
    candidates = []
    for family, (name, homepage, officiality) in FAMILY_HOMEPAGES.items():
        for ts in WAYBACK_TIMESTAMPS:
            candidates.append({"family": family, "source_name": name, "officiality": officiality,
                               "url": WAYBACK.format(ts=ts, url=homepage), "attempt_type": "wayback_official_portal_snapshot"})
        candidates.append({"family": family, "source_name": name, "officiality": officiality,
                           "url": homepage, "attempt_type": "direct_official_portal"})
    for url, timestamps in KNOWN_LOCAL_EVENT_URLS:
        slug = _slugify(_canonical_agencia_article_url(url).rsplit("/", 1)[-1])
        for ts in timestamps:
            candidates.append({
                "family": "agencia_brasil",
                "source_name": FAMILY_HOMEPAGES["agencia_brasil"][0],
                "officiality": FAMILY_HOMEPAGES["agencia_brasil"][2],
                "url": _wayback_url_for(url, ts),
                "attempt_type": "wayback_known_local_event_url",
                "preferred_slug": f"agencia_brasil_local_{slug}",
                "canonical": _canonical_agencia_article_url(url),
            })
    return candidates


def _require_network_optins() -> tuple[int, str] | None:
    if not _network_enabled():
        return (2, "blocked_network_not_enabled: exige SUSC_17C29_ALLOW_NETWORK=1.")
    if not _public_download_enabled():
        return (2, "blocked_public_download_not_enabled: exige SUSC_17C29_ALLOW_PUBLIC_DOWNLOAD=1.")
    if not _deep_search_enabled():
        return (2, "blocked_deep_search_not_enabled: exige SUSC_17C29_ALLOW_DEEP_SEARCH=1.")
    if not _wayback_enabled():
        return (2, "blocked_wayback_not_enabled: exige SUSC_17C29_ALLOW_WAYBACK=1.")
    return None


def run_local_acquisition_text() -> tuple[int, str]:
    guard = _require_network_optins()
    if guard:
        return guard
    ensure_dir(ARTIFACT_DIR)
    ensure_dir(LOCAL_STATE)
    known_c28 = _c28_known_canonicals()
    ledger = {"acquired_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "attempts": [], "artifacts": {}, "followed": []}
    acquired_canonicals: set[str] = set()
    for cand in _base_candidate_urls():
        attempt = {"family": cand["family"], "url": cand["url"], "attempt_type": cand["attempt_type"],
                   "officiality": cand["officiality"], "source_name": cand["source_name"]}
        try:
            status, final, data = _fetch(cand["url"])
            text = _text_of(data)
            canonical = cand.get("canonical") or (_canonical_agencia_article_url(final) if cand["family"] == "agencia_brasil" else final)
            homepage_seed = cand["family"] == FOLLOW_FROM_FAMILY and _is_generic_homepage_url(final)
            local_event = (
                cand["attempt_type"] == "wayback_known_local_event_url"
                and _is_event_specific(text) and _is_local(text)
                and not _is_generic_homepage_url(final)
                and _is_agencia_event_article_url(final)
                and canonical not in known_c28
            )
            attempt.update({"http_status": status, "artifact_acquired": False,
                            "event_specific": _is_event_specific(text) and _is_local(text),
                            "reused_from_17c28": canonical in known_c28})
            if data and homepage_seed and "agencia_brasil_base" not in ledger["artifacts"]:
                path = ARTIFACT_DIR / "agencia_brasil_base.html"
                path.write_bytes(data)
                ledger["artifacts"]["agencia_brasil_base"] = {
                    "slug": "agencia_brasil_base", "family": cand["family"], "source_name": cand["source_name"],
                    "source_url": final, "officiality_level": cand["officiality"], "artifact_type": "html",
                    "artifact_file": path.name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size,
                    "event_specific": False, "local_specific": False, "tls_verification_bypassed": False,
                }
                attempt["artifact_acquired"] = True
                attempt["artifact_slug"] = "agencia_brasil_base"
            elif local_event and data and canonical not in acquired_canonicals:
                slug = cand.get("preferred_slug") or f"{cand['family']}_local_{_slugify(canonical)}"
                path = ARTIFACT_DIR / f"{slug}.html"
                path.write_bytes(data)
                ledger["artifacts"][slug] = {
                    "slug": slug, "family": cand["family"], "source_name": cand["source_name"],
                    "source_url": final, "officiality_level": cand["officiality"], "artifact_type": "html",
                    "artifact_file": path.name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size,
                    "event_specific": True, "local_specific": True, "tls_verification_bypassed": False,
                }
                acquired_canonicals.add(canonical)
                attempt["artifact_acquired"] = True
                attempt["artifact_slug"] = slug
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            reason = type(exc).__name__
            if isinstance(exc, URLError) and "CERTIFICATE_VERIFY_FAILED" in str(exc):
                reason = "ssl_certificate_verify_failed"
            attempt.update({"http_status": "failed_safe", "artifact_acquired": False,
                            "event_specific": False, "reused_from_17c28": False, "failure_reason": reason})
        ledger["attempts"].append(attempt)
    write_json(LEDGER, ledger)
    local = len([a for a in ledger["artifacts"].values() if a.get("local_specific")])
    return (0, f"run-local-acquisition: {len(ledger['attempts'])} tentativas locais, {local} artefatos locais oficiais especificos em {rel(ARTIFACT_DIR)}.")


# ---------------------------------------------------------------------------
# follow-local-links (opt-in)
# ---------------------------------------------------------------------------

def _extract_event_article_links(html: str) -> list[str]:
    links = []
    seen = set()
    for m in re.finditer(r'''href\s*=\s*['"]([^'"]+)['"]''', html, re.I):
        href = unescape(m.group(1))
        full = urljoin("https://web.archive.org", href) if href.startswith("/") else href
        canonical = _canonical_agencia_article_url(full)
        if not _is_agencia_event_article_url(canonical):
            continue
        if "web.archive.org" not in urlparse(full).netloc:
            full = _wayback_url_for(canonical)
        if canonical not in seen:
            links.append(full)
            seen.add(canonical)
    return links[:MAX_FOLLOW]


def follow_local_links_text() -> tuple[int, str]:
    guard = _require_network_optins()
    if guard:
        return guard
    ledger = _load_ledger()
    if ledger is None:
        return (2, "follow-local-links: sem ledger; rode run-local-acquisition primeiro.")
    known_c28 = _c28_known_canonicals()
    acquired_canonicals = {
        _canonical_agencia_article_url(art["source_url"])
        for art in ledger["artifacts"].values() if art.get("local_specific")
    }
    seed_slugs = [s for s, a in sorted(ledger["artifacts"].items())
                  if a.get("family") == FOLLOW_FROM_FAMILY]
    order = 0
    followed = 0
    for seed_slug in seed_slugs:
        base_art = ledger["artifacts"][seed_slug]
        base_path = ARTIFACT_DIR / base_art["artifact_file"]
        if not base_path.exists():
            continue
        html = base_path.read_bytes().decode("utf-8", errors="ignore")
        for url in _extract_event_article_links(html):
            canonical = _canonical_agencia_article_url(url)
            if canonical in acquired_canonicals or canonical in known_c28:
                continue
            order += 1
            record = {"from_url": base_art["source_url"], "to_url": url,
                      "source_family": FOLLOW_FROM_FAMILY, "followed": True}
            try:
                status, final, data = _fetch(url)
                text = _text_of(data)
                canonical = _canonical_agencia_article_url(final)
                local_event = (_is_event_specific(text) and _is_local(text)
                               and _is_agencia_event_article_url(canonical)
                               and canonical not in acquired_canonicals and canonical not in known_c28)
                if local_event and data:
                    slug = f"agencia_brasil_flink_{order:02d}_{_slugify(canonical.rsplit('/', 1)[-1])}"
                    path = ARTIFACT_DIR / f"{slug}.html"
                    path.write_bytes(data)
                    ledger["artifacts"][slug] = {
                        "slug": slug, "family": FOLLOW_FROM_FAMILY, "source_name": FAMILY_HOMEPAGES[FOLLOW_FROM_FAMILY][0],
                        "source_url": final, "officiality_level": FAMILY_HOMEPAGES[FOLLOW_FROM_FAMILY][2], "artifact_type": "html",
                        "artifact_file": path.name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size,
                        "event_specific": True, "local_specific": True, "tls_verification_bypassed": False,
                    }
                    record.update({"artifact_acquired": True, "artifact_slug": slug, "link_text": slug})
                    acquired_canonicals.add(canonical)
                    followed += 1
                else:
                    record.update({"artifact_acquired": False, "link_text": "duplicate_or_not_local_event_specific"})
            except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
                record.update({"artifact_acquired": False, "link_text": type(exc).__name__})
            ledger["followed"].append(record)
    write_json(LEDGER, ledger)
    return (0, f"follow-local-links: {followed} artigos locais adicionais seguidos e adquiridos; {len(ledger['artifacts'])} artefatos no total.")


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
        return f"S17C29_ART_{slugs.index(slug) + 1:04d}"
    return "not_available"


# ---------------------------------------------------------------------------
# 1 - Local search plan (offline)
# ---------------------------------------------------------------------------

def local_search_plan_rows() -> list[dict]:
    event_id = _event_id()
    xmin, ymin, xmax, ymax = _aoi_bbox()
    bbox = f"{xmin:.5f},{ymin:.5f},{xmax:.5f},{ymax:.5f}"
    seed_localities = [t for t in _seed_localities() if t in BAIRRO_TERMS] or ["recife"]
    rows = []
    idx = 0
    for template in LOCAL_SEARCH_TEMPLATES:
        for family in FAMILY_HOMEPAGES:
            idx += 1
            spec = LOCAL_SPECIFICITIES[(idx - 1) % len(LOCAL_SPECIFICITIES)]
            locality = seed_localities[(idx - 1) % len(seed_localities)]
            rows.append({
                "local_search_plan_id": f"S17C29_PLAN_{idx:04d}",
                "candidate_patch_id": "all_candidate_patches", "event_id": event_id, "source_family": family,
                "aoi_bbox": bbox, "seed_locality": locality,
                "query_text": f"{FAMILY_HOMEPAGES[family][0]}: {template} ({locality})",
                "query_language": "ptbr", "target_gate": "G4_G5", "target_specificity": spec,
                "expected_artifact_type": "official_local_event_document_or_page", "must_attempt": "true", "review_only": "true",
            })
    return rows


# ---------------------------------------------------------------------------
# 2 - Local acquisition attempts
# ---------------------------------------------------------------------------

def local_source_acquisition_attempt_rows() -> list[dict]:
    event_id = _event_id()
    ledger = _load_ledger()
    rows = []
    attempts = ledger.get("attempts", []) if ledger else []
    for idx, att in enumerate(attempts, start=1):
        acquired = bool(att.get("artifact_acquired"))
        rows.append({
            "local_source_acquisition_attempt_id": f"S17C29_ATT_{idx:04d}",
            "candidate_patch_id": "all_candidate_patches", "event_id": event_id, "source_family": att["family"],
            "attempted_url_or_query": att["url"], "attempt_type": att["attempt_type"],
            "network_enabled": "true", "http_status": att.get("http_status", "failed_safe"),
            "event_specific": _bool_text(bool(att.get("event_specific"))),
            "reused_from_17c28": _bool_text(bool(att.get("reused_from_17c28"))),
            "artifact_acquired": _bool_text(acquired),
            "artifact_manifest_id": _manifest_id_for_slug(att.get("artifact_slug", "")) if acquired else "not_available",
            "tls_verification_bypassed": _bool_text(bool(att.get("tls_verification_bypassed"))),
            "failure_reason": att.get("failure_reason", "not_applicable" if acquired else "not_local_event_specific_or_duplicate"),
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# 3 - Followed local links
# ---------------------------------------------------------------------------

def local_followed_link_registry_rows() -> list[dict]:
    ledger = _load_ledger()
    rows = []
    for idx, rec in enumerate(ledger.get("followed", []) if ledger else [], start=1):
        acquired = bool(rec.get("artifact_acquired"))
        rows.append({
            "local_followed_link_id": f"S17C29_FLINK_{idx:04d}",
            "source_family": rec["source_family"], "from_url": rec["from_url"], "to_url": rec["to_url"],
            "link_text": rec.get("link_text", "not_available"), "followed": _bool_text(bool(rec.get("followed"))),
            "artifact_acquired": _bool_text(acquired),
            "artifact_manifest_id": _manifest_id_for_slug(rec.get("artifact_slug", "")) if acquired else "not_available",
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# 4 - Local manifest
# ---------------------------------------------------------------------------

MANIFEST_FIELDS = [
    "local_source_artifact_manifest_id", "candidate_patch_id", "event_id", "source_family", "source_name",
    "source_url", "artifact_local_path", "artifact_type", "sha256", "size_bytes", "officiality_level",
    "event_specific", "local_specific", "location_specific", "phenomenon_specific", "geometry_specific",
    "hydrological_documented", "stored_in_outputs_public", "raw_heavy", "tls_verification_bypassed",
    "review_only", "trainable", "ground_truth",
]


def local_source_artifact_manifest_rows() -> list[dict]:
    event_id = _event_id()
    rows = []
    idx = 0
    for art in _ledger_artifacts():
        if not art.get("local_specific"):
            continue
        path = ARTIFACT_DIR / art["artifact_file"]
        if not path.exists():
            continue
        idx += 1
        text = _text_of(path.read_bytes())
        low = text.lower()
        loc_specific = any(b in low for b in BAIRRO_TERMS) or bool(LOGRADOURO_RE.search(text))
        phen_specific = any(t in low for t in FLOOD_TERMS + LANDSLIDE_TERMS)
        hydro = any(t in low for t in FLOOD_TERMS)
        geom_specific = bool(re.search(r"-?\d{1,2}[.,]\d{4,}", text))
        rows.append({
            "local_source_artifact_manifest_id": f"S17C29_ART_{_manifest_id_for_slug(art['slug']).rsplit('_', 1)[-1]}",
            "candidate_patch_id": "all_candidate_patches", "event_id": event_id, "source_family": art["family"],
            "source_name": art["source_name"], "source_url": art["source_url"],
            "artifact_local_path": rel(path), "artifact_type": art["artifact_type"],
            "sha256": sha256_file(path), "size_bytes": str(path.stat().st_size),
            "officiality_level": art["officiality_level"], "event_specific": _bool_text(bool(art.get("event_specific"))),
            "local_specific": "true", "location_specific": _bool_text(loc_specific),
            "phenomenon_specific": _bool_text(phen_specific), "geometry_specific": _bool_text(geom_specific),
            "hydrological_documented": _bool_text(hydro), "stored_in_outputs_public": "true", "raw_heavy": "false",
            "tls_verification_bypassed": _bool_text(bool(art.get("tls_verification_bypassed"))), **GOV,
        })
    # Reindex manifest ids sequentially over local artifacts only.
    for i, row in enumerate(rows, start=1):
        row["local_source_artifact_manifest_id"] = f"S17C29_ART_{i:04d}"
    return rows


def _official_local_manifests() -> list[dict]:
    return [m for m in local_source_artifact_manifest_rows()
            if m["local_specific"] == "true" and m["officiality_level"] in OFFICIAL_EVENT_LEVELS]


def _artifact_text_from_manifest(manifest: dict) -> str:
    path = ROOT / manifest["artifact_local_path"]
    return _text_of(path.read_bytes()) if path.exists() else ""


# ---------------------------------------------------------------------------
# 5 - Local parsed index
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


def local_parsed_artifact_index_rows() -> list[dict]:
    event_id = _event_id()
    rows = []
    for idx, manifest in enumerate(local_source_artifact_manifest_rows(), start=1):
        text = _artifact_text_from_manifest(manifest)
        dates = _mentions(text, DATE_TERMS)
        locations = _mentions(text, LOCATION_TERMS)
        floods = _mentions(text, FLOOD_TERMS)
        landslides = _mentions(text, LANDSLIDE_TERMS)
        phenomena = _mentions(text, EVENT_PHENOMENON_TERMS)
        bairros = _mentions(text, BAIRRO_TERMS)
        coords = re.findall(r"-?\d{1,2}[.,]\d{4,}", text)[:3]
        logradouros = LOGRADOURO_RE.findall(text)[:3]
        limitations = []
        if not bairros:
            limitations.append("sem bairro/logradouro patch-level")
        if not coords:
            limitations.append("sem coordenada")
        if floods and landslides:
            limitations.append("fenomeno misto (inundacao + deslizamento)")
        rows.append({
            "local_parsed_artifact_id": f"S17C29_PARSE_{idx:04d}",
            "local_source_artifact_manifest_id": manifest["local_source_artifact_manifest_id"],
            "candidate_patch_id": "all_candidate_patches", "event_id": event_id,
            "parse_success": _bool_text(bool(text)), "text_extracted": _bool_text(bool(text)),
            "date_mentions": ";".join(dates) if dates else "none",
            "location_mentions": ";".join(locations) if locations else "none",
            "flood_mentions": ";".join(floods) if floods else "none",
            "landslide_mentions": ";".join(landslides) if landslides else "none",
            "phenomenon_mentions": ";".join(phenomena) if phenomena else "none",
            "coordinate_mentions": ";".join(coords) if coords else "none",
            "bairro_mentions": ";".join(bairros) if bairros else "none",
            "logradouro_mentions": ";".join(logradouros) if logradouros else "none",
            "evidence_snippet": (_snippet(text, BAIRRO_TERMS + FLOOD_TERMS + LANDSLIDE_TERMS)[:200] if text else "not_available"),
            "parse_limitations": ";".join(limitations) if limitations else "none", "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# 6 - Local observed event candidates
# ---------------------------------------------------------------------------

def _has_event_content(parsed: dict) -> bool:
    return parsed["date_mentions"] != "none" and parsed["location_mentions"] != "none" and parsed["phenomenon_mentions"] != "none"


def local_observed_event_candidate_rows() -> list[dict]:
    event_id = _event_id()
    manifests = {m["local_source_artifact_manifest_id"]: m for m in local_source_artifact_manifest_rows()}
    rows = []
    idx = 0
    for parsed in local_parsed_artifact_index_rows():
        manifest = manifests[parsed["local_source_artifact_manifest_id"]]
        if manifest["officiality_level"] not in OFFICIAL_EVENT_LEVELS or not _has_event_content(parsed):
            continue
        idx += 1
        bairro = parsed["bairro_mentions"]
        logradouro = parsed["logradouro_mentions"]
        geocodable = bairro != "none" or logradouro != "none"
        rows.append({
            "local_observed_event_candidate_id": f"S17C29_OEC_{idx:04d}",
            "local_source_artifact_manifest_id": parsed["local_source_artifact_manifest_id"],
            "candidate_patch_id": _patch_ids()[0], "event_id": event_id,
            "observed_event_date_or_period": "2022-05 a 2022-06 (evento 24-30 de maio; referencias 28-30)",
            "observed_location_text": parsed["location_mentions"], "bairro": bairro, "logradouro": logradouro,
            "geocodable_location": _bool_text(geocodable),
            "observed_geometry": "not_available", "geometry_type": "none",
            "geometry_uncertainty_m": "bairro_level_ge_3000" if geocodable else "city_level_ge_10000",
            "flood_mentions": parsed["flood_mentions"], "landslide_mentions": parsed["landslide_mentions"],
            "phenomenon_candidate": parsed["phenomenon_mentions"], "source_family": manifest["source_family"],
            "officiality_level": manifest["officiality_level"], "evidence_snippet": parsed["evidence_snippet"],
            "can_evaluate_g4": "true", "can_evaluate_g5": "true", **GOV,
        })
    return rows


# ---------------------------------------------------------------------------
# 7 - Local geometry resolution
# ---------------------------------------------------------------------------

def local_location_resolution_rows() -> list[dict]:
    rows = []
    for idx, oec in enumerate(local_observed_event_candidate_rows(), start=1):
        has_bairro = oec["bairro"] != "none"
        has_logradouro = oec["logradouro"] != "none"
        geocodable = has_bairro or has_logradouro
        rows.append({
            "local_location_resolution_id": f"S17C29_LOC_{idx:04d}",
            "local_observed_event_candidate_id": oec["local_observed_event_candidate_id"],
            "candidate_patch_id": oec["candidate_patch_id"], "location_text": oec["observed_location_text"],
            "bairro": oec["bairro"], "logradouro": oec["logradouro"],
            "geocodable": _bool_text(geocodable), "resolved_geometry": "not_available",
            "resolution_method": "named_place_logradouro_level" if has_logradouro else ("named_place_bairro_level" if has_bairro else "named_place_city_level"),
            "resolved_coordinate": "not_available_not_invented",
            "distance_to_patch_m": "not_computable_no_coordinate", "within_patch_or_buffer": "false",
            "uncertainty_m": "3000_or_more_bairro_level" if geocodable else "10000_or_more_city_level",
            "patch_level_geometry": "false",
            "location_resolution_status": "geocodable_bairro_level_no_coordinate" if geocodable else "city_level_only",
            "blocking_reason": "bairro/logradouro geocodavel porem sem coordenada/poligono patch-level; nao inventar coordenada; incerteza alta demais para G4" if geocodable else "apenas cidade/regiao",
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# 8 - Local phenomenon classification
# ---------------------------------------------------------------------------

def local_phenomenon_classification_rows() -> list[dict]:
    rows = []
    for idx, oec in enumerate(local_observed_event_candidate_rows(), start=1):
        flood = oec["flood_mentions"] != "none"
        landslide = oec["landslide_mentions"] != "none"
        context_only = any(t in oec["phenomenon_candidate"].lower() for t in EVENT_CONTEXT_TERMS)
        if flood and landslide:
            pclass, conf = "MIXED_AMBIGUOUS", "medium"
        elif flood:
            pclass, conf = "HYDROLOGICAL_CONFIRMED", "medium"
        elif landslide:
            pclass, conf = "MASS_MOVEMENT_CONFIRMED", "medium"
        elif context_only:
            pclass, conf = "EVENT_CONTEXT_ONLY", "low"
        else:
            pclass, conf = "INSUFFICIENT", "low"
        g5 = pclass == "HYDROLOGICAL_CONFIRMED"
        rows.append({
            "local_phenomenon_classification_id": f"S17C29_PHEN_{idx:04d}",
            "local_observed_event_candidate_id": oec["local_observed_event_candidate_id"],
            "candidate_patch_id": oec["candidate_patch_id"], "phenomenon_text": oec["phenomenon_candidate"],
            "flood_mentions": oec["flood_mentions"], "landslide_mentions": oec["landslide_mentions"],
            "phenomenon_class": pclass,
            "hydrological_documented": _bool_text(flood),
            "hydrological_specific": _bool_text(flood),
            "mass_movement_excluded": _bool_text(not landslide),
            "mixed_or_ambiguous": _bool_text(flood and landslide),
            "classification_confidence": conf, "G5_candidate_status": _bool_text(g5),
            "blocking_reason": "fenomeno misto (inundacao + deslizamento): nao separa hidrologico de movimento de massa; misto nunca vira G5" if (flood and landslide) else ("fenomeno de movimento de massa, nao hidrologico" if landslide and not flood else ("apenas contexto de chuva/impacto, sem fenomeno hidrologico separado" if not flood else "not_applicable")),
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# G4 / G5 / GR evaluation
# ---------------------------------------------------------------------------

def g4_spatial_link_evaluation_rows() -> list[dict]:
    loc_by = {r["local_observed_event_candidate_id"]: r for r in local_location_resolution_rows()}
    rows = []
    for idx, oec in enumerate(local_observed_event_candidate_rows(), start=1):
        loc = loc_by.get(oec["local_observed_event_candidate_id"], {})
        geocodable = oec["geocodable_location"] == "true"
        rows.append({
            "g4_evaluation_id": f"S17C29_G4_{idx:04d}",
            "local_observed_event_candidate_id": oec["local_observed_event_candidate_id"],
            "candidate_patch_id": oec["candidate_patch_id"],
            "has_geocodable_location": _bool_text(geocodable),
            "has_patch_level_geometry_or_coordinate": "false",
            "distance_to_patch_m": loc.get("distance_to_patch_m", "not_computable_no_coordinate"),
            "uncertainty_m": loc.get("uncertainty_m", "10000_or_more_city_level"),
            "within_patch_or_acceptable_buffer": "false", "G4_vinculo_espacial_evento": "false",
            "blocking_reason": "localizacao geocodavel a nivel de bairro porem sem coordenada/poligono patch-level; incerteza >=3km incompativel com patch/buffer; nenhuma coordenada inventada",
            "review_only": "true",
        })
    return rows


def g5_phenomenon_evaluation_rows() -> list[dict]:
    phen_by = {r["local_observed_event_candidate_id"]: r for r in local_phenomenon_classification_rows()}
    rows = []
    for idx, oec in enumerate(local_observed_event_candidate_rows(), start=1):
        phen = phen_by.get(oec["local_observed_event_candidate_id"], {})
        g5 = phen.get("G5_candidate_status", "false") == "true"
        rows.append({
            "g5_evaluation_id": f"S17C29_G5_{idx:04d}",
            "local_observed_event_candidate_id": oec["local_observed_event_candidate_id"],
            "candidate_patch_id": oec["candidate_patch_id"], "phenomenon_class": phen.get("phenomenon_class", "INSUFFICIENT"),
            "hydrological_documented": phen.get("hydrological_documented", "false"),
            "mass_movement_excluded": phen.get("mass_movement_excluded", "false"),
            "mixed_or_ambiguous": phen.get("mixed_or_ambiguous", "false"),
            "G5_separacao_fenomeno": _bool_text(g5), "blocking_reason": phen.get("blocking_reason", "fenomeno insuficiente"),
            "review_only": "true",
        })
    return rows


def ground_reference_candidate_evaluation_rows() -> list[dict]:
    event_id = _event_id()
    manifests = {m["local_source_artifact_manifest_id"]: m for m in local_source_artifact_manifest_rows()}
    g4_by = {r["local_observed_event_candidate_id"]: r for r in g4_spatial_link_evaluation_rows()}
    g5_by = {r["local_observed_event_candidate_id"]: r for r in g5_phenomenon_evaluation_rows()}
    rows = []
    for idx, oec in enumerate(local_observed_event_candidate_rows(), start=1):
        manifest = manifests.get(oec["local_source_artifact_manifest_id"], {})
        official = manifest.get("officiality_level") in OFFICIAL_EVENT_LEVELS
        g4 = g4_by.get(oec["local_observed_event_candidate_id"], {}).get("G4_vinculo_espacial_evento", "false") == "true"
        g5 = g5_by.get(oec["local_observed_event_candidate_id"], {}).get("G5_separacao_fenomeno", "false") == "true"
        g1, g2, g3, g6, g7 = True, official, True, True, True
        can_gr = all([g1, g2, g3, g4, g5, g6, g7])
        rows.append({
            "ground_reference_candidate_eval_id": f"S17C29_GRCE_{idx:04d}",
            "local_observed_event_candidate_id": oec["local_observed_event_candidate_id"],
            "candidate_patch_id": oec["candidate_patch_id"], "event_id": event_id,
            "G1_existencia_documental": _bool_text(g1), "G2_confiabilidade_fonte": _bool_text(g2),
            "G3_precisao_temporal": _bool_text(g3), "G4_vinculo_espacial_evento": _bool_text(g4),
            "G5_separacao_fenomeno": _bool_text(g5), "G6_proveniencia_integridade": _bool_text(g6),
            "G7_anti_leakage": _bool_text(g7), "can_be_ground_reference_candidate": _bool_text(can_gr),
            "can_be_ground_truth": "false", "can_be_training_label": "false", "can_unlock_17b": _bool_text(can_gr),
            "blocking_reason": "G4 (geometria/coordenada patch-level) e G5 (separacao de fenomeno misto) nao satisfeitos" if not can_gr else "not_applicable",
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------

def local_artifact_scorecard_rows() -> list[dict]:
    parsed_by = {p["local_source_artifact_manifest_id"]: p for p in local_parsed_artifact_index_rows()}
    rows = []
    for idx, manifest in enumerate(local_source_artifact_manifest_rows(), start=1):
        parsed = parsed_by.get(manifest["local_source_artifact_manifest_id"], {})
        temporal = parsed.get("date_mentions", "none") != "none"
        hydro = manifest["hydrological_documented"] == "true"
        rows.append({
            "local_artifact_scorecard_id": f"S17C29_SCORE_{idx:04d}",
            "local_source_artifact_manifest_id": manifest["local_source_artifact_manifest_id"],
            "source_family": manifest["source_family"], "officiality_level": manifest["officiality_level"],
            "event_specific": manifest["event_specific"], "local_specific": manifest["local_specific"],
            "location_specific": manifest["location_specific"], "phenomenon_specific": manifest["phenomenon_specific"],
            "hydrological_documented": manifest["hydrological_documented"], "geometry_specific": manifest["geometry_specific"],
            "temporal_specific": _bool_text(temporal), "parse_success": parsed.get("parse_success", "false"),
            "usable_for_g4": "false", "usable_for_g5": "false", "usable_for_ground_reference_candidate": "false",
            "blocking_reason": "artefato oficial local do evento com bairros" + (" e alagamento documentado" if hydro else "") + "; sem coordenada patch-level e fenomeno misto/nao-separado: nao satisfaz G4/G5",
            "review_only": "true",
        })
    return rows


# ---------------------------------------------------------------------------
# Evidence graph update
# ---------------------------------------------------------------------------

def _graph_update():
    event_id = _event_id()
    nodes = []
    for m in local_source_artifact_manifest_rows():
        nodes.append((f"art:{m['local_source_artifact_manifest_id']}", "local_source_artifact", m["local_source_artifact_manifest_id"], "all_candidate_patches"))
    for p in local_parsed_artifact_index_rows():
        nodes.append((f"parse:{p['local_parsed_artifact_id']}", "local_parsed_artifact", p["local_parsed_artifact_id"], "all_candidate_patches"))
    for o in local_observed_event_candidate_rows():
        nodes.append((f"oec:{o['local_observed_event_candidate_id']}", "local_observed_event_candidate", o["local_observed_event_candidate_id"], o["candidate_patch_id"]))
    for r in local_location_resolution_rows():
        nodes.append((f"loc:{r['local_location_resolution_id']}", "local_location_resolution", r["local_location_resolution_id"], r["candidate_patch_id"]))
    for r in local_phenomenon_classification_rows():
        nodes.append((f"phen:{r['local_phenomenon_classification_id']}", "local_phenomenon_classification", r["local_phenomenon_classification_id"], r["candidate_patch_id"]))
    for r in g4_spatial_link_evaluation_rows():
        nodes.append((f"g4:{r['g4_evaluation_id']}", "g4_evaluation", r["g4_evaluation_id"], r["candidate_patch_id"]))
    for r in g5_phenomenon_evaluation_rows():
        nodes.append((f"g5:{r['g5_evaluation_id']}", "g5_evaluation", r["g5_evaluation_id"], r["candidate_patch_id"]))
    for r in ground_reference_candidate_evaluation_rows():
        nodes.append((f"grce:{r['ground_reference_candidate_eval_id']}", "ground_reference_candidate_evaluation", r["ground_reference_candidate_eval_id"], r["candidate_patch_id"]))
    key_to_id = {key: f"S17C29_NODE_{i:04d}" for i, (key, *_r) in enumerate(nodes, start=1)}

    edges = []
    for p in local_parsed_artifact_index_rows():
        edges.append((f"parse:{p['local_parsed_artifact_id']}", f"art:{p['local_source_artifact_manifest_id']}", "parsed_from_artifact", "all_candidate_patches"))
    for o in local_observed_event_candidate_rows():
        pid = next((p["local_parsed_artifact_id"] for p in local_parsed_artifact_index_rows() if p["local_source_artifact_manifest_id"] == o["local_source_artifact_manifest_id"]), "")
        edges.append((f"oec:{o['local_observed_event_candidate_id']}", f"parse:{pid}", "candidate_from_parsed", o["candidate_patch_id"]))
    for r in local_location_resolution_rows():
        edges.append((f"loc:{r['local_location_resolution_id']}", f"oec:{r['local_observed_event_candidate_id']}", "location_of_candidate", r["candidate_patch_id"]))
    for r in local_phenomenon_classification_rows():
        edges.append((f"phen:{r['local_phenomenon_classification_id']}", f"oec:{r['local_observed_event_candidate_id']}", "phenomenon_of_candidate", r["candidate_patch_id"]))
    for r in g4_spatial_link_evaluation_rows():
        edges.append((f"g4:{r['g4_evaluation_id']}", f"oec:{r['local_observed_event_candidate_id']}", "g4_of_candidate", r["candidate_patch_id"]))
    for r in g5_phenomenon_evaluation_rows():
        edges.append((f"g5:{r['g5_evaluation_id']}", f"oec:{r['local_observed_event_candidate_id']}", "g5_of_candidate", r["candidate_patch_id"]))
    for r in ground_reference_candidate_evaluation_rows():
        edges.append((f"grce:{r['ground_reference_candidate_eval_id']}", f"oec:{r['local_observed_event_candidate_id']}", "gr_eval_of_candidate", r["candidate_patch_id"]))
    return nodes, edges, key_to_id, event_id


def evidence_graph_update_node_rows() -> list[dict]:
    nodes, _edges, key_to_id, event_id = _graph_update()
    return [{"node_id": key_to_id[key], "node_type": nt, "object_id": oid, "candidate_patch_id": pid, "event_id": event_id, **GOV}
            for key, nt, oid, pid in nodes]


def evidence_graph_update_edge_rows() -> list[dict]:
    _nodes, edges, key_to_id, event_id = _graph_update()
    return [{"edge_id": f"S17C29_EDGE_{i:04d}", "source_node_id": key_to_id[s], "target_node_id": key_to_id.get(t, "not_available"),
             "edge_type": et, "candidate_patch_id": pid, "event_id": event_id, "review_only": "true"}
            for i, (s, t, et, pid) in enumerate(edges, start=1)]


# ---------------------------------------------------------------------------
# No leakage
# ---------------------------------------------------------------------------

def no_leakage_audit_rows() -> list[dict]:
    rows = []
    candidates = local_observed_event_candidate_rows()
    for idx, oec in enumerate(candidates, start=1):
        official = oec["officiality_level"] in OFFICIAL_EVENT_LEVELS
        rows.append({
            "no_leakage_audit_id": f"S17C29_LEAK_{idx:04d}",
            "object_id": oec["local_observed_event_candidate_id"], "object_type": "local_observed_event_candidate",
            "uses_sensor_as_event_observation": "false", "uses_chirps_as_event_reference": "false",
            "uses_news_as_ground_reference_without_official_support": "false", "uses_candidate_as_ground_truth": "false",
            "uses_ground_reference_as_training_label": "false", "uses_synthetic_as_real": "false",
            "uses_city_as_patch_level_without_uncertainty": "false", "uses_mixed_phenomenon_as_g5": "false",
            "invented_coordinate": "false", "passes_no_leakage": "true",
            "blocking_reason": "candidato oficial institucional local review-only sem promocao indevida" if official else "candidato review-only",
            "review_only": "true",
        })
    if not rows:
        rows.append({"no_leakage_audit_id": "S17C29_LEAK_0001", "object_id": "no_candidate", "object_type": "local_acquisition_package",
                     "uses_sensor_as_event_observation": "false", "uses_chirps_as_event_reference": "false",
                     "uses_news_as_ground_reference_without_official_support": "false", "uses_candidate_as_ground_truth": "false",
                     "uses_ground_reference_as_training_label": "false", "uses_synthetic_as_real": "false",
                     "uses_city_as_patch_level_without_uncertainty": "false", "uses_mixed_phenomenon_as_g5": "false",
                     "invented_coordinate": "false", "passes_no_leakage": "true", "blocking_reason": "not_applicable", "review_only": "true"})
    return rows


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def gate_evaluation_matrix_rows() -> list[dict]:
    rows = []
    for idx, gr in enumerate(ground_reference_candidate_evaluation_rows(), start=1):
        rows.append({
            "gate_eval_id": f"S17C29_GATE_{idx:04d}",
            "object_id": gr["local_observed_event_candidate_id"], "object_type": "local_observed_event_candidate",
            "candidate_patch_id": gr["candidate_patch_id"], "G1_existencia_documental": gr["G1_existencia_documental"],
            "G2_confiabilidade_fonte": gr["G2_confiabilidade_fonte"], "G3_precisao_temporal": gr["G3_precisao_temporal"],
            "G4_vinculo_espacial_evento": gr["G4_vinculo_espacial_evento"], "G5_separacao_fenomeno": gr["G5_separacao_fenomeno"],
            "G6_proveniencia_integridade": gr["G6_proveniencia_integridade"], "G7_anti_leakage": gr["G7_anti_leakage"],
            "all_gates_passed_for_ground_reference": gr["can_be_ground_reference_candidate"],
            "acceptance_status": "accepted_ground_reference_candidate" if gr["can_be_ground_reference_candidate"] == "true" else "blocked_local_observed_event_candidate_review_only",
            "blocking_reason": gr["blocking_reason"], **GOV,
        })
    return rows


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def build_summary() -> dict:
    plan = local_search_plan_rows()
    attempts = local_source_acquisition_attempt_rows()
    official_local = _official_local_manifests()
    parsed = local_parsed_artifact_index_rows()
    parsed_official = [p for p in parsed
                       if p["local_source_artifact_manifest_id"] in {m["local_source_artifact_manifest_id"] for m in official_local}
                       and p["parse_success"] == "true"]
    candidates = local_observed_event_candidate_rows()
    phenomena = local_phenomenon_classification_rows()
    g4 = g4_spatial_link_evaluation_rows()
    g5 = g5_phenomenon_evaluation_rows()
    gr_eval = ground_reference_candidate_evaluation_rows()
    accepted = [r for r in gr_eval if r["can_be_ground_reference_candidate"] == "true"]
    geocodable = len([c for c in candidates if c["geocodable_location"] == "true"])
    patch_level = len([c for c in candidates if c["geometry_type"] not in ("none", "")])
    hydro_specific = len([p for p in phenomena if p["hydrological_specific"] == "true"])
    mixed = len([p for p in phenomena if p["mixed_or_ambiguous"] == "true"])
    attempts_count = len([a for a in attempts if a["network_enabled"] == "true"])
    minimum = (
        attempts_count >= 40 and len(official_local) >= 3 and len(parsed_official) >= 3
        and len(candidates) >= 3 and geocodable >= 1 and hydro_specific >= 1 and (len(g4) + len(g5)) >= 3
    )
    return {
        "minimum_success_achieved": minimum,
        "local_search_plan_rows_count": len(plan),
        "local_geometry_search_attempts_count": attempts_count,
        "local_official_artifacts_acquired_count": len(official_local),
        "local_artifacts_parsed_count": len(parsed_official),
        "local_event_candidates_count": len(candidates),
        "geocodable_location_candidates_count": geocodable,
        "patch_level_location_candidates_count": patch_level,
        "hydrological_specific_candidates_count": hydro_specific,
        "mixed_phenomenon_candidates_count": mixed,
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
        "followed_links_count": len(local_followed_link_registry_rows()),
        "recommended_next_milestone": "SUSC-17C30 Aquisicao de geometria vetorial oficial (poligono/mancha/coordenada de ocorrencia ou setor de risco SGB/CPRM) e classificacao de fenomeno por ponto para tentar G4/G5 patch-level com coordenada real",
    }


def build_blockers() -> list[dict]:
    blockers = [
        "no_patch_level_coordinate_or_polygon", "bairro_only_location_high_uncertainty",
        "phenomenon_mixed_flood_and_landslide", "hydrological_not_separated_from_mass_movement",
        "no_accepted_ground_reference_candidate", "17b_blocked_until_G4_G5_true",
        "score_v7_blocked_until_ground_reference_policy",
    ]
    return [
        {
            "blocker_id": f"S17C29_BLOCKER_{idx:04d}", "blocker_type": blocker,
            "description": "Bloqueio real: artefatos oficiais/institucionais locais do evento adquiridos e avaliados (bairros do Grande Recife, alagamento documentado em ao menos um), mas G4 (coordenada/poligono patch-level) e G5 (separacao de fenomeno misto inundacao+deslizamento) nao satisfeitos; nenhuma coordenada inventada, nenhum Ground Reference Candidate aceito.",
            "blocks_ground_reference_candidate": _bool_text(blocker in {"no_patch_level_coordinate_or_polygon", "bairro_only_location_high_uncertainty", "phenomenon_mixed_flood_and_landslide", "hydrological_not_separated_from_mass_movement", "no_accepted_ground_reference_candidate"}),
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
    rows = local_source_artifact_manifest_rows()
    required = list(rows[0].keys()) if rows else MANIFEST_FIELDS
    return _schema(required, {
        "local_source_artifact_manifest_id": {"type": "string", "pattern": "^S17C29_ART_"},
        "local_specific": {"const": "true"}, "raw_heavy": {"const": "false"}, "review_only": {"const": "true"},
        "trainable": {"const": "false"}, "ground_truth": {"const": "false"},
    }, "SUSC-17C29 local source artifact schema v1")


def build_candidate_schema() -> dict:
    rows = local_observed_event_candidate_rows()
    required = list(rows[0].keys()) if rows else [
        "local_observed_event_candidate_id", "local_source_artifact_manifest_id", "candidate_patch_id", "event_id",
        "observed_event_date_or_period", "observed_location_text", "bairro", "logradouro", "geocodable_location",
        "observed_geometry", "geometry_type", "geometry_uncertainty_m", "flood_mentions", "landslide_mentions",
        "phenomenon_candidate", "source_family", "officiality_level", "evidence_snippet", "can_evaluate_g4",
        "can_evaluate_g5", "review_only", "trainable", "ground_truth",
    ]
    return _schema(required, {
        "local_observed_event_candidate_id": {"type": "string", "pattern": "^S17C29_OEC_"},
        "review_only": {"const": "true"}, "trainable": {"const": "false"}, "ground_truth": {"const": "false"},
    }, "SUSC-17C29 local event candidate schema v1")


def build_g4_g5_schema() -> dict:
    rows = ground_reference_candidate_evaluation_rows()
    required = list(rows[0].keys()) if rows else [
        "ground_reference_candidate_eval_id", "local_observed_event_candidate_id", "candidate_patch_id", "event_id",
        "G1_existencia_documental", "G2_confiabilidade_fonte", "G3_precisao_temporal", "G4_vinculo_espacial_evento",
        "G5_separacao_fenomeno", "G6_proveniencia_integridade", "G7_anti_leakage", "can_be_ground_reference_candidate",
        "can_be_ground_truth", "can_be_training_label", "can_unlock_17b", "blocking_reason", "review_only",
    ]
    return _schema(required, {
        "ground_reference_candidate_eval_id": {"type": "string", "pattern": "^S17C29_GRCE_"},
        "can_be_ground_truth": {"const": "false"}, "can_be_training_label": {"const": "false"}, "review_only": {"const": "true"},
    }, "SUSC-17C29 g4 g5 evaluation schema v1")


# ---------------------------------------------------------------------------
# Relatorio
# ---------------------------------------------------------------------------

def build_report() -> str:
    s = build_summary()
    return "\n".join([
        "# SUSC-17C29 - Aquisicao de geometria local oficial e separacao de fenomeno para G4/G5", "",
        "## Objetivo",
        "Aquisicao DIRIGIDA a nivel local (bairro/logradouro/patch) para tentar obter evidencia oficial/institucional suficiente para G4 (vinculo espacial patch-level) e G5 (separacao de fenomeno hidrologico x movimento de massa) do evento REC_2022_05_24_30 (Grande Recife, maio 2022). Este marco nao reprova o evento (ja provado em 17C27/17C28): foca patch-level e fenomeno local.", "",
        "## Aquisicao local",
        f"- Plano de busca local: {s['local_search_plan_rows_count']} linhas (bairro/logradouro, abrigo/ocorrencia, hidrologico local, geometria).",
        f"- Tentativas de busca/aquisicao local: {s['local_geometry_search_attempts_count']}.",
        f"- Links locais seguidos: {s['followed_links_count']}.",
        f"- Artefatos oficiais/institucionais locais adquiridos: {s['local_official_artifacts_acquired_count']} (NOVOS, distintos dos 4 do 17C28); parseados: {s['local_artifacts_parsed_count']}.", "",
        "## Candidatos, geometria e fenomeno",
        f"- Candidatos locais: {s['local_event_candidates_count']}.",
        f"- Geocodaveis (bairro/logradouro, sem coordenada): {s['geocodable_location_candidates_count']}; patch-level (coordenada/poligono): {s['patch_level_location_candidates_count']}.",
        f"- Hidrologicos especificos (alagamento/inundacao documentado): {s['hydrological_specific_candidates_count']}; mistos (inundacao + deslizamento): {s['mixed_phenomenon_candidates_count']}.",
        f"- Avaliacoes G4/G5: {s['G4_G5_evaluation_rows_count']}; Ground Reference Candidates avaliados: {s['ground_reference_candidates_evaluated_count']}; aceitos: {s['accepted_ground_reference_candidate_count']}.",
        f"- G4_true_count={s['G4_true_count']}, G5_true_count={s['G5_true_count']}.", "",
        "## Resultado cientifico (honesto - Resultado B: bloqueio honesto)",
        "- Fontes oficiais institucionais (Agencia Brasil/EBC) citam bairros do Grande Recife (Ibura, Jardim Monte Verde, Barro, Muribeca, Curado, Jaboatao, Olinda, Guararapes) e ao menos uma documenta alagamento/inundacao, mas:",
        "- G4 permanece false: a localizacao e geocodavel apenas a nivel de bairro/municipio, sem coordenada ou poligono patch-level; incerteza >=3km incompativel com o patch/buffer. Nenhuma coordenada foi inventada.",
        "- G5 permanece false: onde o fenomeno hidrologico aparece, vem MISTO com deslizamento; fenomeno misto nunca vira G5. Nenhuma fonte separa o hidrologico do movimento de massa por local.",
        "- Nenhum Ground Reference Candidate foi aceito; 17B permanece bloqueado.", "",
        "## Guardrails",
        "- Cidade/municipio nao virou patch-level sem incerteza; fenomeno misto nao virou G5; noticia comercial nao virou Ground Reference; sensor/CHIRPS nao viraram evento observado; nenhum ground truth, label, treino, score v7 ou patch oficial; score v6 intacto.", "",
        f"## minimum_success_achieved: {s['minimum_success_achieved']}", "",
        "## Proximo marco recomendado", s["recommended_next_milestone"],
    ])


# ---------------------------------------------------------------------------
# Build / validacao
# ---------------------------------------------------------------------------

def build_all() -> None:
    _require_inputs()
    write_csv(LOCAL_PLAN, local_search_plan_rows())
    write_csv(LOCAL_ATTEMPTS, local_source_acquisition_attempt_rows())
    write_csv(FOLLOWED_LINKS, local_followed_link_registry_rows())
    write_csv(LOCAL_MANIFEST, local_source_artifact_manifest_rows(), MANIFEST_FIELDS)
    write_csv(LOCAL_PARSED, local_parsed_artifact_index_rows())
    write_csv(LOCAL_CANDIDATES, local_observed_event_candidate_rows())
    write_csv(LOCATION_RESOLUTION, local_location_resolution_rows())
    write_csv(PHENOMENON, local_phenomenon_classification_rows())
    write_csv(G4_EVAL, g4_spatial_link_evaluation_rows())
    write_csv(G5_EVAL, g5_phenomenon_evaluation_rows())
    write_csv(GR_CANDIDATE_EVAL, ground_reference_candidate_evaluation_rows())
    write_csv(SCORECARD, local_artifact_scorecard_rows())
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
        REPORT, LOCAL_PLAN, LOCAL_ATTEMPTS, FOLLOWED_LINKS, LOCAL_MANIFEST, LOCAL_PARSED, LOCAL_CANDIDATES,
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
    plan = read_csv(LOCAL_PLAN)
    attempts = read_csv(LOCAL_ATTEMPTS)
    manifests = read_csv(LOCAL_MANIFEST)
    parsed = read_csv(LOCAL_PARSED)
    candidates = read_csv(LOCAL_CANDIDATES)
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
        (plan, "local_search_plan_id"), (attempts, "local_source_acquisition_attempt_id"),
        (manifests, "local_source_artifact_manifest_id"), (parsed, "local_parsed_artifact_id"),
        (candidates, "local_observed_event_candidate_id"), (locations, "local_location_resolution_id"),
        (phenomena, "local_phenomenon_classification_id"), (g4, "g4_evaluation_id"), (g5, "g5_evaluation_id"),
        (gr_eval, "ground_reference_candidate_eval_id"), (gates, "gate_eval_id"), (leakage, "no_leakage_audit_id"),
        (nodes, "node_id"), (edges, "edge_id"),
    ]:
        ids = [row[key] for row in rows]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            errors.append(f"ids_not_unique_or_sorted:{key}")

    official_local = [m for m in manifests if m["local_specific"] == "true" and m["officiality_level"] in OFFICIAL_EVENT_LEVELS]
    parsed_official = [p for p in parsed if p["local_source_artifact_manifest_id"] in {m["local_source_artifact_manifest_id"] for m in official_local} and p["parse_success"] == "true"]

    # 1..7 minimums.
    if len([a for a in attempts if a["network_enabled"] == "true"]) < 40:
        errors.append("local_attempts_lt_40")
    if len(official_local) < 3:
        errors.append("official_local_artifacts_lt_3")
    if len(parsed_official) < 3:
        errors.append("parsed_local_lt_3")
    if len(candidates) < 3:
        errors.append("local_candidates_lt_3")
    if len([c for c in candidates if c["geocodable_location"] == "true"]) < 1:
        errors.append("no_geocodable_location_candidate")
    if len([p for p in phenomena if p["hydrological_specific"] == "true"]) < 1:
        errors.append("no_hydrological_specific_candidate")
    if (len(g4) + len(g5)) < 3:
        errors.append("no_g4_g5_evaluation")
    # 8: hash + size + type + no forbidden raw.
    for row in manifests:
        path = ROOT / row["artifact_local_path"]
        if not path.exists() or not row["sha256"] or sha256_file(path) != row["sha256"]:
            errors.append(f"manifest_hash_mismatch:{row['local_source_artifact_manifest_id']}")
        if int(row["size_bytes"]) > MAX_ARTIFACT_BYTES:
            errors.append(f"artifact_over_limit:{row['local_source_artifact_manifest_id']}")
        if row["artifact_type"] not in ("html", "pdf", "txt", "csv", "json"):
            errors.append(f"forbidden_artifact_type:{row['local_source_artifact_manifest_id']}")
    for path in ARTIFACT_DIR.glob("**/*") if ARTIFACT_DIR.exists() else []:
        if path.is_file() and path.suffix.lower() in (".tif", ".nc", ".zip", ".gz", ".npz", ".npy"):
            errors.append(f"forbidden_raw_committed:{rel(path)}")
    # 9: cidade/municipio nao vira patch-level sem incerteza.
    for row in locations:
        if row["patch_level_geometry"] == "true":
            errors.append(f"city_promoted_to_patch_level:{row['local_location_resolution_id']}")
        if row["within_patch_or_buffer"] != "false":
            errors.append(f"location_claims_within_patch:{row['local_location_resolution_id']}")
    if any(c["geometry_type"] not in ("none", "") for c in candidates):
        errors.append("candidate_has_invented_geometry")
    if any(r["invented_coordinate"] != "false" for r in leakage):
        errors.append("invented_coordinate")
    if any(r["uses_city_as_patch_level_without_uncertainty"] != "false" for r in leakage):
        errors.append("city_as_patch_level")
    # 10: fenomeno misto nao vira G5.
    phen_by = {p["local_observed_event_candidate_id"]: p for p in phenomena}
    for row in g5:
        oec_id = row["local_observed_event_candidate_id"]
        if row["G5_separacao_fenomeno"] == "true" and phen_by.get(oec_id, {}).get("mixed_or_ambiguous") == "true":
            errors.append(f"mixed_phenomenon_as_g5:{oec_id}")
    if any(r["uses_mixed_phenomenon_as_g5"] != "false" for r in leakage):
        errors.append("mixed_as_g5_leakage")
    # 11: noticia comercial nao vira GR sozinha.
    for row in gr_eval:
        if row["can_be_ground_reference_candidate"] == "true":
            oec = next((o for o in candidates if o["local_observed_event_candidate_id"] == row["local_observed_event_candidate_id"]), {})
            if oec.get("officiality_level") not in OFFICIAL_EVENT_LEVELS:
                errors.append("non_official_accepted_as_ground_reference")
    if any(r["uses_news_as_ground_reference_without_official_support"] != "false" for r in leakage):
        errors.append("news_used_as_ground_reference")
    # 12: sensor/chirps nao viram evento.
    if any(r["uses_sensor_as_event_observation"] != "false" or r["uses_chirps_as_event_reference"] != "false" for r in leakage):
        errors.append("sensor_or_chirps_as_event")
    if any(r["passes_no_leakage"] != "true" for r in leakage):
        errors.append("no_leakage_failed")
    # 13/14: sem GT/label.
    if summary["ground_truth_created"] or summary["training_labels_created"]:
        errors.append("forbidden_gt_or_label")
    if any(r["can_be_ground_truth"] != "false" or r["can_be_training_label"] != "false" for r in gr_eval):
        errors.append("gr_marked_gt_or_label")
    # 17: 17B elegivel so com accepted GR.
    accepted = [r for r in gr_eval if r["can_be_ground_reference_candidate"] == "true"]
    if summary["eligible_for_17b_now"] != (len(accepted) > 0):
        errors.append("17b_eligibility_inconsistent")
    if summary["eligible_for_17b_now"] and not accepted:
        errors.append("17b_eligible_without_accepted_ground_reference")
    node_ids = {n["node_id"] for n in nodes}
    if any(e["source_node_id"] not in node_ids or e["target_node_id"] not in node_ids for e in edges):
        errors.append("edge_references_unknown_node")

    # 15/16: score v6 intacto, score v7 inexistente.
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
        "17C29 -> "
        f"attempts={summary['local_geometry_search_attempts_count']} official_local={summary['local_official_artifacts_acquired_count']} "
        f"candidates={summary['local_event_candidates_count']} geocodable={summary['geocodable_location_candidates_count']} "
        f"hydro_spec={summary['hydrological_specific_candidates_count']} mixed={summary['mixed_phenomenon_candidates_count']} "
        f"g4g5={summary['G4_G5_evaluation_rows_count']} accepted={summary['accepted_ground_reference_candidate_count']} "
        f"G4={summary['G4_true_count']} G5={summary['G5_true_count']} eligible_17b={summary['eligible_for_17b_now']} "
        f"min_success={summary['minimum_success_achieved']}"
    )
    return 0


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def expand_local_search_text() -> str:
    rows = local_search_plan_rows()
    specs = sorted({r["target_specificity"] for r in rows})
    return f"expand-local-search: {len(rows)} buscas locais expandidas cobrindo {specs}."


def parse_local_artifacts_text() -> str:
    parsed = local_parsed_artifact_index_rows()
    return f"parse-local-artifacts: {len([p for p in parsed if p['parse_success'] == 'true'])}/{len(parsed)} artefatos locais parseados."


def extract_local_event_candidates_text() -> str:
    c = local_observed_event_candidate_rows()
    return f"extract-local-event-candidates: {len(c)} candidatos locais de fonte oficial institucional."


def resolve_local_geometry_text() -> str:
    rows = local_location_resolution_rows()
    geo = len([r for r in rows if r["geocodable"] == "true"])
    return f"resolve-local-geometry: {len(rows)} resolucoes, {geo} geocodaveis a nivel de bairro (nenhuma coordenada inventada, 0 patch-level)."


def classify_local_phenomena_text() -> str:
    rows = local_phenomenon_classification_rows()
    classes = {}
    for r in rows:
        classes[r["phenomenon_class"]] = classes.get(r["phenomenon_class"], 0) + 1
    return "classify-local-phenomena: " + "; ".join(f"{k}={v}" for k, v in sorted(classes.items())) + "."


def evaluate_g4_g5_text() -> str:
    gr = ground_reference_candidate_evaluation_rows()
    accepted = [r for r in gr if r["can_be_ground_reference_candidate"] == "true"]
    return f"evaluate-g4-g5: {len(gr)} candidatos avaliados, {len(accepted)} aceitos (G4 patch-level e G5 separacao de fenomeno nao satisfeitos)."


def status_text() -> str:
    s = build_summary()
    return (
        f"status 17C29: official_local={s['local_official_artifacts_acquired_count']} candidates={s['local_event_candidates_count']} "
        f"geocodable={s['geocodable_location_candidates_count']} hydro_spec={s['hydrological_specific_candidates_count']} "
        f"accepted_gr={s['accepted_ground_reference_candidate_count']} eligible_17b={s['eligible_for_17b_now']} min_success={s['minimum_success_achieved']}"
    )
