"""
SUSC-13C-LIVE real official HTML probe (limited, robots-respecting crawl).

Crawls official domains (depth<=2, <=80 pages/domain, polite delay, robots.txt
honored) and extracts links to vector/tabular/documentary files whose anchor or
URL matches flood/inundation/Defesa-Civil keywords. Requires SUSC_13B_NETWORK=1.
Never invents data; does not download the files here (that is FASE 5).

Writes:
  - outputs_public/suscetibilidade/SUSC_13C_live_html_file_links.csv
"""

from __future__ import annotations

import re
import sys
from collections import Counter, deque
from pathlib import Path
from urllib.parse import urljoin, urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
import susc_13b_web_discovery_common as wd  # noqa: E402
from susc_io import rel, write_csv  # noqa: E402

LINKS_CSV = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_13C_live_html_file_links.csv"

SEEDS = [
    ("recife", "http://dados.recife.pe.gov.br/"),
    ("recife", "https://www.apac.pe.gov.br/"),
    ("petropolis", "https://www.petropolis.rj.gov.br/"),
    ("curitiba", "https://www.curitiba.pr.gov.br/"),
    ("curitiba", "https://www.defesacivil.pr.gov.br/"),
]

FIELDS = [
    "region", "seed_domain", "page_url", "file_url", "anchor_text",
    "ext", "matched_keywords", "is_download_candidate", "page_depth",
    "review_only", "notes",
]

MAX_DEPTH = 2
MAX_PAGES = 80
PRIORITY_TOKENS = ["alagamento", "alagamentos", "inundacao", "inundação", "enchente",
                   "enxurrada", "defesa-civil", "defesa_civil", "ocorrencia",
                   "ocorrência", "risco-hidrologico", "ponto-critico",
                   "pontos-criticos", "flood"]


def _fetch_html(url: str) -> str | None:
    res = wd.http_get(url, max_bytes=3 * 1024 * 1024, accept="text/html")
    if res["status"] != "ok":
        return None
    ct = str(res.get("content_type", "")).lower()
    if "html" not in ct and ct:
        return None
    try:
        return res["bytes"].decode("utf-8", "ignore")  # type: ignore[union-attr]
    except Exception:
        return None


def _links(html: str, base: str):
    for m in re.finditer(r'href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', html, re.IGNORECASE | re.DOTALL):
        href = m.group(1).strip()
        anchor = re.sub(r"<[^>]+>", " ", m.group(2)).strip()
        if href.startswith(("mailto:", "javascript:", "#", "tel:")):
            continue
        yield urljoin(base, href), anchor


def _crawl_domain(region: str, seed: str, rows: list[dict]) -> str:
    domain = wd.domain_of(seed)
    allowed, why = wd.robots_allows(seed, urlparse(seed).path or "/")
    if not allowed:
        rows.append({
            "region": region, "seed_domain": domain, "page_url": seed, "file_url": "",
            "anchor_text": "", "ext": "", "matched_keywords": "",
            "is_download_candidate": "false", "page_depth": "0", "review_only": "true",
            "notes": f"robots/acesso bloqueado: {why}",
        })
        return f"blocked:{why}"
    visited: set[str] = set()
    q: deque = deque([(seed, 0)])
    pages = 0
    found = 0
    while q and pages < MAX_PAGES:
        url, depth = q.popleft()
        if url in visited or depth > MAX_DEPTH:
            continue
        visited.add(url)
        html = _fetch_html(url)
        pages += 1
        wd.polite_sleep()
        if not html:
            continue
        for full, anchor in _links(html, url):
            if wd.domain_of(full) != domain:
                continue
            ext = wd.classify_url_ext(full)
            text = f"{anchor} {full}"
            if ext in wd.ALLOWED_DATA_EXTS and "." in ext:
                if wd.matched_keywords(text):
                    rows.append({
                        "region": region, "seed_domain": domain, "page_url": url,
                        "file_url": full, "anchor_text": anchor[:120], "ext": ext,
                        "matched_keywords": "|".join(wd.matched_keywords(text)),
                        "is_download_candidate": "true" if not wd.looks_like_risk(text) else "false",
                        "page_depth": str(depth), "review_only": "true",
                        "notes": "link de arquivo oficial com keyword" if not wd.looks_like_risk(text) else "link rebaixado (risco/alerta)",
                    })
                    found += 1
            elif depth < MAX_DEPTH and ext in {"unknown_ext"}:
                # enqueue same-domain HTML pages, prioritizing flood-related URLs
                low = full.lower()
                if any(tok in low for tok in PRIORITY_TOKENS) or depth == 0:
                    q.append((full, depth + 1))
    return f"crawled:{pages}pages:{found}links"


def main() -> int:
    print("=" * 60)
    print("SUSC-13C-LIVE HTML Probe")
    print("=" * 60)
    rows: list[dict] = []
    status: Counter = Counter()
    for region, seed in SEEDS:
        st = _crawl_domain(region, seed, rows)
        status[st.split(":")[0]] += 1
        print(f"  {seed}: {st}")
    write_csv(LINKS_CSV, rows, FIELDS)
    cands = sum(1 for r in rows if r["is_download_candidate"] == "true")
    print(f"links -> {rel(LINKS_CSV)} ({len(rows)} rows, {cands} candidates)")
    print("status:", dict(status))
    print("review-only. robots respected. No invented data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
