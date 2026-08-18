"""
SUSC-13C-LIVE real network healthcheck.

Performs real, polite HEAD/GET requests against a fixed list of official domains
to prove whether live acquisition is possible in this environment. If fewer than
3 official domains respond, writes an explicit blocked-network report so the
sprint stops instead of committing another offline-only run.

Writes:
  - outputs_public/suscetibilidade/SUSC_13C_live_network_healthcheck.json
  - outputs_public/suscetibilidade/SUSC_13C_live_network_healthcheck.md
  - outputs_public/suscetibilidade/SUSC_13C_LIVE_BLOCKED_NETWORK_REPORT.md (only if blocked)
"""

from __future__ import annotations

import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_io import write_json, write_markdown  # noqa: E402

OUT = ROOT / "outputs_public" / "suscetibilidade"
HEALTH_JSON = OUT / "SUSC_13C_live_network_healthcheck.json"
HEALTH_MD = OUT / "SUSC_13C_live_network_healthcheck.md"
BLOCKED_REPORT = OUT / "SUSC_13C_LIVE_BLOCKED_NETWORK_REPORT.md"

USER_AGENT = "REV-P-SUSC-13C/1.0 (review-only research; respects robots.txt)"
TIMEOUT = 15
MIN_REACHABLE = 3

DOMAINS = [
    "https://dados.gov.br/",
    "https://www.gov.br/mdr/pt-br",
    "https://www.sgb.gov.br/",
    "https://www.cemaden.gov.br/",
    "https://www.ana.gov.br/",
    "https://dados.recife.pe.gov.br/",
    "https://www.apac.pe.gov.br/",
    "https://www.recife.pe.gov.br/",
    "https://www.petropolis.rj.gov.br/",
    "https://www.inea.rj.gov.br/",
    "https://www.curitiba.pr.gov.br/",
    "https://geocuritiba.ippuc.org.br/",
    "https://www.defesacivil.pr.gov.br/",
]


def _check(url: str) -> dict:
    rec = {
        "url": url, "status_code": "", "content_type": "", "content_length": "",
        "reachable": False, "error": "", "elapsed_ms": "",
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            rec["status_code"] = getattr(resp, "status", 200) or 200
            rec["content_type"] = resp.headers.get("Content-Type", "")
            rec["content_length"] = resp.headers.get("Content-Length", "") or ""
            resp.read(2048)  # touch the stream lightly
            rec["reachable"] = int(rec["status_code"]) < 400
    except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
        rec["status_code"] = e.code
        rec["error"] = f"HTTPError {e.code}"
        rec["reachable"] = e.code < 500  # server answered, host reachable
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {str(e)[:140]}"
    rec["elapsed_ms"] = int((time.time() - t0) * 1000)
    return rec


def main() -> int:
    print("=" * 60)
    print("SUSC-13C-LIVE Network Healthcheck")
    print("=" * 60)
    results = [_check(u) for u in DOMAINS]
    reachable = [r for r in results if r["reachable"]]
    n_reach = len(reachable)
    network_ok = n_reach >= MIN_REACHABLE

    summary = {
        "artifact": "SUSC-13C-LIVE network healthcheck",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "n_domains": len(DOMAINS), "n_reachable": n_reach,
        "min_reachable_required": MIN_REACHABLE, "network_ok": network_ok,
        "reachable_domains": [r["url"] for r in reachable],
        "blocked_domains": [r["url"] for r in results if not r["reachable"]],
        "results": results, "review_only": True,
    }
    write_json(HEALTH_JSON, summary)

    def _row(r):
        return f"| {r['url']} | {r['status_code']} | {r['content_type'][:30]} | {r['reachable']} | {r['elapsed_ms']} | {r['error'][:40]} |"

    md = f"""# SUSC-13C-LIVE - Healthcheck de rede

Status: **review-only**

- Verificado em: `{summary['checked_at']}`
- Dominios testados: **{len(DOMAINS)}**
- Dominios alcancados: **{n_reach}**
- Minimo exigido: **{MIN_REACHABLE}**
- Rede utilizavel: **{"SIM" if network_ok else "NAO"}**

| url | status | content-type | reachable | ms | erro |
|---|---|---|---|---|---|
{chr(10).join(_row(r) for r in results)}

{"Rede real disponivel: a sprint de aquisicao live pode prosseguir." if network_ok else "Rede real BLOQUEADA: menos de 3 dominios oficiais responderam. A sprint para aqui."}
"""
    write_markdown(HEALTH_MD, md)

    if not network_ok:
        blocked = f"""# SUSC-13C-LIVE - RELATORIO DE BLOQUEIO DE REDE

Status: **review-only** | **rede real bloqueada neste ambiente**

O healthcheck alcancou apenas **{n_reach}** de {len(DOMAINS)} dominios oficiais
(minimo exigido: {MIN_REACHABLE}). Conforme a regra 22 do SUSC-13C-LIVE, a sprint
para aqui: nao foi possivel executar aquisicao online real e **nenhum dado foi
fabricado**.

## Dominios alcancados
{chr(10).join("- " + u for u in summary['reachable_domains']) or "- nenhum"}

## Dominios bloqueados
{chr(10).join("- " + u for u in summary['blocked_domains'])}

## Conclusao
Reexecutar em ambiente com rede liberada para os dominios oficiais. Nenhum commit
de aquisicao deve ser feito como se fosse online quando a rede esta bloqueada.
"""
        write_markdown(BLOCKED_REPORT, blocked)
    elif BLOCKED_REPORT.exists():
        # network recovered: keep a truthful, non-blocking note
        write_markdown(BLOCKED_REPORT,
                       "# SUSC-13C-LIVE - Bloqueio de rede NAO aplicavel\n\n"
                       f"Healthcheck alcancou {n_reach} dominios (>= {MIN_REACHABLE}); "
                       "rede real disponivel. Este arquivo permanece como marcador "
                       "historico e nao indica bloqueio.\n")

    print(f"reachable: {n_reach}/{len(DOMAINS)} | network_ok={network_ok}")
    for r in results:
        print(f"  {r['status_code'] or 'ERR':>4} {r['url']} {r['error'][:60]}")
    return 0 if network_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
