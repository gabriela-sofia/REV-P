"""
ACQ-RECIFE-01 -- Baixa o fluxo de atendimentos da Defesa Civil do Recife.

POR QUE ESTA AQUISICAO EXISTE:

o `neg01` tirou 114 negativos de Curitiba da lacuna usando a base bruta do
SIAC 156: com o fluxo completo de chamados da cidade da para checar se o canal
estava ativo naquele dia (N1) e se a cidade registrou alagamento em algum lugar
(N2). Sem esse fluxo, os dois criterios nao tem como ser avaliados.

Recife nao tinha o equivalente no repositorio -- so os 278 pontos ja
selecionados do v12 -- e por isso os 124 negativos continuavam em `ausencia`,
deixando o MVP anunciado apoiado num modelo de positivo contra nao-rotulado.

A FONTE, e por que e ela e nao a outra:

o portal de dados abertos do Recife tem dois candidatos. O `156` da EMLURB e o
analogo mais obvio do 156 de Curitiba, mas a EMLURB e a empresa de manutencao e
limpeza urbana: o canal dela recebe arborizacao, pavimentacao, tapa-buraco.

O `registro-de-atendimentos-da-defesa-civil` e da **Secretaria Executiva de
Defesa Civil** -- a MESMA fonte que produziu os positivos do v12. Isso importa
para o N2: a pergunta e se o canal que registraria a inundacao naquele ponto
estava recebendo e classificando inundacao em outros lugares naquele dia. Um
canal diferente responderia outra pergunta.

COBERTURA: anuais de 2014 a 2025, e os negativos de Recife vao de 2014-03-24 a
2025-01-23. O arquivo de 2025 tem 340 bytes -- esta praticamente vazio na
origem, e isso vai limitar os 14 negativos daquele ano. Declarado aqui, nao
descoberto depois.

O QUE ESTE SCRIPT FAZ E NAO FAZ: baixa e registra proveniencia -- URL, tamanho,
sha256 e status HTTP de cada arquivo. Nao interpreta, nao filtra, nao decide
nada sobre negativo. A leitura e do passo seguinte, e separar as duas coisas e
o que permite reconferir o download sem reprocessar a analise.

FALHA FECHADA: erro de rede num arquivo nao interrompe os demais, mas entra no
manifesto com o status. Um arquivo faltando aparece como faltando, e nao como
ano sem dado.

Uso:
    python scripts/suscetibilidade/acq_recife01_sedec_atendimentos.py --listar
    python scripts/suscetibilidade/acq_recife01_sedec_atendimentos.py --baixar
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

import requests

REPO = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
OUT = REPO / "local_runs" / "acq-recife-01-sedec"
RAW = OUT / "raw"

PORTAL = "http://dados.recife.pe.gov.br"
PACOTE = "registro-de-atendimentos-da-defesa-civil"
# so os recursos de atendimento anual; o pacote tambem traz areas de risco,
# vistorias e lonas, que sao outra coisa e nao entram aqui
PADRAO_NOME = re.compile(r"atendimentos?\s*(20\d{2})", re.IGNORECASE)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def descobrir() -> list[dict]:
    r = requests.get(f"{PORTAL}/api/3/action/package_show",
                     params={"id": PACOTE}, timeout=60)
    r.raise_for_status()
    pac = r.json()["result"]
    org = (pac.get("organization") or {}).get("title", "")
    alvos, vistos = [], set()
    for x in pac.get("resources", []):
        nome = str(x.get("name") or "")
        m = PADRAO_NOME.search(nome)
        if not m or str(x.get("format", "")).upper() != "CSV":
            continue
        ano = m.group(1)
        if ano in vistos:      # o portal repete recurso; fica o primeiro
            continue
        vistos.add(ano)
        alvos.append({"ano": ano, "nome": nome, "url": x.get("url"),
                      "tamanho_declarado": x.get("size"),
                      "id": x.get("id"), "organizacao": org})
    return sorted(alvos, key=lambda a: a["ano"])


def baixar(a: dict) -> dict:
    destino = RAW / f"sedec_atendimentos_{a['ano']}.csv"
    if destino.exists() and destino.stat().st_size > 0:
        return {**a, "status": "ja_baixado", "bytes": destino.stat().st_size,
                "sha256": sha256(destino), "arquivo": destino.name}
    try:
        with requests.get(a["url"], timeout=300, stream=True) as r:
            r.raise_for_status()
            parcial = destino.with_suffix(".parcial")
            with parcial.open("wb") as f:
                for pedaco in r.iter_content(1 << 20):
                    f.write(pedaco)
            parcial.replace(destino)
        return {**a, "status": "baixado", "bytes": destino.stat().st_size,
                "sha256": sha256(destino), "arquivo": destino.name}
    except Exception as exc:  # noqa: BLE001
        return {**a, "status": "falha", "erro": f"{type(exc).__name__}: {exc}",
                "bytes": 0, "sha256": None, "arquivo": None}


def main() -> int:
    args = sys.argv[1:]
    OUT.mkdir(parents=True, exist_ok=True)
    print("===ACQ_RECIFE01_SEDEC_ATENDIMENTOS===")
    print(f"portal={PORTAL} pacote={PACOTE}")

    alvos = descobrir()
    if not alvos:
        print("ABORTADO: nenhum recurso de atendimento anual encontrado. "
              "O pacote mudou de estrutura -- conferir antes de seguir.")
        return 1
    org = alvos[0]["organizacao"]
    total = sum(int(a["tamanho_declarado"] or 0) for a in alvos)
    print(f"organizacao: {org.strip()}")
    print(f"{len(alvos)} recursos anuais, {total/1e6:.0f} MB declarados\n")
    for a in alvos:
        print(f"   {a['ano']}  {int(a['tamanho_declarado'] or 0)/1e6:>7.1f} MB  "
              f"{a['nome'][:44]}")

    if "--baixar" not in args:
        print("\n(--listar) nada foi baixado. Use --baixar para executar.")
        return 0

    RAW.mkdir(parents=True, exist_ok=True)
    print(f"\n--- baixando para {RAW} ---")
    t0 = time.time()
    reg = []
    for a in alvos:
        r = baixar(a)
        reg.append(r)
        marca = {"baixado": "OK", "ja_baixado": "cache", "falha": "FALHA"}[r["status"]]
        print(f"   {a['ano']}  {marca:6s} {r['bytes']/1e6:>7.1f} MB"
              + (f"  {r['erro'][:60]}" if r["status"] == "falha" else ""), flush=True)

    ok = [r for r in reg if r["status"] != "falha"]
    falhas = [r for r in reg if r["status"] == "falha"]
    print(f"\nBAIXADOS={len(ok)}/{len(reg)} "
          f"({sum(r['bytes'] for r in ok)/1e6:.0f} MB) "
          f"em {time.time()-t0:.0f}s")
    if falhas:
        print(f"FALHAS ({len(falhas)}): {[r['ano'] for r in falhas]}")

    # arquivo minusculo nao e erro de rede, e vazio na origem -- precisa
    # aparecer como tal para nao virar 'ano sem inundacao' depois
    vazios = [r for r in ok if r["bytes"] < 10_000]
    if vazios:
        print(f"QUASE VAZIOS na origem ({len(vazios)}): "
              f"{[(r['ano'], r['bytes']) for r in vazios]}")

    (OUT / "manifesto_download.json").write_text(json.dumps({
        "portal": PORTAL, "pacote": PACOTE, "organizacao": org.strip(),
        "baixado_em": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "recursos": reg,
        "anos_ok": sorted(r["ano"] for r in ok),
        "anos_falha": sorted(r["ano"] for r in falhas),
        "anos_quase_vazios": sorted(r["ano"] for r in vazios),
        "por_que_esta_fonte": (
            "e a Secretaria Executiva de Defesa Civil, a MESMA que produziu os "
            "positivos do v12. O 156 da EMLURB e outro canal -- manutencao e "
            "limpeza urbana -- e responderia outra pergunta no criterio N2"),
        "nao_e": "download e proveniencia; nenhuma leitura ou decisao aqui",
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"GRAVADO={OUT}")
    print("===END===")
    return 0 if not falhas else 2


if __name__ == "__main__":
    raise SystemExit(main())
