"""
NIVEL 1 / E -- Baixa os produtos vetoriais de ativacoes CEMS selecionadas.

POR QUE ESTE SCRIPT EXISTE, E POR QUE O ALVO MUDOU:
o inventario do N1C (corrigido em 2026-08-07) revelou duas coisas que
invalidaram o plano original e abriram um caminho melhor.

  1. A API publica so expoe EMSR656 a EMSR913 -- de 2023 em diante. As
     ativacoes historicas que eu havia citado para o noroeste da Inglaterra
     (EMSR147 Cumbria 2015, EMSR150 Yorkshire 2015) NAO estao nela. E nenhuma
     das 7 ativacoes do Reino Unido cai na AOI do piloto: sao Escocia,
     Irlanda e uma tempestade generica. Conclusao: o CEMS nao serve como fonte
     de negativo para a AOI inglesa. A hipotese E2 cai para essa regiao.

  2. Em compensacao apareceu isto:
         EMSR720  Flood in Rio Grande Do Sul State, Brazil  2024-05-02  5 AOIs
     Cinco areas de interesse analisadas, com mancha de inundacao observada
     dentro delas, para a enchente do RS de 2024.

Isso importa mais do que o alvo original. O gate `C4_BLOCKED_NO_FORMAL_NEGATIVES`
e sobre a AUSENCIA DE NEGATIVO NO DATASET BRASILEIRO. A frente inglesa foi
montada para contornar essa ausencia; o CEMS oferece atacar o problema
diretamente, no Brasil, com negativo por observacao:

    dentro da AOI declarada  E  fora do poligono de inundacao
        =  area analisada e nao inundada

DIFERENCA CRITICA em relacao ao Recorded Flood Outlines ingles: la, ausencia
de poligono nao e evidencia de nada (a propria Environment Agency adverte
isso). Aqui, a AOI declara o que foi examinado. E a diferenca entre silencio
e afirmacao.

NAO faz: nao promove negativo, nao extrai feature, nao treina. Baixa e
inventaria o pacote vetorial, e registra proveniencia.

Uso:
    python scripts/externo/n1e_cems_vetores.py
    python scripts/externo/n1e_cems_vetores.py --codigos EMSR720,EMSR783
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

from n1_common import BASE_OUT, baixar, escrever_proveniencia, gb, get_json, salvar_json

OUT = BASE_OUT / "n1e-cems-vetores"
INV = BASE_OUT / "n1c-cems" / "activations_raw.json"
API = "https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api"

# Alvo primario: enchente do Rio Grande do Sul, maio de 2024.
CODIGOS_PADRAO = ["EMSR720"]

# Bateria de endpoints candidatos. Nenhum e documentado publicamente; a API do
# CEMS expoe metadados de ativacao mas o link do pacote vetorial nao aparece na
# resposta de `public-activations-info`. Os tres primeiros ja falharam com
# HTTPError em 2026-08-07; os demais sao tentativas por convencao de Django
# REST. O modo --sondar existe para descobrir qual responde SEM gastar horas.
ENDPOINTS_PRODUTO = [
    "{api}/public-activations-info/{code}/",
    "{api}/public-activation-products/?activation={code}",
    "{api}/public-products-info/?activation={code}",
    "{api}/public-products-info/?activation_code={code}",
    "{api}/public-products-info/?code={code}",
    "{api}/public-activations-info/?code={code}",
    "{api}/public-activations-info/?search={code}",
    "{api}/public-aois-info/?activation={code}",
    "{api}/activations/{code}/products/",
    "{api}/products/?activation={code}",
    "https://rapidmapping.emergency.copernicus.eu/api/activations/{code}/",
    "https://rapidmapping.emergency.copernicus.eu/api/products/?activation={code}",
]

# Padroes de URL direta observados na nomenclatura dos pacotes do EMSR720.
# O zip baixado manualmente continha arquivos como
#   EMSR720_AOI01_DEL_PRODUCT_v1.zip
# entao o servidor guarda os pacotes por AOI/tipo/versao em algum prefixo.
PADROES_DIRETOS = [
    "https://rapidmapping.emergency.copernicus.eu/backend/download/{code}/products/",
    "https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api/download/{code}/",
    "https://emergency.copernicus.eu/mapping/download/{code}/products",
]


def achar_links_vetor(obj, achados: set) -> set:
    """
    Varre recursivamente qualquer JSON atras de link de pacote.

    Criterio deliberadamente largo: o pacote pode vir como `.zip` explicito ou
    como rota de download sem extensao. Filtrar demais aqui ja custou uma
    rodada -- a primeira versao so aceitava `.zip` no fim da string e voltou
    com zero links.
    """
    if isinstance(obj, dict):
        for v in obj.values():
            achar_links_vetor(v, achados)
    elif isinstance(obj, list):
        for v in obj:
            achar_links_vetor(v, achados)
    elif isinstance(obj, str) and obj.startswith("http"):
        if obj.lower().endswith(".zip") or "download" in obj.lower():
            achados.add(obj)
    return achados


# Endpoints confirmados por sondagem em 2026-08-07 (responderam com conteudo).
ENDPOINTS_CONFIRMADOS = [
    "https://rapidmapping.emergency.copernicus.eu/api/products/?activation={code}",
    "https://rapidmapping.emergency.copernicus.eu/api/activations/{code}/",
]


def coletar_links(code: str, outdir: Path) -> tuple[set, list]:
    """Consulta os endpoints confirmados, paginando, e junta os links achados."""
    links: set = set()
    diagnostico = []
    for tpl in ENDPOINTS_CONFIRMADOS + ENDPOINTS_PRODUTO:
        url = tpl.format(api=API, code=code)
        pagina = 0
        while url and pagina < 50:
            payload = get_json(url)
            if payload is None:
                diagnostico.append(f"{tpl}: sem JSON")
                break
            if pagina == 0:
                salvar_json(outdir / f"{code}_resp_{abs(hash(tpl)) % 10000}.json", payload)
                chaves = (list(payload.keys())[:8] if isinstance(payload, dict)
                          else f"lista[{len(payload)}]")
                diagnostico.append(f"{tpl}: {chaves}")
            antes = len(links)
            achar_links_vetor(payload, links)
            diagnostico.append(f"   pagina {pagina}: +{len(links)-antes} links")
            url = payload.get("next") if isinstance(payload, dict) else None
            pagina += 1
        if links:
            break   # primeiro endpoint que entrega link ja resolve
    return links, diagnostico


def modo_sondar(codigo: str) -> int:
    """
    Testa a bateria de endpoints e reporta status cru de cada um.

    Existe porque a descoberta de endpoint precisa acontecer na maquina que tem
    rede para o host -- o ambiente de analise nao alcanca este servidor. Sem
    isso, cada tentativa custaria um ciclo inteiro de ida e volta.
    """
    import requests

    print("===N1E_SONDAGEM===")
    print(f"CODIGO={codigo}")
    achou = []
    for tpl in ENDPOINTS_PRODUTO + PADROES_DIRETOS:
        url = tpl.format(api=API, code=codigo)
        try:
            r = requests.get(url, timeout=45, allow_redirects=True)
            tipo = r.headers.get("Content-Type", "")[:40]
            tam = len(r.content)
            marca = "OK " if r.status_code < 400 else "   "
            print(f"{marca}{r.status_code} {tam:>9}B {tipo:40s} {url[:95]}")
            if r.status_code < 400 and tam > 200:
                achou.append(url)
                if "json" in tipo:
                    try:
                        amostra = json.dumps(r.json())[:300]
                        print(f"      amostra: {amostra}")
                    except Exception:  # noqa: BLE001
                        pass
        except Exception as exc:  # noqa: BLE001
            print(f"    ERR {type(exc).__name__:24s} {url[:95]}")
    print(f"RESPONDERAM={len(achou)}")
    for u in achou:
        print(f"  {u}")
    print("===END===")
    return 0


def main() -> int:
    args = sys.argv[1:]
    codigos = CODIGOS_PADRAO
    if "--codigos" in args:
        codigos = args[args.index("--codigos") + 1].split(",")
    if "--sondar" in args:
        return modo_sondar(codigos[0])

    OUT.mkdir(parents=True, exist_ok=True)
    print("===N1E_CEMS_VETORES===")
    print(f"CODIGOS={codigos}")

    inventario = {}
    if INV.exists():
        raw = json.loads(INV.read_text(encoding="utf-8"))
        inventario = {r.get("code"): r for r in raw if isinstance(r, dict)}
        print(f"INVENTARIO_LOCAL={len(inventario)} ativacoes")

    relatorio = {}
    for code in codigos:
        meta = inventario.get(code, {})
        print(f"---{code}--- {str(meta.get('name'))[:60]} "
              f"| {str(meta.get('eventTime'))[:10]} | aois={meta.get('n_aois')} "
              f"| produtos={meta.get('n_products')}")
        salvar_json(OUT / f"{code}_meta.json", meta)

        links, diagnostico = coletar_links(code, OUT)
        for d in diagnostico[:12]:
            print(f"   {d}")
        print(f"   links_encontrados={len(links)}")
        for u in sorted(links)[:5]:
            print(f"     {u[:110]}")

        baixados, falhas = 0, 0
        for link in sorted(links):
            nome = link.rstrip("/").split("/")[-1]
            if baixar(link, OUT / code / nome):
                baixados += 1
            else:
                falhas += 1
        arquivos = sorted((OUT / code).glob("*")) if (OUT / code).exists() else []
        tam = sum(p.stat().st_size for p in arquivos if p.is_file())
        print(f"   baixados={baixados} falhas={falhas} total={gb(tam)}GB")
        relatorio[code] = {"meta": meta, "links": sorted(links),
                           "baixados": baixados, "falhas": falhas, "bytes": tam}

        if not links:
            print(f"   NENHUM LINK AUTOMATICO. Baixe manualmente em:")
            print(f"   https://rapidmapping.emergency.copernicus.eu/{code}")
            print(f"   e salve os pacotes vetoriais em {OUT / code}")

    salvar_json(OUT / "relatorio.json", relatorio)
    escrever_proveniencia(
        OUT,
        titulo="Copernicus EMS Rapid Mapping - produtos vetoriais",
        fonte="Copernicus Emergency Management Service (CEMS)",
        identificador=", ".join(codigos),
        licenca="Dados Copernicus - uso livre com atribuicao a European Union",
        natureza="OBSERVADO. AOI declara a area efetivamente analisada; a "
                 "camada de inundacao declara o que foi detectado dentro dela.",
        uso_pretendido="NEGATIVO POR OBSERVACAO no Brasil: dentro da AOI e "
                       "fora da mancha de inundacao. Ataca diretamente o gate "
                       "C4_BLOCKED_NO_FORMAL_NEGATIVES.",
        uso_proibido="Tratar AOI como area inundada; usar a AOI sem a camada "
                     "de inundacao correspondente; promover a negativo antes "
                     "da tarefa de adjudicacao.",
        extra="- Achado de 2026-08-07: a API publica cobre apenas EMSR656+.\n"
              "  Ativacoes historicas (ex.: EMSR147, EMSR150) exigem o portal\n"
              "  antigo e ficam como pendencia registrada.\n",
    )
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
